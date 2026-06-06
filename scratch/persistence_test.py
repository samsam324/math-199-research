"""
PERSISTENCE OF MEAN-REVERSION OUT-OF-SAMPLE (the precondition for the strategy).

Question: if a pair reverts in-sample, does it revert out-of-sample? I.e. is
mean-reversion a SELECTABLE property, or noise that doesn't persist?

Design (all NET of the rolling-z mechanical floor):
  For each C(50,2) pair, on each walk-forward split (6mo train -> 3mo test):
    TRAIN window:
      - static OLS hedge  spread = log(p_a) - (a + b*log(p_b))   [b frozen for test]
      - 3 train metrics of reversion strength:
          (i)  OU/AR(1): kappa (mean-reversion speed), half-life
          (ii) variance ratio VR(k)  (VR<1 => mean-reverting)
          (iii) excess |z|>2 reversion over a random-walk mechanical floor
    TEST window (disjoint future):
      - SAME frozen hedge ratio (no re-fit -> honest OOS)
      - rolling z (lookback 240h, no look-ahead), |z|>2 events w/ cooldown
      - OOS reversion = observed mean reversion  MINUS  mechanical-floor reversion
        where the floor = same event/measurement machinery applied to a matched
        random-walk surrogate spread (no mean-reversion by construction).
      - also OOS OU kappa on the test spread.

  PERSISTENCE:
    (a) Spearman rank-corr(train metric, OOS excess reversion) pooled across splits
    (b) top-quintile vs bottom-quintile OOS excess-reversion spread, block-boot SE
    (c) PLACEBO: random quintile assignment -> spread ~ 0

Mechanical floor is the crux: a rolling z mechanically reverts even for a random
walk, so ALL reversion is reported as EXCESS over a per-pair, per-window matched
random-walk surrogate.
"""
from __future__ import annotations
import sys, time
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy.stats import spearmanr

RNG = np.random.default_rng(20260605)

UNI = [l.strip() for l in open(ROOT / "data/l2_universe_top50.txt") if l.strip()]
LOOKBACK = 240        # rolling z window (hours) ~ 10 days
ZTHR = 2.0
HORIZON = 48          # primary OOS reversion horizon (hours) ~ matches half-life~48h
COOLDOWN = 48         # min hours between accepted events per pair
N_SURR = 20           # random-walk surrogates per pair-window for the mechanical floor
TRAIN_H = 24 * 182    # ~6 months
TEST_H = 24 * 91      # ~3 months
MIN_EVENTS = 3        # min OOS events for a pair to be scored in a split


# ----------------------------- data -----------------------------------------
def load_close(sym):
    df = pd.read_parquet(ROOT / f"data/spot_1h/{sym}.parquet")
    return df["close"]


def build_panel(symbols):
    cols = {}
    for s in symbols:
        try:
            cols[s] = load_close(s)
        except Exception:
            pass
    panel = pd.DataFrame(cols).sort_index()
    # forward-fill tiny gaps then require full rows for the common window
    panel = panel.dropna(how="any")
    return panel


# ------------------------- reversion machinery -------------------------------
def rolling_z(spread):
    """Vectorized rolling mean/std z-score (lookback=LOOKBACK), no look-ahead.
    z[t] uses window [t-LOOKBACK+1 .. t]."""
    x = np.asarray(spread, float)
    n = len(x)
    z = np.full(n, np.nan)
    if n < LOOKBACK:
        return z
    c = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x * x, 0, 0.0))
    # window sums for indices t = LOOKBACK-1 .. n-1
    idx = np.arange(LOOKBACK - 1, n)
    s1 = c[idx + 1] - c[idx + 1 - LOOKBACK]
    s2 = c2[idx + 1] - c2[idx + 1 - LOOKBACK]
    mean = s1 / LOOKBACK
    var = s2 / LOOKBACK - mean * mean
    var = np.where(var < 0, 0.0, var) * (LOOKBACK / (LOOKBACK - 1))  # sample var
    sd = np.sqrt(var)
    with np.errstate(invalid="ignore", divide="ignore"):
        z[idx] = (x[idx] - mean) / sd
    return z


def event_reversion(zv, horizon=HORIZON, cooldown=COOLDOWN):
    """Mean signed reversion over |z|>2 events. signed reversion =
    -sign(z_entry)*(z_{t+h}-z_entry): positive => moved toward 0."""
    n = len(zv)
    revs = []
    last = -10**9
    for t in range(LOOKBACK, n - horizon):
        zt = zv[t]
        if not np.isfinite(zt) or abs(zt) <= ZTHR:
            continue
        if t - last < cooldown:
            continue
        last = t
        zf = zv[t + horizon]
        if np.isfinite(zf):
            revs.append(-np.sign(zt) * (zf - zt))
    return np.array(revs)


def ou_kappa_halflife(spread):
    """AR(1): s_t = c + phi*s_{t-1} + e. kappa = -ln(phi); half-life = ln2/kappa.
    Returns (kappa, halflife). kappa>0 => mean-reverting."""
    s = np.asarray(spread, float)
    s = s[np.isfinite(s)]
    if len(s) < 50:
        return np.nan, np.nan
    y = s[1:]; x = s[:-1]
    try:
        res = OLS(y, add_constant(x)).fit()
        phi = float(res.params[1])
    except Exception:
        return np.nan, np.nan
    if phi <= 0 or phi >= 1:
        # phi<=0 (over-reverting/noise) -> very fast; phi>=1 -> not reverting
        if phi >= 1:
            return 0.0, np.inf
        # treat phi<=0 as fast reversion
        kappa = -np.log(max(phi, 1e-6)) if phi > 0 else 5.0
        return kappa, np.log(2) / kappa if kappa > 0 else np.inf
    kappa = -np.log(phi)
    hl = np.log(2) / kappa
    return kappa, hl


def variance_ratio(spread, k=24):
    """VR(k) on the spread LEVELS' increments. For a mean-reverting series the
    k-step variance grows slower than k*1-step -> VR<1."""
    s = np.asarray(spread, float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 5 * k:
        return np.nan
    d1 = np.diff(s)
    var1 = np.var(d1, ddof=1)
    dk = s[k:] - s[:-k]
    vark = np.var(dk, ddof=1)
    if var1 <= 0:
        return np.nan
    return vark / (k * var1)


def matched_rw_spread(spread, rng):
    """Random-walk surrogate matched to the spread's 1-step increment vol.
    Has NO mean-reversion by construction; same length. Used for mechanical
    floor of the rolling-z reversion."""
    s = np.asarray(spread, float)
    s = s[np.isfinite(s)]
    d = np.diff(s)
    sd = np.std(d)
    n = len(spread)
    steps = rng.normal(0.0, sd, size=n)
    return np.cumsum(steps)


def mechanical_floor_reversion(spread, n_surr=N_SURR, horizon=HORIZON):
    """Mean event-reversion expected from a matched random walk (the rolling-z
    mechanical floor). Average over n_surr surrogates."""
    means = []
    for _ in range(n_surr):
        rw = matched_rw_spread(spread, RNG)
        zv = rolling_z(rw)
        r = event_reversion(zv, horizon=horizon)
        if len(r):
            means.append(np.mean(r))
    return np.mean(means) if means else np.nan


# ------------------------- per-split computation -----------------------------
def compute_split(panel, pairs, tr0, tr1, te0, te1, split_id):
    """Returns DataFrame: per pair, train metrics + OOS excess reversion + OOS kappa."""
    logp = np.log(panel)
    train = logp.iloc[tr0:tr1]
    test = logp.iloc[te0:te1]
    rows = []
    for a, b in pairs:
        ytr = train[a].to_numpy(); xtr = train[b].to_numpy()
        if not (np.all(np.isfinite(ytr)) and np.all(np.isfinite(xtr))):
            continue
        # static OLS hedge on TRAIN; freeze a,b
        try:
            res = OLS(ytr, add_constant(xtr)).fit()
            a0, b0 = float(res.params[0]), float(res.params[1])
        except Exception:
            continue
        sp_tr = ytr - (a0 + b0 * xtr)

        # --- TRAIN metrics ---
        kap_tr, hl_tr = ou_kappa_halflife(sp_tr)
        vr_tr = variance_ratio(sp_tr, k=24)
        z_tr = rolling_z(sp_tr)
        obs_tr = event_reversion(z_tr)
        floor_tr = mechanical_floor_reversion(sp_tr)
        excess_tr = (np.mean(obs_tr) - floor_tr) if len(obs_tr) else np.nan
        n_ev_tr = len(obs_tr)

        # --- TEST (OOS), SAME frozen hedge ---
        yte = test[a].to_numpy(); xte = test[b].to_numpy()
        if not (np.all(np.isfinite(yte)) and np.all(np.isfinite(xte))):
            continue
        sp_te = yte - (a0 + b0 * xte)
        kap_te, hl_te = ou_kappa_halflife(sp_te)
        z_te = rolling_z(sp_te)
        obs_te = event_reversion(z_te)
        floor_te = mechanical_floor_reversion(sp_te)
        n_ev_te = len(obs_te)
        excess_te = (np.mean(obs_te) - floor_te) if n_ev_te >= MIN_EVENTS else np.nan

        rows.append(dict(
            split=split_id, pair=f"{a}_{b}",
            # train metrics (selection signals)
            tr_kappa=kap_tr, tr_halflife=hl_tr, tr_vr=vr_tr,
            tr_excess=excess_tr, tr_nev=n_ev_tr,
            # OOS outcomes
            oos_excess=excess_te, oos_kappa=kap_te, oos_nev=n_ev_te,
            oos_floor=floor_te, oos_obs=(np.mean(obs_te) if n_ev_te else np.nan),
        ))
    return pd.DataFrame(rows)


# ------------------------- pooled inference ----------------------------------
def block_boot_quintile_spread(df, metric_col, n_boot=2000):
    """Within each split, rank pairs by metric_col, form top/bottom quintile by
    OOS excess reversion; spread = mean(top oos_excess) - mean(bottom oos_excess).
    Pool across splits. Block-bootstrap over (split) clusters for SE."""
    splits = sorted(df["split"].unique())
    per_split = []
    for sp in splits:
        d = df[df["split"] == sp].dropna(subset=[metric_col, "oos_excess"])
        if len(d) < 10:
            continue
        d = d.copy()
        # higher metric = more mean-reverting. For VR and halflife, LOWER is better
        asc = metric_col in ("tr_vr", "tr_halflife")
        d["rank"] = d[metric_col].rank(ascending=asc)  # rank 1 = best when asc for vr/hl
        # define top = most mean-reverting in-sample
        nq = max(1, len(d) // 5)
        if asc:
            top = d.nsmallest(nq, metric_col)
            bot = d.nlargest(nq, metric_col)
        else:
            top = d.nlargest(nq, metric_col)
            bot = d.nsmallest(nq, metric_col)
        per_split.append((np.mean(top["oos_excess"]), np.mean(bot["oos_excess"]),
                          np.mean(top["oos_excess"]) - np.mean(bot["oos_excess"])))
    if not per_split:
        return None
    arr = np.array(per_split)  # rows: top, bot, spread per split
    point_spread = np.mean(arr[:, 2])
    point_top = np.mean(arr[:, 0]); point_bot = np.mean(arr[:, 1])
    # bootstrap over splits (cluster)
    boots = np.empty(n_boot)
    nS = arr.shape[0]
    for i in range(n_boot):
        idx = RNG.integers(0, nS, size=nS)
        boots[i] = np.mean(arr[idx, 2])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return dict(spread=point_spread, top=point_top, bot=point_bot,
                lo=lo, hi=hi, p=p, n_splits=nS)


def pooled_spearman(df, metric_col):
    """Spearman within each split, then average (Fisher-z) + per-split list."""
    splits = sorted(df["split"].unique())
    rhos = []
    for sp in splits:
        d = df[df["split"] == sp].dropna(subset=[metric_col, "oos_excess"])
        if len(d) < 10:
            continue
        rho, _ = spearmanr(d[metric_col], d["oos_excess"])
        if np.isfinite(rho):
            rhos.append(rho)
    if not rhos:
        return None
    rhos = np.array(rhos)
    # bootstrap mean rho over splits
    nS = len(rhos)
    boots = np.array([np.mean(rhos[RNG.integers(0, nS, nS)]) for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return dict(mean_rho=np.mean(rhos), lo=lo, hi=hi, p=p,
                per_split=rhos.tolist(), n_splits=nS)


def placebo_quintile_spread(df, n_perm=500):
    """Random quintile assignment within each split -> distribution of OOS
    excess-reversion spread. Should center on 0."""
    splits = sorted(df["split"].unique())
    spreads = []
    for _ in range(n_perm):
        per_split = []
        for sp in splits:
            d = df[df["split"] == sp].dropna(subset=["oos_excess"])
            if len(d) < 10:
                continue
            vals = d["oos_excess"].to_numpy()
            perm = RNG.permutation(vals)
            nq = max(1, len(vals) // 5)
            per_split.append(np.mean(perm[:nq]) - np.mean(perm[-nq:]))
        if per_split:
            spreads.append(np.mean(per_split))
    spreads = np.array(spreads)
    return dict(mean=np.mean(spreads), sd=np.std(spreads),
                lo=np.percentile(spreads, 2.5), hi=np.percentile(spreads, 97.5))


# ------------------------------ main -----------------------------------------
def main():
    t0 = time.time()
    panel = build_panel(UNI)
    n = panel.shape[0]
    print(f"Panel: {n} bars x {panel.shape[1]} syms, "
          f"{panel.index.min()} .. {panel.index.max()}", flush=True)

    pairs = list(combinations(list(panel.columns), 2))
    print(f"Pairs: {len(pairs)}  (C({panel.shape[1]},2))", flush=True)

    # walk-forward splits over the common window
    splits = []
    start = 0
    sid = 0
    while start + TRAIN_H + TEST_H <= n:
        tr0, tr1 = start, start + TRAIN_H
        te0, te1 = tr1, tr1 + TEST_H
        splits.append((sid, tr0, tr1, te0, te1))
        start += TEST_H   # roll forward by the test length (disjoint tests)
        sid += 1
    print(f"Walk-forward splits: {len(splits)} "
          f"(train={TRAIN_H}h ~6mo, test={TEST_H}h ~3mo, roll={TEST_H}h)", flush=True)
    for sid, tr0, tr1, te0, te1 in splits:
        print(f"  split {sid}: train {panel.index[tr0].date()}..{panel.index[tr1-1].date()}"
              f"  test {panel.index[te0].date()}..{panel.index[te1-1].date()}", flush=True)

    all_rows = []
    for sid, tr0, tr1, te0, te1 in splits:
        d = compute_split(panel, pairs, tr0, tr1, te0, te1, sid)
        scored = d["oos_excess"].notna().sum()
        print(f"  [split {sid}] pairs scored OOS (>= {MIN_EVENTS} events): {scored}/{len(d)}"
              f"   ({time.time()-t0:.0f}s)", flush=True)
        all_rows.append(d)
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(ROOT / "scratch/persistence_pairs.csv", index=False)
    print(f"\nSaved per-pair-per-split table -> scratch/persistence_pairs.csv "
          f"({len(df)} rows)", flush=True)

    # =================== REPORT ===================
    print("\n" + "=" * 78, flush=True)
    print("OOS PERSISTENCE OF MEAN-REVERSION (all reversion NET of mechanical floor)", flush=True)
    print("=" * 78, flush=True)

    # sanity: how big is the mechanical floor vs observed, pooled
    valid = df.dropna(subset=["oos_excess"])
    print(f"\nPooled OOS (pairs w/ >= {MIN_EVENTS} events): n={len(valid)} pair-splits", flush=True)
    print(f"  mean observed OOS reversion : {np.nanmean(df['oos_obs']):+.4f} z", flush=True)
    print(f"  mean mechanical floor       : {np.nanmean(df['oos_floor']):+.4f} z", flush=True)
    print(f"  mean EXCESS (obs - floor)   : {np.nanmean(valid['oos_excess']):+.4f} z", flush=True)

    metrics = {
        "tr_kappa  (OU speed, higher=revert)": "tr_kappa",
        "tr_halflife (lower=revert)":          "tr_halflife",
        "tr_vr     (VR<1=revert, lower better)":"tr_vr",
        "tr_excess (in-sample excess revert)": "tr_excess",
    }

    print("\n--- (a) Spearman rank-corr(train metric, OOS excess reversion), pooled over splits ---", flush=True)
    for label, col in metrics.items():
        r = pooled_spearman(df, col)
        if r is None:
            print(f"  {label:42s}: insufficient data", flush=True)
            continue
        sig = "SIG" if r["p"] < 0.05 else "ns"
        ps = ",".join(f"{x:+.2f}" for x in r["per_split"])
        print(f"  {label:42s}: rho={r['mean_rho']:+.3f}  95%CI[{r['lo']:+.3f},{r['hi']:+.3f}]"
              f"  p={r['p']:.3f} {sig}  (per-split: {ps})", flush=True)

    print("\n--- (b) Top-quintile vs bottom-quintile OOS excess reversion (block-boot over splits) ---", flush=True)
    for label, col in metrics.items():
        r = block_boot_quintile_spread(df, col)
        if r is None:
            print(f"  {label:42s}: insufficient data", flush=True)
            continue
        sig = "SIG" if r["p"] < 0.05 else "ns"
        print(f"  {label:42s}: top={r['top']:+.4f} bot={r['bot']:+.4f}  "
              f"SPREAD={r['spread']:+.4f}z 95%CI[{r['lo']:+.4f},{r['hi']:+.4f}] p={r['p']:.3f} {sig}"
              f"  (n_splits={r['n_splits']})", flush=True)

    print("\n--- (c) PLACEBO: random quintile assignment (should be ~0) ---", flush=True)
    pl = placebo_quintile_spread(df)
    print(f"  random-quintile spread: mean={pl['mean']:+.4f}z  sd={pl['sd']:.4f}"
          f"  95%CI[{pl['lo']:+.4f},{pl['hi']:+.4f}]", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
