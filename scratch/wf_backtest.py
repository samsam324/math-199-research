"""
Honest net-of-cost walk-forward backtest of the pair mean-reversion strategy.

PRINCIPLED selection (OU half-life in a tradeable band, low residual variance),
strict walk-forward (6mo TRAIN -> 3mo TEST, step 3mo), static OLS hedge fit on
TRAIN only, z-score rule on TEST, costs on every position change.

Controls:
  (a) RANDOM pair selection (avg over seeds)
  (b) PHASE-RANDOMIZED surrogate prices (must give ~0)

Reports GROSS and NET at two cost levels, with HAC (Newey-West) Sharpe SE and
block-bootstrap CI, vs placebos.

Run synchronously:  python scratch/wf_backtest.py
"""
from __future__ import annotations
import os, sys, itertools, time
import numpy as np
import pandas as pd

np.seterr(all="ignore")
ROOT = r"C:\Users\jackw\Desktop\math-199-research"
DATA = os.path.join(ROOT, "data", "spot_1h")
UNIV = os.path.join(ROOT, "data", "l2_universe_top50.txt")

# ----------------------------- config -----------------------------
TRAIN_MONTHS = 6
TEST_MONTHS  = 3
STEP_MONTHS  = 3
N_PAIRS      = 10
HL_MIN, HL_MAX = 12.0, 336.0      # OU half-life tradeable band (hours)
Z_ENTER, Z_EXIT, Z_STOP = 2.0, 0.5, 4.0
Z_WIN = 0  # 0 => use TRAIN spread mean/std (static); set >0 for rolling
GROSS_NOTIONAL = 1.0              # dollar-neutral: $1 long leg, $1 short leg
HOURS_PER_YEAR = 24 * 365
# cost levels: per-leg fee+slippage in bps  -> round-trip 2 legs
COST_LEVELS = {
    "realistic_30bps_rt": dict(fee_bps=10.0, slip_bps=5.0),   # 15bps/leg => ~30bps rt 2-leg one-way change
    "low_2bps_rt":        dict(fee_bps=1.0,  slip_bps=1.0),
}
RANDOM_SEEDS = [0, 1, 2, 3, 4]
MIN_OBS_FRAC = 0.90               # require >=90% non-NaN coverage in a window
START_FLOOR = "2021-01-01"        # restrict to era with broad universe coverage

# ----------------------------- data load -----------------------------
def load_universe():
    syms = [s.strip() for s in open(UNIV) if s.strip()]
    closes = {}
    for s in syms:
        f = os.path.join(DATA, f"{s}.parquet")
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f, columns=["close"])
        s_close = df["close"].copy()
        s_close.index = pd.to_datetime(s_close.index, utc=True)
        s_close = s_close[~s_close.index.duplicated(keep="last")]
        closes[s] = s_close
    px = pd.DataFrame(closes).sort_index()
    # hourly grid, forward-fill small gaps (<=3h) only
    full = pd.date_range(px.index.min().floor("h"), px.index.max().ceil("h"), freq="h", tz="UTC")
    px = px.reindex(full)
    px = px[px.index >= pd.Timestamp(START_FLOOR, tz="UTC")]
    px = px.ffill(limit=3)
    # drop stablecoins (no spread to trade) & any all-nan
    for stub in ["USDCUSDT", "DAIUSDT"]:
        if stub in px.columns:
            px = px.drop(columns=stub)
    return px

# ----------------------------- OU / hedge helpers -----------------------------
def ou_half_life(spread: np.ndarray):
    """AR(1) fit on spread: d s_t = a + b*s_{t-1} + e. half-life = -ln2/ln(1+b)."""
    s = spread[~np.isnan(spread)]
    if len(s) < 100:
        return np.nan, np.nan
    s_lag = s[:-1]
    ds = np.diff(s)
    X = np.column_stack([np.ones_like(s_lag), s_lag])
    try:
        coef, *_ = np.linalg.lstsq(X, ds, rcond=None)
    except Exception:
        return np.nan, np.nan
    b = coef[1]
    if b >= 0 or b <= -2:        # not mean-reverting
        return np.nan, np.nan
    phi = 1.0 + b
    if phi <= 0 or phi >= 1:
        return np.nan, np.nan
    hl = -np.log(2.0) / np.log(phi)
    resid = ds - X @ coef
    return hl, float(np.std(resid))

def ols_hedge(y: np.ndarray, x: np.ndarray):
    """y = alpha + beta*x ; return alpha, beta on log-prices."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef[0], coef[1]

# ----------------------------- backtest one TEST window -----------------------------
def simulate_pair(logpx_test, sym_a, sym_b, alpha, beta, mu, sd, fee_bps, slip_bps, z_win=0):
    """Return hourly net PnL series and stats for one pair over the TEST window.
    Dollar-neutral: long $1 of A, short $beta-scaled... we use unit z-position with
    dollar-neutral legs. Position p in {-1,0,+1} on the spread (long spread = long A short B).
    PnL_t = p_{t-1} * (r_a - r_b) where r are simple log-returns of the two legs scaled
    to dollar-neutral unit notional. Costs charged on |p_t - p_{t-1}| position changes.
    """
    la = logpx_test[sym_a].values
    lb = logpx_test[sym_b].values
    spread = la - alpha - beta * lb
    if sd <= 0 or not np.isfinite(sd):
        return None
    if z_win and z_win > 0:
        # rolling in-TEST re-centering (causal, trailing window) -- robustness variant
        sp = pd.Series(spread)
        rmu = sp.rolling(z_win, min_periods=max(20, z_win//2)).mean()
        rsd = sp.rolling(z_win, min_periods=max(20, z_win//2)).std()
        z = ((sp - rmu) / rsd).values
    else:
        z = (spread - mu) / sd
    n = len(z)
    pos = np.zeros(n)
    cur = 0.0
    for t in range(n):
        zt = z[t]
        if not np.isfinite(zt):
            cur = 0.0  # flat if data missing
        elif abs(zt) >= Z_STOP:
            cur = 0.0
        elif cur == 0.0:
            if zt >= Z_ENTER:   cur = -1.0   # spread high -> short spread
            elif zt <= -Z_ENTER: cur = +1.0  # spread low  -> long spread
        else:
            if abs(zt) <= Z_EXIT:
                cur = 0.0
        pos[t] = cur
    # leg log-returns (NaN-safe: gaps -> 0 return, no spurious jumps)
    ra = np.diff(la, prepend=la[0]); rb = np.diff(lb, prepend=lb[0])
    ra = np.where(np.isfinite(ra), ra, 0.0)
    rb = np.where(np.isfinite(rb), rb, 0.0)
    # dollar-neutral spread return per unit position (long A, short B), unit notional each leg
    spread_ret = ra - rb
    pos_lag = np.roll(pos, 1); pos_lag[0] = 0.0
    gross_pnl = pos_lag * spread_ret
    # costs: each position change of size |dpos| trades both legs.
    # one-way cost per leg = fee+slip (bps). a change of |dpos|=1 trades $1 per leg *2 legs.
    dpos = np.abs(np.diff(pos, prepend=0.0))
    cost_per_unit_change = 2.0 * (fee_bps + slip_bps) / 1e4   # 2 legs, fractional
    cost = dpos * cost_per_unit_change
    net_pnl = gross_pnl - cost
    n_trades = int(np.sum(dpos > 0))
    turnover = float(np.sum(dpos))
    return dict(gross=gross_pnl, net=net_pnl, pos=pos, n_trades=n_trades,
                turnover=turnover, spread_ret=spread_ret)

# ----------------------------- selection -----------------------------
def select_pairs_ou(logpx_train, cols, n_pairs):
    """Rank all pairs by OU half-life in band with lowest residual std."""
    cands = []
    for a, b in itertools.combinations(cols, 2):
        ya = logpx_train[a].values; yb = logpx_train[b].values
        m = np.isfinite(ya) & np.isfinite(yb)
        if m.sum() < len(ya) * MIN_OBS_FRAC:
            continue
        alpha, beta = ols_hedge(ya[m], yb[m])
        if beta <= 0:           # require positive hedge (co-moving)
            continue
        spread = ya - alpha - beta * yb
        hl, rstd = ou_half_life(spread)
        if not np.isfinite(hl) or hl < HL_MIN or hl > HL_MAX:
            continue
        if not np.isfinite(rstd) or rstd <= 0:
            continue
        sd = np.nanstd(spread[m]); mu = np.nanmean(spread[m])
        if sd <= 0:
            continue
        # normalized residual speed: prefer short HL + low residual relative to spread sd
        score = hl  # primary: fastest reversion in band
        cands.append(dict(a=a, b=b, alpha=alpha, beta=beta, mu=mu, sd=sd,
                          hl=hl, rstd=rstd, nrstd=rstd/sd, score=score))
    cands.sort(key=lambda d: (d["score"], d["nrstd"]))  # short HL, then tight residual
    return cands[:n_pairs]

def select_pairs_random(logpx_train, cols, n_pairs, seed):
    rng = np.random.default_rng(seed)
    allp = list(itertools.combinations(cols, 2))
    rng.shuffle(allp)
    out = []
    for a, b in allp:
        ya = logpx_train[a].values; yb = logpx_train[b].values
        m = np.isfinite(ya) & np.isfinite(yb)
        if m.sum() < len(ya) * MIN_OBS_FRAC:
            continue
        alpha, beta = ols_hedge(ya[m], yb[m])
        spread = ya - alpha - beta * yb
        sd = np.nanstd(spread[m]); mu = np.nanmean(spread[m])
        if sd <= 0 or not np.isfinite(sd):
            continue
        out.append(dict(a=a, b=b, alpha=alpha, beta=beta, mu=mu, sd=sd))
        if len(out) >= n_pairs:
            break
    return out

# ----------------------------- surrogates -----------------------------
def phase_randomize(series: np.ndarray, rng):
    """Phase-randomized surrogate preserving power spectrum (FT phase scramble)."""
    x = series.copy()
    n = len(x)
    Xf = np.fft.rfft(x - x.mean())
    phases = np.angle(Xf)
    mag = np.abs(Xf)
    rand_ph = rng.uniform(0, 2*np.pi, size=len(phases))
    rand_ph[0] = phases[0]
    if n % 2 == 0:
        rand_ph[-1] = phases[-1]
    Xs = mag * np.exp(1j * rand_ph)
    xs = np.fft.irfft(Xs, n=n) + x.mean()
    return xs

def make_phase_random_px(logpx, rng):
    out = logpx.copy()
    for c in out.columns:
        v = out[c].values
        m = np.isfinite(v)
        if m.sum() < 50:
            continue
        vv = v.copy()
        vv[m] = phase_randomize(v[m], rng)
        out[c] = vv
    return out

# ----------------------------- stats -----------------------------
def sharpe_hac(pnl_hourly, lags=None):
    """Annualized Sharpe of an hourly PnL series + Newey-West HAC SE of the Sharpe."""
    x = np.asarray(pnl_hourly, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30 or np.std(x) == 0:
        return np.nan, np.nan, np.nan
    mu = x.mean(); sd = x.std(ddof=1)
    sharpe_h = mu / sd
    sharpe_ann = sharpe_h * np.sqrt(HOURS_PER_YEAR)
    if lags is None:
        lags = int(np.floor(4 * (n/100.0)**(2/9)))  # Newey-West rule of thumb
        lags = max(lags, 1)
    # HAC variance of the mean -> delta-method SE of Sharpe ~ accounting for autocorr
    # Var(sharpe_h) approx (1/n)*(1 + 0.5*sharpe_h^2) for iid; scale by HAC inflation of mean var.
    dem = x - mu
    gamma0 = np.mean(dem*dem)
    hac = gamma0
    for k in range(1, lags+1):
        w = 1.0 - k/(lags+1)
        cov = np.mean(dem[k:]*dem[:-k])
        hac += 2*w*cov
    inflation = hac/gamma0 if gamma0 > 0 else 1.0
    se_sharpe_h = np.sqrt((1 + 0.5*sharpe_h**2) * inflation / n)
    se_sharpe_ann = se_sharpe_h * np.sqrt(HOURS_PER_YEAR)
    return sharpe_ann, se_sharpe_ann, inflation

def block_bootstrap_sharpe_ci(pnl_hourly, block=168, n_boot=1000, seed=0):
    x = np.asarray(pnl_hourly, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < block*2 or np.std(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(n/block))
    starts_pool = np.arange(0, n-block+1)
    sh = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(starts_pool, size=nblocks)
        samp = np.concatenate([x[s:s+block] for s in idx])[:n]
        sd = samp.std(ddof=1)
        sh[b] = (samp.mean()/sd)*np.sqrt(HOURS_PER_YEAR) if sd > 0 else 0.0
    return (np.nanpercentile(sh, 2.5), np.nanpercentile(sh, 97.5))

def max_drawdown(equity):
    eq = np.cumsum(equity)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(dd.min())

# ----------------------------- walk-forward driver -----------------------------
def make_splits(index):
    t0 = index.min().normalize()
    tN = index.max()
    splits = []
    cur = t0
    while True:
        tr_s = cur
        tr_e = tr_s + pd.DateOffset(months=TRAIN_MONTHS)
        te_e = tr_e + pd.DateOffset(months=TEST_MONTHS)
        if te_e > tN:
            break
        splits.append((tr_s, tr_e, te_e))
        cur = cur + pd.DateOffset(months=STEP_MONTHS)
    return splits

def run_strategy(px, selector, cost_cfg, label, seed_for_random=None, z_win=0):
    """Run full walk-forward for a given selector & cost; return aggregated hourly net+gross."""
    logpx = np.log(px)
    cols = list(px.columns)
    splits = make_splits(px.index)
    fee = cost_cfg["fee_bps"]; slip = cost_cfg["slip_bps"]
    all_net = []; all_gross = []
    tot_trades = 0; tot_turn = 0.0
    wins = 0; losses = 0
    n_pair_windows = 0
    for (tr_s, tr_e, te_e) in splits:
        tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(tr) < 1000 or len(te) < 200:
            continue
        if selector == "ou":
            picks = select_pairs_ou(tr, cols, N_PAIRS)
        else:
            picks = select_pairs_random(tr, cols, N_PAIRS, seed_for_random)
        if not picks:
            continue
        # equal-weight portfolio of picked pairs over the TEST window
        net_mat = []; gross_mat = []
        for p in picks:
            r = simulate_pair(te, p["a"], p["b"], p["alpha"], p["beta"],
                              p["mu"], p["sd"], fee, slip, z_win=z_win)
            if r is None:
                continue
            net_mat.append(r["net"]); gross_mat.append(r["gross"])
            tot_trades += r["n_trades"]; tot_turn += r["turnover"]
            n_pair_windows += 1
            # per-trade win rate via realized round-trips on net
            pnl = r["net"]; pos = r["pos"]
            seg = 0.0; inpos = False
            for t in range(len(pos)):
                if pos[t] != 0:
                    seg += pnl[t]; inpos = True
                elif inpos:
                    (wins, losses) = (wins+1, losses) if seg > 0 else (wins, losses+1)
                    seg = 0.0; inpos = False
            if inpos:
                (wins, losses) = (wins+1, losses) if seg > 0 else (wins, losses+1)
        if not net_mat:
            continue
        # average across pairs (equal capital), align lengths
        L = min(len(v) for v in net_mat)
        net = np.mean([v[:L] for v in net_mat], axis=0)
        gross = np.mean([v[:L] for v in gross_mat], axis=0)
        all_net.append(net); all_gross.append(gross)
    if not all_net:
        return None
    net = np.concatenate(all_net); gross = np.concatenate(all_gross)
    sh_net, se_net, infl = sharpe_hac(net)
    sh_gross, se_gross, _ = sharpe_hac(gross)
    ci_net = block_bootstrap_sharpe_ci(net, seed=(seed_for_random or 0))
    wr = wins/(wins+losses) if (wins+losses) > 0 else np.nan
    return dict(label=label, n_hours=len(net),
                sharpe_net=sh_net, se_net=se_net, ci_net=ci_net,
                sharpe_gross=sh_gross, se_gross=se_gross,
                net_pnl=float(np.nansum(net)), gross_pnl=float(np.nansum(gross)),
                mdd_net=max_drawdown(np.nan_to_num(net)), n_trades=tot_trades,
                turnover=tot_turn, win_rate=wr, n_splits=len(splits),
                n_pair_windows=n_pair_windows)

def fmt(r):
    if r is None:
        return f"  {('NO RESULT'):<22}"
    ci = r["ci_net"]
    return (f"  net Sharpe={r['sharpe_net']:+.2f} (SE {r['se_net']:.2f}, "
            f"95%CI [{ci[0]:+.2f},{ci[1]:+.2f}])  "
            f"gross Sharpe={r['sharpe_gross']:+.2f} (SE {r['se_gross']:.2f})  "
            f"netPnL={r['net_pnl']*100:+.2f}%  grossPnL={r['gross_pnl']*100:+.2f}%  "
            f"MDD={r['mdd_net']*100:.2f}%  trades={r['n_trades']}  "
            f"turn={r['turnover']:.0f}  WR={r['win_rate']*100:.1f}%")

def main():
    t0 = time.time()
    px = load_universe()
    print("="*100)
    print("HONEST WALK-FORWARD PAIR MEAN-REVERSION BACKTEST")
    print(f"universe={px.shape[1]} tradable syms | hours={px.shape[0]} | "
          f"span={px.index.min().date()}..{px.index.max().date()}")
    splits = make_splits(px.index)
    print(f"walk-forward: {TRAIN_MONTHS}mo TRAIN -> {TEST_MONTHS}mo TEST, step {STEP_MONTHS}mo "
          f"=> {len(splits)} splits | N_PAIRS={N_PAIRS} | OU band {HL_MIN:.0f}-{HL_MAX:.0f}h | "
          f"z enter/exit/stop = {Z_ENTER}/{Z_EXIT}/{Z_STOP}")
    print("="*100)

    results = {}
    for clabel, ccfg in COST_LEVELS.items():
        print(f"\n########## COST LEVEL: {clabel}  (fee {ccfg['fee_bps']}bps + slip {ccfg['slip_bps']}bps per leg) ##########")

        # 1) OU-selected, static TRAIN mean/std (primary protocol)
        r_ou = run_strategy(px, "ou", ccfg, "OU-selected")
        print(f"\n[OU-SELECTED | static TRAIN z]"); print(fmt(r_ou))
        results[(clabel,"ou")] = r_ou

        # 1b) OU-selected, rolling in-TEST z (robustness vs static-mean drift)
        r_our = run_strategy(px, "ou", ccfg, "OU-rollingZ", z_win=336)
        print(f"\n[OU-SELECTED | rolling 336h in-TEST z]"); print(fmt(r_our))
        results[(clabel,"ou_roll")] = r_our

        # 2) RANDOM selection placebo, avg over seeds -- run under BOTH z rules
        for zw, tag in [(0, "static TRAIN z"), (336, "rolling 336h z")]:
            rand_runs = []
            for sd in RANDOM_SEEDS:
                rr = run_strategy(px, "random", ccfg, f"random-s{sd}", seed_for_random=sd, z_win=zw)
                if rr: rand_runs.append(rr)
            if rand_runs:
                avg_net = np.mean([r["sharpe_net"] for r in rand_runs])
                std_net = np.std([r["sharpe_net"] for r in rand_runs])
                avg_gross = np.mean([r["sharpe_gross"] for r in rand_runs])
                avg_netpnl = np.mean([r["net_pnl"] for r in rand_runs])
                per_seed = ', '.join('{:+.2f}'.format(r["sharpe_net"]) for r in rand_runs)
                print(f"\n[RANDOM-PAIR PLACEBO | {tag}]  (avg over {len(rand_runs)} seeds)")
                print(f"  net Sharpe={avg_net:+.2f} (across-seed SD {std_net:.2f})  "
                      f"gross Sharpe={avg_gross:+.2f}  netPnL={avg_netpnl*100:+.2f}%  "
                      f"per-seed net=[{per_seed}]")
                results[(clabel,f"random_z{zw}")] = dict(avg_net=avg_net, std_net=std_net, avg_gross=avg_gross)

    # 3) PHASE-RANDOMIZED surrogate (cost-independent test; use realistic costs)
    #    run under BOTH z rules: a real edge must give ~0 here even though the
    #    rolling-z rule mechanically reverts on ANY series (the known artifact).
    print(f"\n########## PHASE-RANDOMIZED SURROGATE PLACEBO (realistic costs) ##########")
    ccfg = COST_LEVELS["realistic_30bps_rt"]
    for zw, tag in [(0, "static TRAIN z"), (336, "rolling 336h z")]:
        surr_runs = []
        for sd in RANDOM_SEEDS[:3]:
            rng = np.random.default_rng(1000+sd)
            spx = np.exp(make_phase_random_px(np.log(px), rng))
            rr = run_strategy(spx, "ou", ccfg, f"surrogate-ou-s{sd}", z_win=zw)
            if rr: surr_runs.append(rr)
        if surr_runs:
            avg_net = np.mean([r["sharpe_net"] for r in surr_runs])
            avg_gross = np.mean([r["sharpe_gross"] for r in surr_runs])
            per_seed_s = ', '.join('{:+.2f}'.format(r["sharpe_net"]) for r in surr_runs)
            print(f"\n  [{tag}] OU-selection on phase-randomized prices (should be ~0 if edge is real):")
            print(f"  net Sharpe={avg_net:+.2f}  gross Sharpe={avg_gross:+.2f}  "
                  f"per-seed net=[{per_seed_s}]")

    print("\n" + "="*100)
    print(f"DONE in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
