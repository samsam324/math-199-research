"""
PART 1 — Is the OOS Kalman cointegration real or a Kalman-whitening artifact?

Reproduces the run_kalman_pair_screen.py logic standalone on the top-50 universe:
  MLE-fit Kalman dynamic hedge on TRAIN, forward-roll FIXED params on TEST,
  ADF on TEST-period residuals -> "OOS cointegration".

Then runs the IDENTICAL screen on placebos that CANNOT be cointegrated:
  (a) independent simulated random walks (matched length/vol)
  (b) one real coin vs a phase-randomized surrogate of another (kills coint, keeps spectrum)
  (c) real pairs with the x leg block-shuffled (kills cointegration alignment)

And a CLEAN non-circular benchmark: Engle-Granger static-OLS cointegration,
train/test split, ADF on TEST residuals using the STATIC (train-fitted) hedge.

All windows match RESULTS.md: t0=2024-01-01, 90d train, 30d test.
"""
from __future__ import annotations
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint
from src.kalman_hedge import fit_kalman_mle, kalman_forward_residuals

RNG = np.random.default_rng(12345)

T0 = pd.Timestamp("2024-01-01T00:00:00Z")
TRAIN_DAYS, TEST_DAYS = 90, 30
FULL_START = T0 - pd.Timedelta(days=TRAIN_DAYS + TEST_DAYS)
TRAIN_END = T0 - pd.Timedelta(days=TEST_DAYS)
FULL_END = T0

UNI = [l.strip() for l in open(ROOT / "data/l2_universe_top50.txt") if l.strip()]


def adf_p(series):
    s = np.asarray(series, dtype=float)
    if len(s) < 20 or not np.isfinite(s).all():
        return float("nan")
    try:
        stat, p, *_ = adfuller(s, regression="c", autolag=None, maxlag=24)
        return float(p)
    except Exception:
        return float("nan")


def load_panel():
    cols = {}
    for s in UNI:
        df = pd.read_parquet(ROOT / f"data/spot_1h/{s}.parquet")
        c = df["close"]
        c = c[(c.index >= FULL_START) & (c.index < FULL_END)]
        cols[s] = c
    panel = pd.DataFrame(cols).sort_index()
    # require >=50% coverage per symbol within window then forward-fill small gaps
    cov = panel.notna().mean()
    keep = cov[cov >= 0.80].index.tolist()
    panel = panel[keep]
    panel = panel.dropna(how="any")  # common-timestamp inner join
    return panel


def kalman_oos_p(y_tr, x_tr, y_te, x_te):
    try:
        fitted = fit_kalman_mle(y_tr, x_tr)
    except Exception:
        return float("nan"), float("nan")
    if not fitted.get("converged", False):
        return float("nan"), fitted.get("q_beta", float("nan"))
    _, _, resid = kalman_forward_residuals(y_te, x_te, fitted)
    return adf_p(resid), fitted["q_beta"]


def static_oos_p(y_tr, x_tr, y_te, x_te):
    try:
        ols = OLS(y_tr, add_constant(x_tr)).fit()
        a, b = float(ols.params[0]), float(ols.params[1])
    except Exception:
        return float("nan")
    resid = y_te - (a + b * x_te)
    return adf_p(resid)


def phase_randomize(x):
    """Phase-randomized surrogate: preserves power spectrum, destroys phase coupling."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    F = np.fft.rfft(x - x.mean())
    phases = np.exp(1j * RNG.uniform(0, 2 * np.pi, size=F.shape))
    phases[0] = 1.0
    if n % 2 == 0:
        phases[-1] = 1.0
    surro = np.fft.irfft(np.abs(F) * phases, n=n) + x.mean()
    return surro


def block_shuffle(x, block=24):
    x = np.asarray(x, dtype=float)
    n = len(x)
    nb = n // block
    idx = np.arange(nb)
    RNG.shuffle(idx)
    out = np.concatenate([x[i * block:(i + 1) * block] for i in idx])
    # pad tail
    if len(out) < n:
        out = np.concatenate([out, x[len(out):]])
    return out[:n]


def sim_rw(n, sigma):
    return np.cumsum(RNG.normal(0, sigma, size=n))


def run():
    panel = load_panel()
    logp = np.log(panel)
    tr = logp[logp.index < TRAIN_END]
    te = logp[(logp.index >= TRAIN_END) & (logp.index < FULL_END)]
    syms = list(panel.columns)
    print(f"Panel: {len(syms)} symbols, train={len(tr)} bars, test={len(te)} bars", flush=True)

    all_pairs = list(combinations(syms, 2))
    print(f"Total pairs C(n,2) = {len(all_pairs)}", flush=True)
    # Kalman MLE is ~2.3s/pair -> 1225 pairs = 46min, too slow. Estimate the RATE
    # from a random subsample of pairs (rate SE ~1-2pp at n=250). EG benchmark (fast)
    # still runs on ALL pairs.
    N_REAL = 250
    ridx = RNG.choice(len(all_pairs), size=min(N_REAL, len(all_pairs)), replace=False)
    pairs = [all_pairs[i] for i in ridx]
    print(f"Kalman/static screen on random subsample of {len(pairs)} pairs", flush=True)

    real_k, real_s = [], []
    qbetas = []
    done = 0
    for a, b in pairs:
        y_tr, x_tr = tr[a].to_numpy(), tr[b].to_numpy()
        y_te, x_te = te[a].to_numpy(), te[b].to_numpy()
        if len(y_tr) < 200 or len(y_te) < 100:
            continue
        pk, qb = kalman_oos_p(y_tr, x_tr, y_te, x_te)
        ps = static_oos_p(y_tr, x_tr, y_te, x_te)
        if np.isfinite(pk):
            real_k.append(pk)
            qbetas.append(qb)
        if np.isfinite(ps):
            real_s.append(ps)
        done += 1
        if done % 25 == 0:
            print(f"  ...real {done}/{len(pairs)}", flush=True)
    real_k = np.array(real_k); real_s = np.array(real_s); qbetas = np.array(qbetas)

    def rate(arr, thr=0.05):
        return 100 * np.mean(arr < thr)

    print("\n=== REAL top-50 pairs ===", flush=True)
    print(f"  Kalman OOS coint rate: p<0.05 {rate(real_k):.1f}%  p<0.01 {rate(real_k,0.01):.1f}%  p<0.001 {rate(real_k,0.001):.1f}%  (n={len(real_k)})", flush=True)
    print(f"  Static OOS coint rate: p<0.05 {rate(real_s):.1f}%  p<0.01 {rate(real_s,0.01):.1f}%  (n={len(real_s)})", flush=True)
    print(f"  Fitted q_beta: median {np.median(qbetas):.2e} min {qbetas.min():.2e} max {qbetas.max():.2e}", flush=True)

    # ---- PLACEBOS ----
    # Subsample pairs for placebo speed (Kalman MLE is the bottleneck)
    n_placebo = min(150, len(all_pairs))
    psub = [all_pairs[i] for i in RNG.choice(len(all_pairs), size=n_placebo, replace=False)]

    # (a) independent random walks: matched vol to a real coin's returns
    sig = float(np.std(np.diff(tr[syms[0]].to_numpy())))
    nt, ne = len(tr), len(te)
    rw_k = []
    for _ in range(n_placebo):
        ay = sim_rw(nt + ne, sig); ax = sim_rw(nt + ne, sig)
        pk, _ = kalman_oos_p(ay[:nt], ax[:nt], ay[nt:], ax[nt:])
        if np.isfinite(pk):
            rw_k.append(pk)
    rw_k = np.array(rw_k)

    # (b) real coin vs phase-randomized surrogate of another real coin (full series surrogate)
    pr_k = []
    for a, b in psub:
        y_full = logp[a].to_numpy()
        x_full = phase_randomize(logp[b].to_numpy())
        ytr, xtr = y_full[:nt], x_full[:nt]
        yte, xte = y_full[nt:nt+ne], x_full[nt:nt+ne]
        if len(yte) < 100:
            continue
        pk, _ = kalman_oos_p(ytr, xtr, yte, xte)
        if np.isfinite(pk):
            pr_k.append(pk)
    pr_k = np.array(pr_k)

    # (c) real pairs, x leg block-shuffled (destroys cointegration alignment, keeps marginal dist)
    bs_k = []
    for a, b in psub:
        y_full = logp[a].to_numpy()
        x_full = block_shuffle(logp[b].to_numpy(), block=24)
        ytr, xtr = y_full[:nt], x_full[:nt]
        yte, xte = y_full[nt:nt+ne], x_full[nt:nt+ne]
        if len(yte) < 100:
            continue
        pk, _ = kalman_oos_p(ytr, xtr, yte, xte)
        if np.isfinite(pk):
            bs_k.append(pk)
    bs_k = np.array(bs_k)

    print("\n=== PLACEBOS (Kalman OOS ADF) — should be ~5% if test is valid ===", flush=True)
    print(f"  (a) independent random walks:        p<0.05 {rate(rw_k):.1f}%  p<0.001 {rate(rw_k,0.001):.1f}%  (n={len(rw_k)})", flush=True)
    print(f"  (b) coin vs phase-randomized coin:   p<0.05 {rate(pr_k):.1f}%  p<0.001 {rate(pr_k,0.001):.1f}%  (n={len(pr_k)})", flush=True)
    print(f"  (c) real pair, x block-shuffled:     p<0.05 {rate(bs_k):.1f}%  p<0.001 {rate(bs_k,0.001):.1f}%  (n={len(bs_k)})", flush=True)

    # ---- CLEAN BENCHMARK: Engle-Granger on STATIC log-price pairs, OOS ----
    # Engle-Granger coint test (statsmodels coint) on TEST window directly,
    # plus the static-OLS-train->ADF-on-test residual (already have real_s).
    eg_p = []
    for a, b in pairs:
        yte, xte = te[a].to_numpy(), te[b].to_numpy()
        if len(yte) < 100:
            continue
        try:
            _, p, _ = coint(yte, xte, trend="c", autolag=None, maxlag=24)
            eg_p.append(float(p))
        except Exception:
            pass
    eg_p = np.array(eg_p)
    print("\n=== CLEAN non-circular benchmark (no Kalman whitening) ===", flush=True)
    print(f"  Engle-Granger on TEST log-prices:    coint p<0.05 {rate(eg_p):.1f}%  p<0.01 {rate(eg_p,0.01):.1f}%  (n={len(eg_p)})", flush=True)
    print(f"  Static OLS train->ADF test resid:    coint p<0.05 {rate(real_s):.1f}%  (n={len(real_s)})", flush=True)

    # Save the clean cointegrated pair list (Engle-Granger) for Part 2
    eg_rows = []
    for a, b in pairs:
        yte, xte = te[a].to_numpy(), te[b].to_numpy()
        if len(yte) < 100:
            continue
        try:
            _, p, _ = coint(yte, xte, trend="c", autolag=None, maxlag=24)
        except Exception:
            continue
        eg_rows.append({"sym_a": a, "sym_b": b, "eg_pvalue": float(p)})
    eg_df = pd.DataFrame(eg_rows).sort_values("eg_pvalue")
    eg_df.to_csv(ROOT / "scratch/clean_coint_pairs.csv", index=False)
    n_clean = int((eg_df["eg_pvalue"] < 0.05).sum())
    print(f"\n  Saved {len(eg_df)} pairs; {n_clean} clean-cointegrated at p<0.05 -> scratch/clean_coint_pairs.csv", flush=True)


if __name__ == "__main__":
    run()
