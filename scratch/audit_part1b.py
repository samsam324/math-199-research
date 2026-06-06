"""Part 1b: placebos + clean EG benchmark only (real screen already done in part1.log)."""
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
from statsmodels.tsa.stattools import adfuller, coint
from src.kalman_hedge import fit_kalman_mle, kalman_forward_residuals

RNG = np.random.default_rng(2024)
T0 = pd.Timestamp("2024-01-01T00:00:00Z")
TRAIN_DAYS, TEST_DAYS = 90, 30
FULL_START = T0 - pd.Timedelta(days=TRAIN_DAYS + TEST_DAYS)
TRAIN_END = T0 - pd.Timedelta(days=TEST_DAYS)
FULL_END = T0
UNI = [l.strip() for l in open(ROOT / "data/l2_universe_top50.txt") if l.strip()]
LOG = open(ROOT / "scratch/part1b.log", "w")
def out(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


def adf_p(s):
    s = np.asarray(s, float)
    if len(s) < 20 or not np.isfinite(s).all():
        return float("nan")
    try:
        st, p, *_ = adfuller(s, regression="c", autolag=None, maxlag=24)
        return float(p)
    except Exception:
        return float("nan")


def load_panel():
    cols = {}
    for s in UNI:
        c = pd.read_parquet(ROOT / f"data/spot_1h/{s}.parquet")["close"]
        c = c[(c.index >= FULL_START) & (c.index < FULL_END)]
        cols[s] = c
    panel = pd.DataFrame(cols).sort_index()
    cov = panel.notna().mean()
    panel = panel[cov[cov >= 0.80].index.tolist()].dropna(how="any")
    return panel


def kp(y_tr, x_tr, y_te, x_te):
    try:
        f = fit_kalman_mle(y_tr, x_tr)
    except Exception:
        return float("nan")
    _, _, r = kalman_forward_residuals(y_te, x_te, f)
    return adf_p(r)


def phase_randomize(x):
    x = np.asarray(x, float); n = len(x)
    F = np.fft.rfft(x - x.mean())
    ph = np.exp(1j * RNG.uniform(0, 2*np.pi, size=F.shape)); ph[0] = 1.0
    if n % 2 == 0: ph[-1] = 1.0
    return np.fft.irfft(np.abs(F)*ph, n=n) + x.mean()


def block_shuffle(x, block=24):
    x = np.asarray(x, float); n = len(x); nb = n // block
    idx = np.arange(nb); RNG.shuffle(idx)
    o = np.concatenate([x[i*block:(i+1)*block] for i in idx])
    if len(o) < n: o = np.concatenate([o, x[len(o):]])
    return o[:n]


def rate(a, thr=0.05):
    a = np.asarray(a); return 100*np.mean(a < thr) if len(a) else float("nan")


def main():
    panel = load_panel()
    logp = np.log(panel)
    tr = logp[logp.index < TRAIN_END]; te = logp[(logp.index >= TRAIN_END) & (logp.index < FULL_END)]
    syms = list(panel.columns); nt, ne = len(tr), len(te)
    all_pairs = list(combinations(syms, 2))
    out(f"panel {len(syms)} syms, nt={nt} ne={ne}, pairs={len(all_pairs)}")

    NP = 120
    psub = [all_pairs[i] for i in RNG.choice(len(all_pairs), size=NP, replace=False)]
    sig = float(np.std(np.diff(tr[syms[0]].to_numpy())))

    t0 = time.time()
    # (a) independent random walks
    rw = []
    for k in range(NP):
        ay = np.cumsum(RNG.normal(0, sig, nt+ne)); ax = np.cumsum(RNG.normal(0, sig, nt+ne))
        p = kp(ay[:nt], ax[:nt], ay[nt:], ax[nt:])
        if np.isfinite(p): rw.append(p)
        if (k+1) % 30 == 0: out(f"  rw {k+1}/{NP} ({time.time()-t0:.0f}s)")
    out(f"(a) independent random walks:      p<0.05 {rate(rw):.1f}%  p<0.001 {rate(rw,0.001):.1f}%  n={len(rw)}")

    # (b) coin vs phase-randomized coin
    pr = []
    for k, (a, b) in enumerate(psub):
        yf = logp[a].to_numpy(); xf = phase_randomize(logp[b].to_numpy())
        p = kp(yf[:nt], xf[:nt], yf[nt:nt+ne], xf[nt:nt+ne])
        if np.isfinite(p): pr.append(p)
        if (k+1) % 30 == 0: out(f"  phase {k+1}/{NP} ({time.time()-t0:.0f}s)")
    out(f"(b) coin vs phase-randomized coin: p<0.05 {rate(pr):.1f}%  p<0.001 {rate(pr,0.001):.1f}%  n={len(pr)}")

    # (c) real pair, x block-shuffled
    bs = []
    for k, (a, b) in enumerate(psub):
        yf = logp[a].to_numpy(); xf = block_shuffle(logp[b].to_numpy(), 24)
        p = kp(yf[:nt], xf[:nt], yf[nt:nt+ne], xf[nt:nt+ne])
        if np.isfinite(p): bs.append(p)
        if (k+1) % 30 == 0: out(f"  blockshuf {k+1}/{NP} ({time.time()-t0:.0f}s)")
    out(f"(c) real pair x block-shuffled:    p<0.05 {rate(bs):.1f}%  p<0.001 {rate(bs,0.001):.1f}%  n={len(bs)}")

    # CLEAN EG benchmark on ALL pairs (fast, no Kalman)
    eg = []; rows = []
    for a, b in all_pairs:
        yte, xte = te[a].to_numpy(), te[b].to_numpy()
        if len(yte) < 100: continue
        try:
            _, p, _ = coint(yte, xte, trend="c", autolag=None, maxlag=24)
        except Exception:
            continue
        eg.append(float(p)); rows.append({"sym_a": a, "sym_b": b, "eg_pvalue": float(p)})
    out(f"CLEAN Engle-Granger OOS (test logp): coint p<0.05 {rate(eg):.1f}%  p<0.01 {rate(eg,0.01):.1f}%  n={len(eg)}")
    egdf = pd.DataFrame(rows).sort_values("eg_pvalue")
    egdf.to_csv(ROOT / "scratch/clean_coint_pairs.csv", index=False)
    out(f"clean-coint pairs p<0.05: {int((egdf['eg_pvalue']<0.05).sum())}  saved scratch/clean_coint_pairs.csv")
    out("EXIT=0")


if __name__ == "__main__":
    main()
