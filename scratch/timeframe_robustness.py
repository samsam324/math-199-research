"""
Timeframe robustness for the two screening artifacts.

Open question from the build plan: are the Kalman-innovation cointegration artifact
and the rolling-z mechanical-reversion artifact specific to the hourly bar, or do they
hold at coarser bars too? If they are frequency-invariant, the "single interval (hourly)"
limitation is closed for the artifacts and Section 3's claim ("these screens are
mechanically fooled regardless of cadence") gets stronger.

Method:
  Kalman artifact -- resample the top-50 hourly close panel to 4h and 1D. Run the SAME
    out-of-sample Kalman ADF screen on real pairs and on independent random-walk placebos.
    Artifact confirmed if placebos pass at about the same (~100%) rate as real pairs, while
    a clean Engle-Granger benchmark stays near the 5% null floor.
  Rolling-z artifact -- simulate random walks at each frequency's test-window length, apply
    the rolling z-score event-reversion machinery, and report the mean reversion "floor".
    Artifact confirmed if the floor is positive at every frequency.

Reuses the validated src.kalman_hedge. Hourly reproduces audit_part1.py / persistence_test.py
as the within-frequency anchor.
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
from statsmodels.tsa.stattools import adfuller, coint
from src.kalman_hedge import fit_kalman_mle, kalman_forward_residuals

RNG = np.random.default_rng(7)
UNI = [l.strip() for l in open(ROOT / "data/l2_universe_top50.txt") if l.strip()]

# per-frequency config: resample rule, train bars, test bars, adf maxlag, rolling-z lookback, horizon
FREQS = {
    "1H": dict(rule="1H", train=24 * 180, test=24 * 90, maxlag=24, lb=240, hor=48),
    "4H": dict(rule="4h", train=6 * 180,  test=6 * 90,  maxlag=12, lb=60,  hor=24),
    "1D": dict(rule="1D", train=400,      test=220,     maxlag=10, lb=30,  hor=10),
}
N_REAL = 60
N_PLAC = 60


def adf_p(series, maxlag):
    s = np.asarray(series, float)
    if len(s) < 20 or not np.isfinite(s).all():
        return float("nan")
    try:
        stat, p, *_ = adfuller(s, regression="c", autolag=None, maxlag=maxlag)
        return float(p)
    except Exception:
        return float("nan")


def load_hourly_panel():
    cols = {}
    for s in UNI:
        try:
            df = pd.read_parquet(ROOT / f"data/spot_1h/{s}.parquet")
            cols[s] = df["close"]
        except Exception:
            pass
    return pd.DataFrame(cols).sort_index()


def resample_panel(panel, rule):
    p = panel.copy() if rule == "1H" else panel.resample(rule).last()
    return p.dropna(how="any")


def kalman_oos_p(y_tr, x_tr, y_te, x_te, maxlag):
    try:
        fitted = fit_kalman_mle(y_tr, x_tr)
    except Exception:
        return float("nan")
    if not fitted.get("converged", False):
        return float("nan")
    _, _, resid = kalman_forward_residuals(y_te, x_te, fitted)
    return adf_p(resid, maxlag)


def static_oos_p(y_tr, x_tr, y_te, x_te, maxlag):
    try:
        ols = OLS(y_tr, add_constant(x_tr)).fit()
        a, b = float(ols.params[0]), float(ols.params[1])
    except Exception:
        return float("nan")
    return adf_p(y_te - (a + b * x_te), maxlag)


def sim_rw(n, sigma):
    return np.cumsum(RNG.normal(0, sigma, size=n))


def rate(arr, thr=0.05):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    return 100 * np.mean(arr < thr) if len(arr) else float("nan")


# ---- rolling-z machinery (mirrors persistence_test.py) ----
def rolling_z(spread, lb):
    x = np.asarray(spread, float); n = len(x)
    z = np.full(n, np.nan)
    if n < lb:
        return z
    c = np.cumsum(np.insert(x, 0, 0.0)); c2 = np.cumsum(np.insert(x * x, 0, 0.0))
    idx = np.arange(lb - 1, n)
    s1 = c[idx + 1] - c[idx + 1 - lb]; s2 = c2[idx + 1] - c2[idx + 1 - lb]
    mean = s1 / lb; var = s2 / lb - mean * mean
    var = np.where(var < 0, 0.0, var) * (lb / (lb - 1)); sd = np.sqrt(var)
    with np.errstate(invalid="ignore", divide="ignore"):
        z[idx] = (x[idx] - mean) / sd
    return z


def event_reversion(zv, lb, horizon, zthr=2.0, cooldown=None):
    if cooldown is None:
        cooldown = horizon
    n = len(zv); revs = []; last = -10 ** 9
    for t in range(lb, n - horizon):
        zt = zv[t]
        if not np.isfinite(zt) or abs(zt) <= zthr:
            continue
        if t - last < cooldown:
            continue
        last = t; zf = zv[t + horizon]
        if np.isfinite(zf):
            revs.append(-np.sign(zt) * (zf - zt))
    return np.array(revs)


def rollingz_floor(n, lb, horizon, n_surr=200):
    means = []
    for _ in range(n_surr):
        rw = np.cumsum(RNG.normal(0, 1.0, size=n))
        r = event_reversion(rolling_z(rw, lb), lb, horizon)
        if len(r):
            means.append(np.mean(r))
    if not means:
        return float("nan"), float("nan"), 0
    return float(np.mean(means)), float(np.std(means)), len(means)


def run_kalman_artifact(panel):
    print("\n#### KALMAN COINTEGRATION ARTIFACT across timeframes ####", flush=True)
    hdr = f"{'freq':>5} | {'real<.05':>9} {'real<.001':>9} | {'RWplac<.05':>11} {'RWplac<.001':>11} | {'EG<.05':>7} {'staticOLS<.05':>14} | n"
    print(hdr, flush=True)
    out = []
    for fk, cfg in FREQS.items():
        p = resample_panel(panel, cfg["rule"])
        logp = np.log(p)
        ntr, nte = cfg["train"], cfg["test"]
        if len(logp) < ntr + nte:
            if len(logp) < 200:
                print(f"{fk:>5} | insufficient bars ({len(logp)})", flush=True)
                continue
            ntr = int(len(logp) * 0.66); nte = len(logp) - ntr
        seg = logp.iloc[-(ntr + nte):]
        tr = seg.iloc[:ntr]; te = seg.iloc[ntr:]
        syms = list(seg.columns)
        allpairs = list(combinations(syms, 2))
        idx = RNG.choice(len(allpairs), size=min(N_REAL, len(allpairs)), replace=False)
        pairs = [allpairs[i] for i in idx]
        rk, rs, eg = [], [], []
        for a, b in pairs:
            ytr, xtr = tr[a].to_numpy(), tr[b].to_numpy()
            yte, xte = te[a].to_numpy(), te[b].to_numpy()
            pk = kalman_oos_p(ytr, xtr, yte, xte, cfg["maxlag"])
            ps = static_oos_p(ytr, xtr, yte, xte, cfg["maxlag"])
            if np.isfinite(pk):
                rk.append(pk)
            if np.isfinite(ps):
                rs.append(ps)
            try:
                _, pe, _ = coint(yte, xte, trend="c", autolag=None, maxlag=cfg["maxlag"])
                eg.append(float(pe))
            except Exception:
                pass
        sig = float(np.std(np.diff(tr[syms[0]].to_numpy())))
        pk_pl = []
        for _ in range(N_PLAC):
            ay = sim_rw(ntr + nte, sig); ax = sim_rw(ntr + nte, sig)
            pk = kalman_oos_p(ay[:ntr], ax[:ntr], ay[ntr:], ax[ntr:], cfg["maxlag"])
            if np.isfinite(pk):
                pk_pl.append(pk)
        row = dict(freq=fk, train_bars=ntr, test_bars=nte,
                   real_k05=rate(rk), real_k001=rate(rk, 0.001),
                   plac_k05=rate(pk_pl), plac_k001=rate(pk_pl, 0.001),
                   eg05=rate(eg), static05=rate(rs), n=len(rk))
        out.append(row)
        print(f"{fk:>5} | {row['real_k05']:>9.1f} {row['real_k001']:>9.1f} | "
              f"{row['plac_k05']:>11.1f} {row['plac_k001']:>11.1f} | "
              f"{row['eg05']:>7.1f} {row['static05']:>14.1f} | {row['n']}", flush=True)
    pd.DataFrame(out).to_csv(ROOT / "scratch/timeframe_kalman_artifact.csv", index=False)
    print("saved -> scratch/timeframe_kalman_artifact.csv", flush=True)


def run_rollingz_artifact():
    print("\n#### ROLLING-Z MECHANICAL REVERSION FLOOR across timeframes ####", flush=True)
    print(f"{'freq':>5} | {'testbars':>8} {'lookback':>8} {'horizon':>7} | {'mean floor(z)':>13} {'sd':>7} {'nsurr':>6}", flush=True)
    out = []
    for fk, cfg in FREQS.items():
        m, sd, k = rollingz_floor(cfg["test"], cfg["lb"], cfg["hor"])
        out.append(dict(freq=fk, test_bars=cfg["test"], lookback=cfg["lb"], horizon=cfg["hor"],
                        mean_floor=m, sd=sd, nsurr=k))
        print(f"{fk:>5} | {cfg['test']:>8} {cfg['lb']:>8} {cfg['hor']:>7} | {m:>13.4f} {sd:>7.4f} {k:>6}", flush=True)
    pd.DataFrame(out).to_csv(ROOT / "scratch/timeframe_rollingz_floor.csv", index=False)
    print("saved -> scratch/timeframe_rollingz_floor.csv", flush=True)


def main():
    t0 = time.time()
    panel = load_hourly_panel()
    print(f"hourly panel: {panel.shape[0]} bars x {panel.shape[1]} syms, "
          f"{panel.index.min()} .. {panel.index.max()}", flush=True)
    run_kalman_artifact(panel)
    run_rollingz_artifact()
    print(f"\nDONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
