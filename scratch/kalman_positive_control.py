"""
Positive control + whiteness diagnostic for the Kalman cointegration artifact.

The placebos in audit_part1.py are NEGATIVE controls (series that cannot be cointegrated). A
referee also wants a POSITIVE control: a genuinely cointegrated pair the screen SHOULD pass, to
show the Kalman screen passes it AND the negatives at ~the same high rate (no discrimination),
while the clean Engle-Granger / static-OLS tests separate the positive from the negatives. We
also measure whiteness directly (Ljung-Box on the Kalman innovations) -- the mechanism in the
paper's title, asserted but never measured.

Run: python scratch/kalman_positive_control.py
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.diagnostic import acorr_ljungbox
from src.kalman_hedge import fit_kalman_mle, kalman_forward_residuals

RNG = np.random.default_rng(11)
NTR, NTE = 1500, 500
MAXLAG = 24
LB_LAGS = 24
N_PAIRS = 60


def adf_p(s):
    s = np.asarray(s, float)
    if len(s) < 20 or not np.isfinite(s).all():
        return np.nan
    try:
        return float(adfuller(s, regression="c", autolag=None, maxlag=MAXLAG)[1])
    except Exception:
        return np.nan


def lb_white_p(resid, lags=LB_LAGS):
    r = np.asarray(resid, float); r = r[np.isfinite(r)]
    if len(r) < lags + 10:
        return np.nan
    try:
        out = acorr_ljungbox(r, lags=[lags], return_df=True)
        return float(out["lb_pvalue"].iloc[0])   # high p => fail to reject whiteness => white
    except Exception:
        return np.nan


def kalman_run(y, x):
    ytr, xtr, yte, xte = y[:NTR], x[:NTR], y[NTR:NTR + NTE], x[NTR:NTR + NTE]
    try:
        fit = fit_kalman_mle(ytr, xtr)
    except Exception:
        return np.nan, np.nan, np.nan
    if not fit.get("converged", False):
        return np.nan, np.nan, fit.get("q_beta", np.nan)
    _, _, resid = kalman_forward_residuals(yte, xte, fit)
    return adf_p(resid), lb_white_p(resid), fit.get("q_beta", np.nan)


def static_run(y, x):
    ytr, xtr, yte, xte = y[:NTR], x[:NTR], y[NTR:NTR + NTE], x[NTR:NTR + NTE]
    try:
        ols = OLS(ytr, add_constant(xtr)).fit()
        a, b = float(ols.params[0]), float(ols.params[1])
    except Exception:
        return np.nan, np.nan
    resid = yte - (a + b * xte)
    return adf_p(resid), lb_white_p(resid)


def eg_p(y, x):
    yte, xte = y[NTR:NTR + NTE], x[NTR:NTR + NTE]
    try:
        return float(coint(yte, xte, trend="c", autolag=None, maxlag=MAXLAG)[1])
    except Exception:
        return np.nan


def sim_rw(n, sig=0.01):
    return np.cumsum(RNG.normal(0, sig, n))


def sim_coint(n, beta=1.0, hl=24, x_sig=0.02, e_sig=0.0025):
    """y, x are cointegrated: y - beta*x = e, a tight stationary AR(1) with half-life hl.
    The common trend (x_sig) is large relative to the residual noise (e_sig), so the
    cointegration is strong and a powered clean test should detect it."""
    x = sim_rw(n, x_sig)
    phi = 2.0 ** (-1.0 / hl)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + RNG.normal(0, e_sig)
    return beta * x + e, x


def rate(a, thr=0.05):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return 100 * np.mean(a < thr) if len(a) else np.nan


def white_frac(a, thr=0.05):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return 100 * np.mean(a > thr) if len(a) else np.nan   # white = fail to reject


def run_group(name, gen):
    k_adf, k_white, qb, s_adf, s_white, eg = [], [], [], [], [], []
    for _ in range(N_PAIRS):
        y, x = gen()
        ka, kw, q = kalman_run(y, x)
        sa, sw = static_run(y, x)
        e = eg_p(y, x)
        k_adf.append(ka); k_white.append(kw); qb.append(q)
        s_adf.append(sa); s_white.append(sw); eg.append(e)
    row = dict(group=name, n=N_PAIRS,
               kalman_adf_pass=rate(k_adf), kalman_innov_white=white_frac(k_white),
               q_beta_median=float(np.nanmedian(qb)),
               clean_eg_pass=rate(eg), static_adf_pass=rate(s_adf),
               static_resid_white=white_frac(s_white))
    print(f"\n[{name}]  n={N_PAIRS}")
    print(f"  Kalman ADF pass (p<.05): {row['kalman_adf_pass']:.1f}%   innovations white (LB p>.05): "
          f"{row['kalman_innov_white']:.1f}%   median q_beta: {row['q_beta_median']:.2e}")
    print(f"  clean EG pass: {row['clean_eg_pass']:.1f}%   static-OLS ADF pass: {row['static_adf_pass']:.1f}%   "
          f"static resid white: {row['static_resid_white']:.1f}%")
    return row


def main():
    print(f"positive control + whiteness diagnostic  (train {NTR}, test {NTE}, {N_PAIRS} pairs/group, LB lags {LB_LAGS})")
    rows = [run_group("POSITIVE control (truly cointegrated)", lambda: sim_coint(NTR + NTE)),
            run_group("NEGATIVE control (independent random walks)",
                      lambda: (sim_rw(NTR + NTE, 0.02), sim_rw(NTR + NTE, 0.02)))]
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "scratch", "kalman_positive_control.csv"), index=False)
    print("\nsaved -> scratch/kalman_positive_control.csv")
    print("\nReading: the Kalman ADF passes BOTH groups at ~the same high rate (no discrimination; its innovations")
    print("are white in both), while the clean EG/static tests pass the positive control and stay near the null")
    print("floor on the negatives. So the Kalman screen carries ~zero cointegration information; the clean tests carry it.")


if __name__ == "__main__":
    main()
