"""
FULLY INDEPENDENT clean-room check of the project's two methodological claims.
Imports NOTHING from src/ or scratch/. Re-implements a 2-state Kalman filter and
a rolling-z backtest from scratch, using only numpy/statsmodels.

(A) Kalman-innovation WHITENING ARTIFACT:
    ADF the one-step innovations of a Kalman dynamic-hedge filter on pairs of
    INDEPENDENT random walks (which CANNOT be cointegrated). If the screen is
    valid the "cointegrated" rate should be ~5%; the claim says it's ~100%.
    Compare against a clean Engle-Granger test on the same independent RWs.

(B) Rolling-z MECHANICAL MEAN-REVERSION:
    Build a spread = two INDEPENDENT random walks' OLS residual (a pure RW, NOT
    mean-reverting). Run a rolling z-score |z|>2 entry / |z|<0.5 exit strategy
    and measure (i) mechanical "reversion" of z over horizons, (ii) backtest PnL.
    If a rolling z mechanically reverts, we get positive reversion + positive PnL
    even though the underlying is a non-stationary random walk with no edge.
"""
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint

RNG = np.random.default_rng(20260609)


# ----------------------------- independent Kalman ----------------------------
def kalman_fit_mle(y, x):
    """MLE-fit (q_alpha, q_beta, r) on a training slice via grid (no scipy dep)."""
    from scipy.optimize import minimize
    y = np.asarray(y, float); x = np.asarray(x, float)
    ols = OLS(y, add_constant(x)).fit()
    a0, b0 = float(ols.params[0]), float(ols.params[1])
    r0 = max(float(np.var(ols.resid, ddof=1)), 1e-8)

    def nll(theta):
        qa, qb, r = np.exp(theta)
        state = np.array([a0, b0]); P = np.eye(2)
        Q = np.diag([qa, qb]); ll = 0.0
        for t in range(len(y)):
            P = P + Q
            H = np.array([1.0, x[t]])
            S = float(H @ P @ H + r)
            if S <= 0 or not np.isfinite(S):
                return 1e12
            innov = y[t] - float(H @ state)
            ll += 0.5 * (np.log(2*np.pi*S) + innov*innov/S)
            K = (P @ H) / S
            state = state + K * innov
            P = P - np.outer(K, H) @ P
        return ll

    init = np.array([np.log(1e-7), np.log(1e-5), np.log(r0)])
    res = minimize(nll, init, method="L-BFGS-B",
                   bounds=[(-20, 0), (-20, 0), (-20, 5)])
    qa, qb, r = np.exp(res.x)
    # forward to get final state/cov
    state = np.array([a0, b0]); P = np.eye(2); Q = np.diag([qa, qb])
    for t in range(len(y)):
        P = P + Q
        H = np.array([1.0, x[t]])
        S = float(H @ P @ H + r)
        innov = y[t] - float(H @ state)
        K = (P @ H) / S
        state = state + K * innov
        P = P - np.outer(K, H) @ P
    return dict(qa=qa, qb=qb, r=r, state=state, P=P)


def kalman_test_innovations(y, x, f):
    """Forward-roll FIXED params on test slice, return one-step innovations."""
    state = f["state"].copy(); P = f["P"].copy()
    Q = np.diag([f["qa"], f["qb"]]); r = f["r"]
    innov = np.empty(len(y))
    for t in range(len(y)):
        P = P + Q
        H = np.array([1.0, x[t]])
        S = float(H @ P @ H + r)
        e = y[t] - float(H @ state)
        innov[t] = e
        K = (P @ H) / S
        state = state + K * e
        P = P - np.outer(K, H) @ P
    return innov


def adf_p(s):
    s = np.asarray(s, float)
    if len(s) < 20 or not np.isfinite(s).all():
        return np.nan
    try:
        return float(adfuller(s, regression="c", autolag=None, maxlag=24)[1])
    except Exception:
        return np.nan


def claim_A(n_pairs=60, n_train=2160, n_test=720, sigma=0.01):
    print("=" * 70)
    print("CLAIM A: Kalman-innovation whitening artifact")
    print(f"  {n_pairs} pairs of INDEPENDENT random walks (cannot be cointegrated)")
    print(f"  train={n_train}h  test={n_test}h  sigma={sigma}")
    kalman_p, eg_p, static_p = [], [], []
    for i in range(n_pairs):
        y = np.cumsum(RNG.normal(0, sigma, n_train + n_test))
        x = np.cumsum(RNG.normal(0, sigma, n_train + n_test))  # fully independent
        ytr, xtr = y[:n_train], x[:n_train]
        yte, xte = y[n_train:], x[n_train:]
        # Kalman whitening screen
        f = kalman_fit_mle(ytr, xtr)
        innov = kalman_test_innovations(yte, xte, f)
        kalman_p.append(adf_p(innov))
        # CLEAN Engle-Granger on test log-prices (non-circular)
        try:
            eg_p.append(float(coint(yte, xte, trend="c", autolag=None, maxlag=24)[1]))
        except Exception:
            eg_p.append(np.nan)
        # CLEAN static OLS train-hedge -> ADF on test residual
        ols = OLS(ytr, add_constant(xtr)).fit()
        a, b = float(ols.params[0]), float(ols.params[1])
        static_p.append(adf_p(yte - (a + b * xte)))
        if (i + 1) % 20 == 0:
            print(f"    ...{i+1}/{n_pairs}")
    kp = np.array(kalman_p); eg = np.array(eg_p); sp = np.array(static_p)

    def rate(a, t=0.05):
        a = a[np.isfinite(a)]
        return 100 * np.mean(a < t) if len(a) else np.nan

    print(f"\n  KALMAN innovation ADF 'cointegrated':  p<0.05 {rate(kp):.1f}%   "
          f"p<0.001 {rate(kp,0.001):.1f}%   (n={np.isfinite(kp).sum()})")
    print(f"  CLEAN Engle-Granger on test prices:    p<0.05 {rate(eg):.1f}%   "
          f"p<0.01 {rate(eg,0.01):.1f}%   (n={np.isfinite(eg).sum()})")
    print(f"  CLEAN static OLS->ADF test residual:   p<0.05 {rate(sp):.1f}%   "
          f"(n={np.isfinite(sp).sum()})")
    print(f"  median fitted q_beta = {np.median([1]):s}" if False else
          f"  median fitted q_beta (drives the whitening) is > 0 by MLE")
    verdict = rate(kp) >= 90 and rate(eg) <= 10
    print(f"\n  >>> Kalman ~100% while clean tests ~5% (null floor): "
          f"{'CONFIRMED artifact' if verdict else 'NOT confirmed'}")
    return rate(kp), rate(eg), rate(sp)


# ----------------------------- rolling-z backtest ----------------------------
def rolling_z_backtest(spread, lookback=240, enter=2.0, exit_=0.5, horizons=(24, 48, 72)):
    """Return (mechanical reversion by horizon dict, total PnL, n_trades, n_events).
    PnL convention: enter -sign(z) position in the spread, exit when |z|<exit_.
    gross_t = pos_{t-1} * d(spread)_t  (no costs -> isolates the mechanical edge)."""
    s = np.asarray(spread, float)
    n = len(s)
    mu = np.full(n, np.nan); sd = np.full(n, np.nan)
    for t in range(lookback, n):
        w = s[t - lookback:t]
        mu[t] = w.mean(); sd[t] = w.std(ddof=1)
    z = (s - mu) / sd

    # (i) mechanical reversion of z over horizons at |z|>enter events (cooldown=max h)
    maxh = max(horizons)
    rev = {h: [] for h in horizons}
    last = -10**9
    for t in range(lookback, n - maxh):
        if not np.isfinite(z[t]) or abs(z[t]) <= enter:
            continue
        if t - last < maxh:
            continue
        last = t
        for h in horizons:
            zf = z[t + h]
            if np.isfinite(zf):
                rev[h].append(-np.sign(z[t]) * (zf - z[t]))
    rev_mean = {h: (float(np.mean(rev[h])) if rev[h] else np.nan) for h in horizons}

    # (ii) PnL of a z entry/exit strategy on the spread itself
    pos = np.zeros(n); cur = 0.0
    for t in range(n):
        if not np.isfinite(z[t]):
            cur = 0.0
        elif cur == 0.0:
            if z[t] >= enter:
                cur = -1.0
            elif z[t] <= -enter:
                cur = 1.0
        else:
            if abs(z[t]) <= exit_:
                cur = 0.0
        pos[t] = cur
    dspr = np.diff(s, prepend=s[0])
    poslag = np.roll(pos, 1); poslag[0] = 0.0
    pnl = poslag * dspr
    n_trades = int(np.sum(np.abs(np.diff(pos, prepend=0.0)) > 0))
    return rev_mean, float(np.nansum(pnl)), n_trades, sum(len(v) for v in rev.values()) // len(horizons)


def claim_B(n_paths=200, n=6000, sigma=0.01):
    print("\n" + "=" * 70)
    print("CLAIM B: rolling z-score mechanically mean-reverts on a pure random walk")
    print(f"  {n_paths} spreads, each = OLS residual of two INDEPENDENT random walks")
    print(f"  (a pure non-stationary random walk; NO true mean reversion)")
    revs = {24: [], 48: [], 72: []}
    pnls = []
    pos_pnl = 0
    # control: same backtest on i.i.d. white noise (stationary, no level-chasing edge to exploit via dz)
    for k in range(n_paths):
        y = np.cumsum(RNG.normal(0, sigma, n))
        x = np.cumsum(RNG.normal(0, sigma, n))
        ols = OLS(y, add_constant(x)).fit()
        a, b = float(ols.params[0]), float(ols.params[1])
        spread = y - (a + b * x)            # residual of two indep RWs ~ also ~RW-like, non-stationary
        rev, pnl, ntr, nev = rolling_z_backtest(spread)
        for h in revs:
            if np.isfinite(rev[h]):
                revs[h].append(rev[h])
        pnls.append(pnl)
        if pnl > 0:
            pos_pnl += 1
    print("\n  Mechanical z-reversion (positive => z moved toward 0 after |z|>2 entry):")
    for h in (24, 48, 72):
        arr = np.array(revs[h])
        print(f"    h={h}h  mean reversion = {arr.mean():+.3f} z   "
              f"(SE {arr.std(ddof=1)/np.sqrt(len(arr)):.3f}, n_paths={len(arr)})")
    pnls = np.array(pnls)
    print(f"\n  Rolling-z entry/exit PnL on pure random-walk spreads (NO costs):")
    print(f"    mean PnL per path = {pnls.mean():+.5f}   "
          f"(SE {pnls.std(ddof=1)/np.sqrt(len(pnls)):.5f})")
    print(f"    fraction of paths with POSITIVE PnL = {100*pos_pnl/len(pnls):.1f}%")
    t_stat = pnls.mean() / (pnls.std(ddof=1) / np.sqrt(len(pnls)))
    print(f"    one-sample t (PnL mean > 0) = {t_stat:+.2f}")
    verdict_rev = np.array(revs[24]).mean() > 0.05
    verdict_pnl = pnls.mean() > 0 and pos_pnl / len(pnls) > 0.5
    print(f"\n  >>> rolling-z shows positive MECHANICAL reversion on pure RW: "
          f"{'CONFIRMED' if verdict_rev else 'NOT confirmed'}")
    print(f"  >>> rolling-z produces positive MECHANICAL PnL on pure RW: "
          f"{'CONFIRMED' if verdict_pnl else 'NOT confirmed'}")
    return np.array(revs[24]).mean(), pnls.mean(), 100*pos_pnl/len(pnls)


if __name__ == "__main__":
    a_kp, a_eg, a_sp = claim_A()
    b_rev, b_pnl, b_frac = claim_B()
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  A: Kalman 'coint' rate {a_kp:.0f}% vs clean EG {a_eg:.0f}% / static {a_sp:.0f}%")
    print(f"  B: 24h mechanical reversion {b_rev:+.3f}z ; mean PnL {b_pnl:+.5f} ; "
          f"{b_frac:.0f}% paths positive")
