"""
Kalman filter for a time-varying hedge ratio on log-price pairs.

Ported as-is from phase 1 (hourly OHLCV). The state-space model is agnostic to
sampling cadence: it operates on a pair of log-price series indexed by time.
For phase 2 we feed it microprice log-series sampled at the chosen bar cadence
(default 1s).

State-space:
    state_t = [alpha_t, beta_t]
    state_t = state_{t-1} + w_t,        w_t ~ N(0, Q)
    y_t     = [1, x_t] @ state_t + v_t, v_t ~ N(0, R)

Q and R are MLE-fit on a training slice; the test slice is forward-rolled with
fixed parameters and the trained final state (honest OOS).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class KalmanConfig:
    q_alpha: float = 1e-7
    q_beta: float = 1e-5
    r_obs: Optional[float] = None
    p0: float = 1.0


def kalman_dynamic_hedge(
    y: np.ndarray,
    x: np.ndarray,
    cfg: KalmanConfig = KalmanConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    T = len(y)
    if T == 0 or len(x) != T:
        raise ValueError("y and x must be non-empty and the same length")

    X = add_constant(x)
    ols = OLS(y, X).fit()
    alpha0, beta0 = float(ols.params[0]), float(ols.params[1])
    r = float(cfg.r_obs) if cfg.r_obs is not None else float(np.var(ols.resid, ddof=1))
    if not np.isfinite(r) or r <= 0:
        r = 1e-4

    Q = np.diag([cfg.q_alpha, cfg.q_beta])
    state = np.array([alpha0, beta0], dtype=float)
    P = np.diag([cfg.p0, cfg.p0])

    alphas = np.empty(T, dtype=float)
    betas = np.empty(T, dtype=float)
    spreads = np.empty(T, dtype=float)
    state_var = np.empty((T, 2), dtype=float)

    for t in range(T):
        P_pred = P + Q
        H = np.array([1.0, x[t]], dtype=float)
        y_hat = float(H @ state)
        innov = y[t] - y_hat
        S = float(H @ P_pred @ H + r)
        if not np.isfinite(S) or S <= 0:
            alphas[t] = state[0]; betas[t] = state[1]; spreads[t] = innov
            state_var[t] = np.diag(P)
            continue
        K = (P_pred @ H) / S
        state = state + K * innov
        P = P_pred - np.outer(K, H) @ P_pred
        alphas[t] = state[0]; betas[t] = state[1]; spreads[t] = innov
        state_var[t] = np.diag(P)

    return alphas, betas, spreads, state_var


def static_spread(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    model = OLS(y, add_constant(x)).fit()
    alpha, beta = float(model.params[0]), float(model.params[1])
    resid = y - (alpha + beta * x)
    return resid, alpha, beta


def kalman_log_likelihood(
    y: np.ndarray, x: np.ndarray,
    log_q_alpha: float, log_q_beta: float, log_r: float,
    alpha0: float, beta0: float, p0: float = 1.0,
) -> float:
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    T = len(y)
    if T == 0 or len(x) != T:
        raise ValueError("y and x must be non-empty and the same length")

    q_alpha = float(np.exp(log_q_alpha))
    q_beta = float(np.exp(log_q_beta))
    r = float(np.exp(log_r))
    Q = np.diag([q_alpha, q_beta])
    state = np.array([alpha0, beta0], dtype=float)
    P = np.diag([p0, p0])

    nll = 0.0
    for t in range(T):
        P_pred = P + Q
        H = np.array([1.0, x[t]], dtype=float)
        y_hat = float(H @ state)
        innov = y[t] - y_hat
        S = float(H @ P_pred @ H + r)
        if not np.isfinite(S) or S <= 0:
            return 1e12
        nll += 0.5 * (np.log(2.0 * np.pi * S) + (innov * innov) / S)
        K = (P_pred @ H) / S
        state = state + K * innov
        P = P_pred - np.outer(K, H) @ P_pred
    return float(nll)


def fit_kalman_mle(
    y_train: np.ndarray, x_train: np.ndarray, *,
    p0: float = 1.0,
    bounds: tuple = ((-20.0, 0.0), (-20.0, 0.0), (-20.0, 5.0)),
) -> dict:
    """
    MLE-fit (Q_alpha, Q_beta, R) on a training slice. Returns dict with fitted
    params, final state, AND train_residuals computed with the fitted params
    (so a downstream train+test concatenation has no parameter discontinuity
    at the boundary — this was a real audit bug in phase 1).
    """
    from scipy.optimize import minimize

    y = np.asarray(y_train, dtype=float).ravel()
    x = np.asarray(x_train, dtype=float).ravel()
    if len(y) < 30 or len(x) != len(y):
        raise ValueError("Training slice too short or shape mismatch")

    ols = OLS(y, add_constant(x)).fit()
    alpha0 = float(ols.params[0])
    beta0 = float(ols.params[1])
    r0 = max(float(np.var(ols.resid, ddof=1)), 1e-8)
    init = np.array([np.log(1e-7), np.log(1e-5), np.log(r0)], dtype=float)

    def obj(theta):
        return kalman_log_likelihood(y, x, theta[0], theta[1], theta[2], alpha0, beta0, p0)

    res = minimize(obj, init, method="L-BFGS-B", bounds=list(bounds))
    log_q_alpha, log_q_beta, log_r = res.x.tolist()
    q_alpha = float(np.exp(log_q_alpha))
    q_beta = float(np.exp(log_q_beta))
    r = float(np.exp(log_r))

    Q = np.diag([q_alpha, q_beta])
    state = np.array([alpha0, beta0], dtype=float)
    P = np.diag([p0, p0])
    T = len(y)
    train_alphas = np.empty(T, dtype=float)
    train_betas = np.empty(T, dtype=float)
    train_residuals = np.empty(T, dtype=float)
    for t in range(T):
        P_pred = P + Q
        H = np.array([1.0, x[t]], dtype=float)
        S = float(H @ P_pred @ H + r)
        innov = y[t] - float(H @ state)
        K = (P_pred @ H) / S
        state = state + K * innov
        P = P_pred - np.outer(K, H) @ P_pred
        train_alphas[t] = state[0]
        train_betas[t] = state[1]
        train_residuals[t] = innov

    return {
        "q_alpha": q_alpha, "q_beta": q_beta, "r": r,
        "final_state": state.copy(), "final_cov": P.copy(),
        "alpha0": alpha0, "beta0": beta0,
        "nll": float(res.fun), "converged": bool(res.success),
        "alpha_train_final": float(state[0]),
        "beta_train_final": float(state[1]),
        "train_alphas": train_alphas,
        "train_betas": train_betas,
        "train_residuals": train_residuals,
    }


def kalman_forward_residuals(
    y_test: np.ndarray, x_test: np.ndarray, fitted: dict,
) -> tuple:
    """OOS forward filter using fitted params + trained final state."""
    y = np.asarray(y_test, dtype=float).ravel()
    x = np.asarray(x_test, dtype=float).ravel()
    T = len(y)
    if T == 0 or len(x) != T:
        return np.empty(0), np.empty(0), np.empty(0)

    state = np.asarray(fitted["final_state"], dtype=float).copy()
    P = np.asarray(fitted["final_cov"], dtype=float).copy()
    Q = np.diag([fitted["q_alpha"], fitted["q_beta"]])
    r = float(fitted["r"])

    alphas = np.empty(T, dtype=float)
    betas = np.empty(T, dtype=float)
    resids = np.empty(T, dtype=float)
    for t in range(T):
        P_pred = P + Q
        H = np.array([1.0, x[t]], dtype=float)
        y_hat = float(H @ state)
        innov = y[t] - y_hat
        S = float(H @ P_pred @ H + r)
        if not np.isfinite(S) or S <= 0:
            alphas[t] = state[0]; betas[t] = state[1]; resids[t] = innov
            continue
        K = (P_pred @ H) / S
        state = state + K * innov
        P = P_pred - np.outer(K, H) @ P_pred
        alphas[t] = state[0]; betas[t] = state[1]; resids[t] = innov
    return alphas, betas, resids


def adf_comparison(static_resid: np.ndarray, dyn_resid: np.ndarray, burn_in: int = 50) -> dict:
    s = np.asarray(static_resid, dtype=float)
    d = np.asarray(dyn_resid, dtype=float)[burn_in:]
    out: dict = {}
    for label, series in (("static", s), ("kalman", d)):
        try:
            stat, pvalue, *_ = adfuller(series, regression="c", autolag=None, maxlag=24)
            out[f"{label}_adf_stat"] = float(stat)
            out[f"{label}_adf_pvalue"] = float(pvalue)
        except Exception:
            out[f"{label}_adf_stat"] = float("nan")
            out[f"{label}_adf_pvalue"] = float("nan")
    out["static_std"] = float(np.std(s, ddof=1))
    out["kalman_std"] = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
    return out
