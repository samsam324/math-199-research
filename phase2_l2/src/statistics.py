"""
Statistical utilities: HAC (Newey-West) long-run variance, deflated Sharpe
ratio (Bailey & Lopez de Prado 2014), and block bootstrap CIs.

Consolidated from phase 1's run_hac_sharpe.py, run_deflated_sharpe.py, and
plot_walk_forward_summary.py so the math has one home and one set of tests.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats


GAMMA_E = 0.5772156649015329


def hac_long_run_var(x: np.ndarray, lag: int) -> float:
    """Newey-West long-run variance with Bartlett kernel."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    x = x - x.mean()
    gamma_0 = float(x @ x) / n
    var = gamma_0
    L = min(lag, n - 1)
    for j in range(1, L + 1):
        gamma_j = float(x[:-j] @ x[j:]) / n
        weight = 1.0 - j / (L + 1)
        var += 2.0 * weight * gamma_j
    return max(var, 1e-20)


def hac_sharpe(pnl: np.ndarray, lag: int, bars_per_year: int) -> Tuple[float, float, float]:
    """Returns (iid_sharpe, hac_sharpe, inflation_factor), per-bar -> annualized."""
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) < 2:
        return 0.0, 0.0, float("nan")
    mu = float(pnl.mean())
    iid_std = float(pnl.std(ddof=1))
    hac_std = float(np.sqrt(hac_long_run_var(pnl, lag)))
    if iid_std <= 0 or hac_std <= 0:
        return 0.0, 0.0, float("nan")
    iid_sr = mu / iid_std * np.sqrt(bars_per_year)
    hac_sr = mu / hac_std * np.sqrt(bars_per_year)
    inflation = (mu / iid_std) / (mu / hac_std) if (mu / hac_std) != 0 else float("nan")
    return float(iid_sr), float(hac_sr), float(inflation)


def deflated_sharpe(
    sr_observed: float,
    sr_var_across_trials: float,
    n_trials: int,
    skew: float,
    kurt: float,
    T: int,
) -> Tuple[float, float]:
    """
    Returns (sr_0, DSR p-value). Inputs in per-bar units.

    sr_0 = expected max Sharpe under the null over n_trials.
    DSR  = Pr(true SR > 0 | observed) after non-normality + selection correction.
    """
    sqrt_v = float(np.sqrt(max(sr_var_across_trials, 1e-30)))
    z1 = stats.norm.ppf(1 - 1.0 / max(n_trials, 2))
    z2 = stats.norm.ppf(1 - 1.0 / (max(n_trials, 2) * np.e))
    sr_0 = sqrt_v * ((1 - GAMMA_E) * z1 + GAMMA_E * z2)

    denom = np.sqrt(max(1 - skew * sr_observed + (kurt - 1) / 4 * sr_observed * sr_observed, 1e-12))
    if T < 2:
        return float(sr_0), float("nan")
    test_stat = (sr_observed - sr_0) * np.sqrt(T - 1) / denom
    return float(sr_0), float(stats.norm.cdf(test_stat))


def block_bootstrap_ci(
    values: np.ndarray, n: int, lo: float, hi: float,
    block_size: int, rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """
    Circular block bootstrap (wrap-around). Reduces precision inflation on
    autocorrelated walk-forward metric streams.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    m = len(values)
    if m == 0:
        return float("nan"), float("nan"), float("nan")
    if m == 1:
        v = float(values[0])
        return v, v, v
    b = max(1, min(block_size, m))
    n_blocks = (m + b - 1) // b
    samples = np.empty(n, dtype=float)
    for i in range(n):
        starts = rng.integers(0, m, size=n_blocks)
        idx = (starts[:, None] + np.arange(b)[None, :]) % m
        flat = idx.reshape(-1)[:m]
        samples[i] = float(values[flat].mean())
    return float(values.mean()), float(np.quantile(samples, lo)), float(np.quantile(samples, hi))
