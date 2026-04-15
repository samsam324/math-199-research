# src/pair_selection.py
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class PairConfig:
    # Candidate screen
    min_corr: float = 0.5
    max_pairs_to_test: int = 6000

    # Stationarity filter
    max_adf_pvalue: float = 0.05

    # Mean reversion band preferences
    half_life_min_hours: float = 4.0
    half_life_max_hours: float = 240.0
    half_life_target_hours: float = 48.0

    # Beta stability
    beta_stability_segments: int = 4
    max_beta_instability: float = 0.35  # higher is worse, 0.35 is moderate

    # Spread amplitude filter
    min_spread_vol: float = 0.002  # residual std in log price units, tune per horizon

    # Minimum observations in the window
    min_obs: int = 1200

    # Score weights, applied after z scoring each metric across the candidate set
    w_adf: float = 0.40
    w_half_life: float = 0.25
    w_spread_vol: float = 0.20
    w_beta_stability: float = 0.10
    w_corr: float = 0.05


def _half_life_hours(resid: np.ndarray) -> float:
    """
    AR(1) style half life
      dx_t = a + b x_{t-1} + e_t
      half life = -ln(2) / b
    """
    x = resid.astype(float)
    if len(x) < 3:
        return float("inf")
    x_lag = x[:-1]
    dx = x[1:] - x[:-1]
    X = add_constant(x_lag)
    model = OLS(dx, X).fit()
    b = float(model.params[1])
    if not np.isfinite(b) or b >= 0:
        return float("inf")
    return float(-np.log(2) / b)


def _beta_segments(y: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    """
    Compute beta in k contiguous segments for stability checking
    y and x are log prices aligned, same length
    """
    n = len(y)
    if k <= 1 or n < 50:
        return np.array([np.nan], dtype=float)

    edges = np.linspace(0, n, k + 1).astype(int)
    betas = []
    for i in range(k):
        a, b = edges[i], edges[i + 1]
        if b - a < 30:
            continue
        yy = y[a:b]
        xx = x[a:b]
        X = add_constant(xx)
        m = OLS(yy, X).fit()
        betas.append(float(m.params[1]))
    if not betas:
        return np.array([np.nan], dtype=float)
    return np.asarray(betas, dtype=float)


def _beta_instability(betas: np.ndarray) -> float:
    """
    Dimensionless instability score
      std(beta_segments) / (abs(mean_beta) + eps)
    Lower is better
    """
    betas = betas[np.isfinite(betas)]
    if len(betas) < 2:
        return float("inf")
    mu = float(np.mean(betas))
    sd = float(np.std(betas, ddof=1))
    eps = 1e-9
    return sd / (abs(mu) + eps)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def score_pairs(close_panel: pd.DataFrame, pcfg: PairConfig) -> pd.DataFrame:
    """
    Input
      close_panel index timestamps, columns symbols, values close
    Output
      ranked DataFrame with diagnostics and final score
    """
    if close_panel.empty or close_panel.shape[1] < 2:
        return pd.DataFrame()

    if len(close_panel) < pcfg.min_obs:
        print(f"[WARN] only {len(close_panel)} rows, consider expanding training window")

    logp = np.log(close_panel)
    rets = logp.diff().dropna()
    corr = rets.corr()

    symbols = list(close_panel.columns)
    candidates: List[Tuple[str, str, float]] = []
    for a, b in combinations(symbols, 2):
        c = float(corr.loc[a, b])
        if np.isfinite(c) and c >= pcfg.min_corr:
            candidates.append((a, b, c))

    candidates.sort(key=lambda t: t[2], reverse=True)
    candidates = candidates[: pcfg.max_pairs_to_test]

    rows = []
    for a, b, c in candidates:
        y = logp[a].values
        x = logp[b].values

        # OLS hedge ratio
        X = add_constant(x)
        model = OLS(y, X).fit()
        alpha = float(model.params[0])
        beta = float(model.params[1])
        resid = y - (alpha + beta * x)

        # Spread volatility filter
        spread_vol = float(np.std(resid, ddof=1))
        if not np.isfinite(spread_vol) or spread_vol < pcfg.min_spread_vol:
            continue

        # ADF on residual, use autolag for robustness
        try:
            adf_stat, adf_pvalue, _, _, _, _ = adfuller(resid, regression="c", autolag="AIC")
            adf_stat = float(adf_stat)
            adf_pvalue = float(adf_pvalue)
        except Exception:
            continue

        if not np.isfinite(adf_pvalue) or adf_pvalue > pcfg.max_adf_pvalue:
            continue

        # Half life
        hl = _half_life_hours(resid)
        if not np.isfinite(hl) or hl <= 0:
            continue
        if hl < pcfg.half_life_min_hours or hl > pcfg.half_life_max_hours:
            continue

        # Beta stability across segments
        betas = _beta_segments(y, x, pcfg.beta_stability_segments)
        instab = _beta_instability(betas)
        if not np.isfinite(instab) or instab > pcfg.max_beta_instability:
            continue

        rows.append(
            {
                "sym_a": a,
                "sym_b": b,
                "corr": c,
                "alpha": alpha,
                "beta_a_on_b": beta,
                "adf_stat": adf_stat,
                "adf_pvalue": adf_pvalue,
                "half_life_hours": hl,
                "spread_vol": spread_vol,
                "beta_instability": instab,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Ranking features
    # Stronger stationarity means more negative ADF stat, so use strength = -adf_stat
    adf_strength = (-out["adf_stat"].values).astype(float)

    # Half life preference peaks at target using log distance penalty
    hl = out["half_life_hours"].values.astype(float)
    hl_target = float(pcfg.half_life_target_hours)
    hl_pref = (-np.abs(np.log(hl / hl_target))).astype(float)

    # Higher spread_vol better
    spread_vol = out["spread_vol"].values.astype(float)

    # Lower instability better, so stability = -instability
    beta_stab = (-out["beta_instability"].values).astype(float)

    # Correlation as weak preference
    corr_v = out["corr"].values.astype(float)

    # Z score each component across remaining candidates
    z_adf = _zscore(adf_strength)
    z_hl = _zscore(hl_pref)
    z_vol = _zscore(spread_vol)
    z_beta = _zscore(beta_stab)
    z_corr = _zscore(corr_v)

    score = (
        pcfg.w_adf * z_adf
        + pcfg.w_half_life * z_hl
        + pcfg.w_spread_vol * z_vol
        + pcfg.w_beta_stability * z_beta
        + pcfg.w_corr * z_corr
    )

    out["score"] = score
    out["pair"] = out["sym_a"] + "_" + out["sym_b"]

    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out[
        [
            "pair",
            "sym_a",
            "sym_b",
            "corr",
            "beta_a_on_b",
            "beta_instability",
            "spread_vol",
            "adf_stat",
            "adf_pvalue",
            "half_life_hours",
            "score",
        ]
    ]
