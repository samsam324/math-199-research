"""
Cointegration screen and correlation fallback.

Ported from phase 1. Inputs change from hourly close panels to microprice
panels at the chosen bar cadence (default 1s). The scoring logic is identical.
"""
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
    min_corr: float = 0.5
    max_pairs_to_test: int = 6000

    max_adf_pvalue: float = 0.05
    adf_maxlag: int = 24
    adf_autolag: str | None = None

    # Mean-reversion half-life range, expressed in bars of the input cadence.
    # At 1s cadence these are SECONDS (4 bars = 4s ... 240 bars = 4 min).
    # At 1m cadence they would be minutes. Callers should set explicitly per
    # cadence rather than relying on defaults inherited from hourly phase 1.
    half_life_min_bars: float = 4.0
    half_life_max_bars: float = 240.0
    half_life_target_bars: float = 48.0

    beta_stability_segments: int = 4
    max_beta_instability: float = 0.35

    min_spread_vol: float = 1e-5  # log-microprice units; smaller than phase 1 because microprice is smoother

    min_obs: int = 1200

    w_adf: float = 0.40
    w_half_life: float = 0.25
    w_spread_vol: float = 0.20
    w_beta_stability: float = 0.10
    w_corr: float = 0.05


def _half_life_bars(resid: np.ndarray) -> float:
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
    n = len(y)
    if k <= 1 or n < 50:
        return np.array([np.nan], dtype=float)
    edges = np.linspace(0, n, k + 1).astype(int)
    betas = []
    for i in range(k):
        a, b = edges[i], edges[i + 1]
        if b - a < 30:
            continue
        X = add_constant(x[a:b])
        m = OLS(y[a:b], X).fit()
        betas.append(float(m.params[1]))
    return np.asarray(betas, dtype=float) if betas else np.array([np.nan])


def _beta_instability(betas: np.ndarray) -> float:
    betas = betas[np.isfinite(betas)]
    if len(betas) < 2:
        return float("inf")
    mu = float(np.mean(betas))
    sd = float(np.std(betas, ddof=1))
    return sd / (abs(mu) + 1e-9)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def score_pairs(close_panel: pd.DataFrame, pcfg: PairConfig) -> pd.DataFrame:
    """
    Input: panel indexed by timestamp, columns symbols, values (micro)price.
    Output: ranked DataFrame with diagnostics and composite score.
    """
    if close_panel.empty or close_panel.shape[1] < 2:
        return pd.DataFrame()

    if len(close_panel) < pcfg.min_obs:
        print(f"[WARN] only {len(close_panel)} rows, consider widening training window")

    logp = np.log(close_panel)
    rets = logp.diff().dropna()
    corr = rets.corr()

    candidates: List[Tuple[str, str, float]] = []
    for a, b in combinations(list(close_panel.columns), 2):
        c = float(corr.loc[a, b])
        if np.isfinite(c) and c >= pcfg.min_corr:
            candidates.append((a, b, c))
    candidates.sort(key=lambda t: t[2], reverse=True)
    candidates = candidates[: pcfg.max_pairs_to_test]

    rows = []
    for a, b, c in candidates:
        y = logp[a].values
        x = logp[b].values
        X = add_constant(x)
        model = OLS(y, X).fit()
        alpha = float(model.params[0]); beta = float(model.params[1])
        resid = y - (alpha + beta * x)

        spread_vol = float(np.std(resid, ddof=1))
        if not np.isfinite(spread_vol) or spread_vol < pcfg.min_spread_vol:
            continue
        try:
            adf_stat, adf_pvalue, *_ = adfuller(resid, maxlag=pcfg.adf_maxlag, regression="c", autolag=pcfg.adf_autolag)
            adf_stat = float(adf_stat); adf_pvalue = float(adf_pvalue)
        except Exception:
            continue
        if not np.isfinite(adf_pvalue) or adf_pvalue > pcfg.max_adf_pvalue:
            continue
        hl = _half_life_bars(resid)
        if not np.isfinite(hl) or hl <= 0:
            continue
        if hl < pcfg.half_life_min_bars or hl > pcfg.half_life_max_bars:
            continue
        betas = _beta_segments(y, x, pcfg.beta_stability_segments)
        instab = _beta_instability(betas)
        if not np.isfinite(instab) or instab > pcfg.max_beta_instability:
            continue

        rows.append({
            "sym_a": a, "sym_b": b, "corr": c,
            "alpha": alpha, "beta_a_on_b": beta,
            "adf_stat": adf_stat, "adf_pvalue": adf_pvalue,
            "half_life_bars": hl, "spread_vol": spread_vol,
            "beta_instability": instab,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    z_adf = _zscore(-out["adf_stat"].values.astype(float))
    hl_pref = -np.abs(np.log(out["half_life_bars"].values.astype(float) / pcfg.half_life_target_bars))
    z_hl = _zscore(hl_pref)
    z_vol = _zscore(out["spread_vol"].values.astype(float))
    z_beta = _zscore(-out["beta_instability"].values.astype(float))
    z_corr = _zscore(out["corr"].values.astype(float))

    out["score"] = (
        pcfg.w_adf * z_adf
        + pcfg.w_half_life * z_hl
        + pcfg.w_spread_vol * z_vol
        + pcfg.w_beta_stability * z_beta
        + pcfg.w_corr * z_corr
    )
    out["pair"] = out["sym_a"] + "_" + out["sym_b"]
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def rank_pairs_by_correlation(close_panel: pd.DataFrame, top_n: int = 20, min_corr: float = 0.20) -> pd.DataFrame:
    """Fallback selector when strict screen returns no pairs. Not claimed cointegrated."""
    if close_panel.empty or close_panel.shape[1] < 2:
        return pd.DataFrame()
    logp = np.log(close_panel)
    rets = logp.diff().dropna()
    corr = rets.corr()
    rows = []
    for a, b in combinations(close_panel.columns, 2):
        c = float(corr.loc[a, b])
        if not np.isfinite(c) or c < min_corr:
            continue
        y = logp[a].values; x = logp[b].values
        model = OLS(y, add_constant(x)).fit()
        alpha = float(model.params[0]); beta = float(model.params[1])
        resid = y - (alpha + beta * x)
        rows.append({
            "pair": f"{a}_{b}", "sym_a": a, "sym_b": b, "corr": c,
            "alpha": alpha, "beta_a_on_b": beta,
            "beta_instability": np.nan, "spread_vol": float(np.std(resid, ddof=1)),
            "adf_stat": np.nan, "adf_pvalue": np.nan, "half_life_bars": np.nan,
            "score": c, "selection_method": "correlation_fallback",
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
