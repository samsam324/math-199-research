"""
Per-pair feature engineering at bar cadence.

Inputs:
  - pair_bars: output of `bars.build_pair_bars` indexed by bar end timestamp
  - pair_row: a row from a selected_pairs frame (sym_a, sym_b, alpha, beta_a_on_b)
  - optional spread_override: a Kalman-derived spread series indexed on the
    same bar timestamps (when --use-kalman path is set in the driver)

All feature columns are backward-rolling. Verified by spike-injection in
tests/test_leakage_audit.py.

The headline feature for the pre-registered hypothesis is
`inst_buy_imbalance_over_leg`: net institutional buy notional on the over-leg
of the spread, normalized by total notional on that leg.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    # Spread + z-score / change / vol (cadence-agnostic; lookbacks are in bars)
    "spread", "spread_z", "spread_diff_1b", "spread_diff_short", "spread_diff_long",
    "spread_vol_short", "spread_vol_long",
    # Book microstructure on each leg
    "quoted_spread_bps_a", "quoted_spread_bps_b",
    "obi_5_a", "obi_5_b",
    # Trade flow on each leg
    "signed_notional_a", "signed_notional_b",
    "trade_notional_a", "trade_notional_b",
    # Institutional flow (the pre-registered focus)
    "inst_signed_notional_a", "inst_signed_notional_b",
    "retail_signed_notional_a", "retail_signed_notional_b",
    "inst_buy_imbalance_a", "inst_buy_imbalance_b",
    "retail_buy_imbalance_a", "retail_buy_imbalance_b",
    # The pre-registered headline: net institutional buy imbalance on the over-leg
    # (positive when net institutional buy pressure aligns with the side that is
    # currently expensive vs the hedge -- the side a mean-reversion trade would short)
    "inst_buy_imbalance_over_leg",
]


@dataclass(frozen=True)
class FeatureConfig:
    # Lookbacks in BARS of the chosen cadence. At 1s cadence: short=60s, long=3600s.
    z_window_bars: int = 3600         # 1h at 1s
    short_window_bars: int = 60       # 1min at 1s
    long_window_bars: int = 600       # 10min at 1s
    diff_short_bars: int = 60         # 1min
    diff_long_bars: int = 600         # 10min
    target_horizon_bars: int = 60     # predict spread change over next 60 bars (= 1min at 1s)
    min_rows: int = 600


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=max(10, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(10, window // 4)).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    return num / denom.replace(0.0, np.nan)


def compute_pair_features(
    pair_bars: pd.DataFrame,
    pair_row: pd.Series,
    fcfg: FeatureConfig = FeatureConfig(),
    spread_override: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build per-bar features for a pair from the merged pair_bars table.

    pair_bars columns expected (suffixed _a / _b): microprice, midprice,
    quoted_spread_bps, obi_5, signed_notional, trade_notional, buy_notional,
    sell_notional, inst_buy_notional, inst_sell_notional, retail_buy_notional,
    retail_sell_notional, signed_notional_institutional, signed_notional_retail.
    """
    if pair_bars.empty:
        return pd.DataFrame()

    sym_a = str(pair_row["sym_a"])
    sym_b = str(pair_row["sym_b"])
    alpha = float(pair_row.get("alpha", 0.0))
    beta = float(pair_row["beta_a_on_b"])

    log_a = np.log(pair_bars["microprice_a"].astype(float))
    log_b = np.log(pair_bars["microprice_b"].astype(float))
    if spread_override is not None:
        spread = spread_override.reindex(pair_bars.index)
    else:
        spread = log_a - (alpha + beta * log_b)

    out = pd.DataFrame(index=pair_bars.index)
    out["pair"] = f"{sym_a}_{sym_b}"
    out["sym_a"] = sym_a
    out["sym_b"] = sym_b
    out["spread"] = spread
    out["spread_z"] = _zscore(spread, fcfg.z_window_bars)
    out["spread_diff_1b"] = spread.diff(1)
    out["spread_diff_short"] = spread.diff(fcfg.diff_short_bars)
    out["spread_diff_long"] = spread.diff(fcfg.diff_long_bars)
    out["spread_vol_short"] = spread.rolling(fcfg.short_window_bars, min_periods=max(10, fcfg.short_window_bars // 4)).std()
    out["spread_vol_long"] = spread.rolling(fcfg.long_window_bars, min_periods=max(10, fcfg.long_window_bars // 4)).std()

    out["quoted_spread_bps_a"] = pair_bars["quoted_spread_bps_a"].astype(float)
    out["quoted_spread_bps_b"] = pair_bars["quoted_spread_bps_b"].astype(float)
    out["obi_5_a"] = pair_bars["obi_5_a"].astype(float)
    out["obi_5_b"] = pair_bars["obi_5_b"].astype(float)

    out["signed_notional_a"] = pair_bars["signed_notional_a"].astype(float)
    out["signed_notional_b"] = pair_bars["signed_notional_b"].astype(float)
    out["trade_notional_a"] = pair_bars["trade_notional_a"].astype(float)
    out["trade_notional_b"] = pair_bars["trade_notional_b"].astype(float)

    out["inst_signed_notional_a"] = (pair_bars["inst_buy_notional_a"] - pair_bars["inst_sell_notional_a"]).astype(float)
    out["inst_signed_notional_b"] = (pair_bars["inst_buy_notional_b"] - pair_bars["inst_sell_notional_b"]).astype(float)
    out["retail_signed_notional_a"] = (pair_bars["retail_buy_notional_a"] - pair_bars["retail_sell_notional_a"]).astype(float)
    out["retail_signed_notional_b"] = (pair_bars["retail_buy_notional_b"] - pair_bars["retail_sell_notional_b"]).astype(float)

    # Imbalance = signed institutional flow / total notional on that leg
    out["inst_buy_imbalance_a"] = _safe_div(out["inst_signed_notional_a"], out["trade_notional_a"])
    out["inst_buy_imbalance_b"] = _safe_div(out["inst_signed_notional_b"], out["trade_notional_b"])
    out["retail_buy_imbalance_a"] = _safe_div(out["retail_signed_notional_a"], out["trade_notional_a"])
    out["retail_buy_imbalance_b"] = _safe_div(out["retail_signed_notional_b"], out["trade_notional_b"])

    # The pre-registered headline metric:
    # over-leg = the leg currently expensive relative to its hedge.
    # If spread > 0 (=log_a above expected), leg A is the over-leg.
    # inst_buy_imbalance_over_leg = institutional buy imbalance on the over-leg.
    # A mean-reversion entry would short the over-leg; persistent institutional
    # buying on the over-leg should predict CONTINUED divergence (negative for the
    # reversion hypothesis). A positive prediction for reversion comes from the
    # NEGATIVE of this signal -- i.e. institutional SELLING on the over-leg.
    over_is_a = (spread > 0).astype(float)
    out["inst_buy_imbalance_over_leg"] = (
        over_is_a * out["inst_buy_imbalance_a"] + (1 - over_is_a) * out["inst_buy_imbalance_b"]
    )

    # Targets
    future_spread = spread.shift(-fcfg.target_horizon_bars)
    out["target_spread_change"] = future_spread - spread
    out["target_abs_spread_change"] = future_spread.abs() - spread.abs()
    out["target_reversion"] = (future_spread.abs() < spread.abs()).astype(float)
    out["timestamp"] = out.index
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=FEATURE_COLUMNS + ["target_spread_change", "target_reversion"])
    if len(out) < fcfg.min_rows:
        return pd.DataFrame()
    return out.reset_index(drop=True)


def build_feature_store(
    pair_bars_by_pair: Dict[str, pd.DataFrame],
    pairs: pd.DataFrame,
    out_dir: Path,
    fcfg: FeatureConfig = FeatureConfig(),
    top_pairs: Optional[int] = None,
    spread_overrides: Optional[Dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = pairs.head(top_pairs) if top_pairs else pairs
    overrides = spread_overrides or {}
    frames: List[pd.DataFrame] = []
    for _, pair_row in selected.iterrows():
        pair_name = f"{pair_row['sym_a']}_{pair_row['sym_b']}"
        if pair_name not in pair_bars_by_pair:
            continue
        bars = pair_bars_by_pair[pair_name]
        override = overrides.get(pair_name)
        features = compute_pair_features(bars, pair_row, fcfg=fcfg, spread_override=override)
        if features.empty:
            continue
        features.to_parquet(out_dir / f"{pair_name}.parquet", engine="pyarrow", index=False)
        frames.append(features)
    if not frames:
        return pd.DataFrame()
    all_features = pd.concat(frames, axis=0, ignore_index=True)
    all_features.to_parquet(out_dir / "all_pair_features.parquet", engine="pyarrow", index=False)
    return all_features


def build_kalman_spread_overrides(
    pair_bars_by_pair: Dict[str, pd.DataFrame],
    pairs: pd.DataFrame,
    train_end: pd.Timestamp,
    top_pairs: Optional[int] = None,
) -> Dict[str, pd.Series]:
    """
    Per pair: MLE-fit Kalman on bars with timestamp < train_end (using log
    microprice as the price series), then forward-roll on bars >= train_end
    with fitted params + trained final state. Returns the combined residual
    series indexed on bar timestamps. Same parameters across the boundary
    (audit fix carried over from phase 1).
    """
    from src.kalman_hedge import fit_kalman_mle, kalman_forward_residuals

    train_end = pd.Timestamp(train_end)
    if train_end.tzinfo is None:
        train_end = train_end.tz_localize("UTC")
    selected = pairs.head(top_pairs) if top_pairs else pairs
    out: Dict[str, pd.Series] = {}

    for _, pair_row in selected.iterrows():
        sym_a, sym_b = pair_row["sym_a"], pair_row["sym_b"]
        pair_name = f"{sym_a}_{sym_b}"
        if pair_name not in pair_bars_by_pair:
            continue
        bars = pair_bars_by_pair[pair_name]
        sub = np.log(bars[["microprice_a", "microprice_b"]].astype(float)).dropna()
        if sub.empty:
            continue
        train = sub[sub.index < train_end]
        test = sub[sub.index >= train_end]
        if len(train) < 200:
            continue
        try:
            fitted = fit_kalman_mle(train["microprice_a"].to_numpy(), train["microprice_b"].to_numpy())
        except Exception:
            continue
        train_resid = np.asarray(fitted["train_residuals"], dtype=float)
        if not test.empty:
            _, _, test_resid = kalman_forward_residuals(
                test["microprice_a"].to_numpy(),
                test["microprice_b"].to_numpy(),
                fitted,
            )
        else:
            test_resid = np.empty(0, dtype=float)
        combined = np.concatenate([train_resid, test_resid])
        out[pair_name] = pd.Series(combined, index=sub.index, name="spread")
    return out
