from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.data_store import StoreConfig, load_symbol


FEATURE_COLUMNS = [
    "spread",
    "spread_z_168h",
    "spread_diff_1h",
    "spread_diff_4h",
    "spread_diff_24h",
    "spread_vol_24h",
    "spread_vol_168h",
    "volume_ratio",
    "volume_z_a_168h",
    "volume_z_b_168h",
    "pair_volume_sum",
    "pair_volume_gmean",
    "btc_return_24h",
    "rolling_corr_24h",
    "realized_vol_a_24h",
    "realized_vol_b_24h",
]


@dataclass(frozen=True)
class FeatureConfig:
    z_window: int = 168
    short_vol_window: int = 24
    corr_window: int = 24
    target_horizon: int = 24
    min_rows: int = 240


def _to_utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _calendar(cfg: StoreConfig, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DatetimeIndex:
    start = _to_utc(start).floor(cfg.interval)
    end_exclusive = _to_utc(end_exclusive).floor(cfg.interval)
    if end_exclusive <= start:
        return pd.DatetimeIndex([], tz="UTC", name="timestamp")
    step = pd.tseries.frequencies.to_offset(cfg.interval)
    return pd.date_range(start=start, end=end_exclusive - step, freq=cfg.interval, tz="UTC", name="timestamp")


def load_ohlcv_panel(
    cfg: StoreConfig,
    symbols: Iterable[str],
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    min_symbol_coverage: float = 0.80,
) -> Dict[str, pd.DataFrame]:
    """
    Load close and USDT-volume panels on a common hourly calendar.

    Volume is converted to quote terms as close * base volume, matching the
    liquidity filter used for pair selection.
    """
    idx = _calendar(cfg, start, end_exclusive)
    if idx.empty:
        return {"close": pd.DataFrame(), "volume_usdt": pd.DataFrame()}

    close_frames = []
    volume_frames = []
    for sym in symbols:
        df = load_symbol(cfg, sym, columns=["close", "volume"])
        if df is None or df.empty:
            continue
        w = df.loc[(df.index >= idx.min()) & (df.index <= idx.max()), ["close", "volume"]]
        if w.empty:
            continue
        w = w.reindex(idx)
        coverage = float(w["close"].notna().mean())
        if coverage < min_symbol_coverage:
            continue
        close_frames.append(w[["close"]].rename(columns={"close": sym}))
        volume_usdt = (w["close"] * w["volume"]).rename(sym)
        volume_frames.append(volume_usdt.to_frame())

    if not close_frames:
        return {"close": pd.DataFrame(index=idx), "volume_usdt": pd.DataFrame(index=idx)}

    close = pd.concat(close_frames, axis=1).sort_index().ffill()
    volume_usdt = pd.concat(volume_frames, axis=1).sort_index().fillna(0.0)
    valid_cols = close.dropna(axis=1, how="all").columns
    close = close[valid_cols].dropna(axis=0, how="any")
    volume_usdt = volume_usdt.reindex(close.index)[valid_cols].fillna(0.0)
    return {"close": close, "volume_usdt": volume_usdt}


def compute_pair_features(
    close: pd.DataFrame,
    volume_usdt: pd.DataFrame,
    pair_row: pd.Series,
    fcfg: FeatureConfig = FeatureConfig(),
    btc_symbol: str = "BTCUSDT",
    spread_override: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build per-bar features for a pair.

    If spread_override is provided (e.g. Kalman dynamic residuals indexed by
    timestamp), it is used as the spread series directly. Otherwise the
    static OLS spread = log_a - (alpha + beta * log_b) is computed from
    pair_row's alpha and beta.
    """
    sym_a = str(pair_row["sym_a"])
    sym_b = str(pair_row["sym_b"])
    alpha = float(pair_row.get("alpha", 0.0))
    beta = float(pair_row["beta_a_on_b"])

    needed = [sym_a, sym_b]
    if any(s not in close.columns for s in needed) or any(s not in volume_usdt.columns for s in needed):
        return pd.DataFrame()

    log_a = np.log(close[sym_a].astype(float))
    log_b = np.log(close[sym_b].astype(float))
    if spread_override is not None:
        spread = spread_override.reindex(close.index)
    else:
        spread = log_a - (alpha + beta * log_b)
    spread_mean = spread.rolling(fcfg.z_window, min_periods=max(24, fcfg.z_window // 4)).mean()
    spread_std = spread.rolling(fcfg.z_window, min_periods=max(24, fcfg.z_window // 4)).std()

    vol_a = volume_usdt[sym_a].astype(float)
    vol_b = volume_usdt[sym_b].astype(float)
    vol_a_mean = vol_a.rolling(fcfg.z_window, min_periods=24).mean()
    vol_b_mean = vol_b.rolling(fcfg.z_window, min_periods=24).mean()
    vol_a_std = vol_a.rolling(fcfg.z_window, min_periods=24).std()
    vol_b_std = vol_b.rolling(fcfg.z_window, min_periods=24).std()

    ret_a = log_a.diff()
    ret_b = log_b.diff()
    if btc_symbol in close.columns:
        btc_return_24h = np.log(close[btc_symbol].astype(float)).diff(24)
    else:
        btc_return_24h = pd.Series(0.0, index=close.index)

    out = pd.DataFrame(index=close.index)
    out["pair"] = f"{sym_a}_{sym_b}"
    out["sym_a"] = sym_a
    out["sym_b"] = sym_b
    out["spread"] = spread
    out["spread_z_168h"] = (spread - spread_mean) / spread_std.replace(0.0, np.nan)
    out["spread_diff_1h"] = spread.diff(1)
    out["spread_diff_4h"] = spread.diff(4)
    out["spread_diff_24h"] = spread.diff(24)
    out["spread_vol_24h"] = spread.rolling(fcfg.short_vol_window, min_periods=12).std()
    out["spread_vol_168h"] = spread_std
    out["volume_ratio"] = vol_a / vol_b.replace(0.0, np.nan)
    out["volume_z_a_168h"] = (vol_a - vol_a_mean) / vol_a_std.replace(0.0, np.nan)
    out["volume_z_b_168h"] = (vol_b - vol_b_mean) / vol_b_std.replace(0.0, np.nan)
    out["pair_volume_sum"] = vol_a + vol_b
    out["pair_volume_gmean"] = np.sqrt((vol_a.clip(lower=0.0) * vol_b.clip(lower=0.0)).astype(float))
    out["btc_return_24h"] = btc_return_24h
    out["rolling_corr_24h"] = ret_a.rolling(fcfg.corr_window, min_periods=12).corr(ret_b)
    out["realized_vol_a_24h"] = ret_a.rolling(fcfg.short_vol_window, min_periods=12).std()
    out["realized_vol_b_24h"] = ret_b.rolling(fcfg.short_vol_window, min_periods=12).std()

    future_spread = spread.shift(-fcfg.target_horizon)
    out["target_spread_change_24h"] = future_spread - spread
    out["target_abs_spread_change_24h"] = future_spread.abs() - spread.abs()
    out["target_reversion_24h"] = (future_spread.abs() < spread.abs()).astype(float)
    out["timestamp"] = out.index
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=FEATURE_COLUMNS + ["target_spread_change_24h", "target_reversion_24h"])
    if len(out) < fcfg.min_rows:
        return pd.DataFrame()
    return out.reset_index(drop=True)


def build_feature_store(
    close: pd.DataFrame,
    volume_usdt: pd.DataFrame,
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
        override = overrides.get(pair_name)
        features = compute_pair_features(close, volume_usdt, pair_row, fcfg=fcfg, spread_override=override)
        if features.empty:
            continue
        pair_name = str(features["pair"].iloc[0])
        features.to_parquet(out_dir / f"{pair_name}.parquet", engine="pyarrow", index=False)
        frames.append(features)
    if not frames:
        return pd.DataFrame()
    all_features = pd.concat(frames, axis=0, ignore_index=True)
    all_features.to_parquet(out_dir / "all_pair_features.parquet", engine="pyarrow", index=False)
    return all_features


def build_kalman_spread_overrides(
    close: pd.DataFrame,
    pairs: pd.DataFrame,
    train_end: pd.Timestamp,
    top_pairs: Optional[int] = None,
) -> Dict[str, pd.Series]:
    """
    For each pair: fit Kalman MLE on close history strictly before train_end,
    then forward-roll the filter across the full available history to produce
    a dynamic-beta residual series indexed on close.index.

    Both the training-period and test-period residuals are computed with the
    SAME (MLE-fitted) Q_alpha, Q_beta, R parameters. This is essential to
    avoid a parameter discontinuity at the train/test boundary: an earlier
    version of this function used kalman_dynamic_hedge() with default
    parameters for the training period, which meant the spread series jumped
    at train_end as the filter switched from default to fitted Q. That was
    a real methodological bug (caught in the paranoia audit).

    The training residuals are the innovations from the same forward pass
    that fit_kalman_mle runs after optimization to record the final state,
    so they ARE the fitted-parameter residuals. The MLE fit itself uses only
    training data, so no leakage into the test residuals.
    """
    from src.kalman_hedge import fit_kalman_mle, kalman_forward_residuals  # local import

    train_end = _to_utc(train_end)
    selected = pairs.head(top_pairs) if top_pairs else pairs
    out: Dict[str, pd.Series] = {}

    for _, pair_row in selected.iterrows():
        sym_a, sym_b = pair_row["sym_a"], pair_row["sym_b"]
        if sym_a not in close.columns or sym_b not in close.columns:
            continue
        sub = np.log(close[[sym_a, sym_b]].astype(float)).dropna()
        if sub.empty:
            continue
        train = sub[sub.index < train_end]
        test = sub[sub.index >= train_end]
        if len(train) < 200:
            continue
        try:
            fitted = fit_kalman_mle(train[sym_a].to_numpy(), train[sym_b].to_numpy())
        except Exception:
            continue

        # Training residuals: computed inside fit_kalman_mle with FITTED params
        # during the final forward pass that records the final state. Same
        # parameters as the OOS pass, so no boundary discontinuity.
        train_resid = np.asarray(fitted["train_residuals"], dtype=float)

        # OOS residuals: forward-roll on test using fitted params + trained state.
        if not test.empty:
            _, _, test_resid = kalman_forward_residuals(
                test[sym_a].to_numpy(),
                test[sym_b].to_numpy(),
                fitted,
            )
        else:
            test_resid = np.empty(0, dtype=float)
        combined = np.concatenate([train_resid, test_resid])
        series = pd.Series(combined, index=sub.index, name="spread")
        out[f"{sym_a}_{sym_b}"] = series
    return out
