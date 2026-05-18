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
) -> pd.DataFrame:
    sym_a = str(pair_row["sym_a"])
    sym_b = str(pair_row["sym_b"])
    alpha = float(pair_row.get("alpha", 0.0))
    beta = float(pair_row["beta_a_on_b"])

    needed = [sym_a, sym_b]
    if any(s not in close.columns for s in needed) or any(s not in volume_usdt.columns for s in needed):
        return pd.DataFrame()

    log_a = np.log(close[sym_a].astype(float))
    log_b = np.log(close[sym_b].astype(float))
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
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = pairs.head(top_pairs) if top_pairs else pairs
    frames: List[pd.DataFrame] = []
    for _, pair_row in selected.iterrows():
        features = compute_pair_features(close, volume_usdt, pair_row, fcfg=fcfg)
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
