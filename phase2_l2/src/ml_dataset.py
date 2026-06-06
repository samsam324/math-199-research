"""
Sliding-window dataset construction with 3-class label.

Window: W consecutive bars of FEATURE_COLUMNS values.
Horizon: target_horizon_bars ahead, set in FeatureConfig.

3-class label (same convention as phase 1):
  0 = revert    (|future_spread| - |current_spread| < -tau)
  2 = diverge   (|future_spread| - |current_spread| > +tau)
  1 = persist   (otherwise)

tau is either fixed (default 1e-5 in log-microprice units; smaller than phase
1 because microprice is much smoother than 1h close) or per-pair, computed
on data strictly before `label_train_end`. Same audit-fix as phase 1.

TABULAR_COLUMNS is the input to the booster. It mixes microstructure features
and summary statistics over the window.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS


TABULAR_COLUMNS = [
    "latest_spread_z",
    "mean_spread_z",
    "min_spread_z",
    "max_spread_z",
    "spread_slope",
    "latest_quoted_spread_bps_a",
    "latest_quoted_spread_bps_b",
    "latest_obi_5_a",
    "latest_obi_5_b",
    "mean_inst_buy_imbalance_over_leg",
    "latest_inst_buy_imbalance_over_leg",
    "mean_signed_notional_a",
    "mean_signed_notional_b",
    "mean_inst_signed_notional_a",
    "mean_inst_signed_notional_b",
    "mean_retail_signed_notional_a",
    "mean_retail_signed_notional_b",
]


@dataclass(frozen=True)
class DatasetConfig:
    window: int = 120                # 2 minutes of 1s bars
    horizon: int = 60                # 1 minute ahead
    classification_threshold: float = 1e-5
    train_days: float = 5.0
    test_days: float = 1.0
    step_days: float = 1.0
    per_pair_label: bool = False
    label_scale_factor: float = 0.5
    label_train_fraction: float = 0.5
    label_train_end: Optional[pd.Timestamp] = None


def _spread_slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    y = values.astype(float)
    if len(y) < 2 or not np.isfinite(y).all():
        return np.nan
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.nan
    return float(np.dot(x, y - y.mean()) / denom)


def _class_label(current_abs: float, future_abs: float, threshold: float) -> int:
    delta = future_abs - current_abs
    if delta < -threshold:
        return 0
    if delta > threshold:
        return 2
    return 1


def build_samples_from_features(
    features: pd.DataFrame,
    cfg: DatasetConfig = DatasetConfig(),
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows: List[Dict[str, object]] = []
    sequences: List[np.ndarray] = []

    features = features.sort_values(["pair", "timestamp"]).copy()
    for pair, g in features.groupby("pair", sort=False):
        g = g.reset_index(drop=True)
        arr = g[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

        if cfg.per_pair_label:
            if cfg.label_train_end is not None:
                cutoff = pd.Timestamp(cfg.label_train_end)
                if cutoff.tzinfo is None:
                    cutoff = cutoff.tz_localize("UTC")
                pair_ts = pd.to_datetime(g["timestamp"], utc=True)
                label_train = g.loc[pair_ts < cutoff]
                if len(label_train) < cfg.window:
                    label_train = g.iloc[: cfg.window]
            else:
                n_train_for_label = max(cfg.window, int(len(g) * cfg.label_train_fraction))
                label_train = g.iloc[:n_train_for_label]

            label_change = (
                (label_train["spread"] + label_train["target_spread_change"]).abs()
                - label_train["spread"].abs()
            ).to_numpy(dtype=float)
            label_change = label_change[np.isfinite(label_change)]
            pair_threshold = float(cfg.label_scale_factor * np.std(label_change, ddof=1)) if len(label_change) > 1 else cfg.classification_threshold
        else:
            pair_threshold = cfg.classification_threshold

        for end_pos in range(cfg.window - 1, len(g)):
            window_df = g.iloc[end_pos - cfg.window + 1 : end_pos + 1]
            latest = window_df.iloc[-1]
            seq = arr[end_pos - cfg.window + 1 : end_pos + 1]
            if not np.isfinite(seq).all():
                continue

            spread = window_df["spread"].to_numpy(dtype=float)
            spread_z = window_df["spread_z"].to_numpy(dtype=float)
            future_abs = abs(float(latest["spread"] + latest["target_spread_change"]))
            current_abs = abs(float(latest["spread"]))

            row = {
                "sample_id": len(rows),
                "pair": pair,
                "timestamp": latest["timestamp"],
                "sym_a": latest["sym_a"],
                "sym_b": latest["sym_b"],
                "y_regression": float(latest["target_spread_change"]),
                "y_reversion": int(float(latest["target_reversion"])),
                "y_class": _class_label(current_abs, future_abs, pair_threshold),
                "latest_spread_z": float(spread_z[-1]),
                "mean_spread_z": float(np.mean(spread_z)),
                "min_spread_z": float(np.min(spread_z)),
                "max_spread_z": float(np.max(spread_z)),
                "spread_slope": _spread_slope(spread),
                "latest_quoted_spread_bps_a": float(latest["quoted_spread_bps_a"]),
                "latest_quoted_spread_bps_b": float(latest["quoted_spread_bps_b"]),
                "latest_obi_5_a": float(latest["obi_5_a"]),
                "latest_obi_5_b": float(latest["obi_5_b"]),
                "mean_inst_buy_imbalance_over_leg": float(np.mean(window_df["inst_buy_imbalance_over_leg"].to_numpy(dtype=float))),
                "latest_inst_buy_imbalance_over_leg": float(latest["inst_buy_imbalance_over_leg"]),
                "mean_signed_notional_a": float(window_df["signed_notional_a"].mean()),
                "mean_signed_notional_b": float(window_df["signed_notional_b"].mean()),
                "mean_inst_signed_notional_a": float(window_df["inst_signed_notional_a"].mean()),
                "mean_inst_signed_notional_b": float(window_df["inst_signed_notional_b"].mean()),
                "mean_retail_signed_notional_a": float(window_df["retail_signed_notional_a"].mean()),
                "mean_retail_signed_notional_b": float(window_df["retail_signed_notional_b"].mean()),
                "current_spread": float(latest["spread"]),
            }
            if not np.isfinite([row[c] for c in TABULAR_COLUMNS + ["y_regression", "current_spread"]]).all():
                continue
            rows.append(row)
            sequences.append(seq)

    samples = pd.DataFrame(rows)
    if samples.empty:
        return samples, np.empty((0, cfg.window, len(FEATURE_COLUMNS)), dtype=np.float32), np.empty((0,), dtype=np.int64)
    seq_array = np.stack(sequences).astype(np.float32)
    y_class = samples["y_class"].to_numpy(dtype=np.int64)
    return samples, seq_array, y_class


def add_walk_forward_splits(samples: pd.DataFrame, cfg: DatasetConfig = DatasetConfig()) -> pd.DataFrame:
    if samples.empty:
        return samples
    out = samples.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    min_ts = out["timestamp"].min().floor("D")
    max_ts = out["timestamp"].max().ceil("D")
    split_ids = [[] for _ in range(len(out))]
    split_id = 0
    train_delta = pd.Timedelta(days=cfg.train_days)
    test_delta = pd.Timedelta(days=cfg.test_days)
    step_delta = pd.Timedelta(days=cfg.step_days)
    train_start = min_ts
    while train_start + train_delta + test_delta <= max_ts:
        train_end = train_start + train_delta
        test_end = train_end + test_delta
        mask = (out["timestamp"] >= train_start) & (out["timestamp"] < test_end)
        for idx in out.index[mask]:
            split_ids[idx].append(split_id)
        split_id += 1
        train_start += step_delta
    out["walk_split_ids"] = [json.dumps(v) for v in split_ids]
    return out


def save_dataset(
    samples: pd.DataFrame, sequences: np.ndarray, out_dir: Path,
    cfg: DatasetConfig = DatasetConfig(),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(out_dir / "samples.parquet", engine="pyarrow", index=False)
    np.savez_compressed(
        out_dir / "sequences.npz",
        X=sequences.astype(np.float32),
        y_class=samples["y_class"].to_numpy(dtype=np.int64) if not samples.empty else np.empty((0,), dtype=np.int64),
        y_regression=samples["y_regression"].to_numpy(dtype=np.float32) if not samples.empty else np.empty((0,), dtype=np.float32),
        feature_columns=np.asarray(FEATURE_COLUMNS, dtype=object),
    )
    cfg_dict = asdict(cfg)
    for k, v in list(cfg_dict.items()):
        if isinstance(v, pd.Timestamp):
            cfg_dict[k] = v.isoformat()
    (out_dir / "dataset_config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")


def load_dataset(dataset_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    samples = pd.read_parquet(dataset_dir / "samples.parquet", engine="pyarrow")
    seq = np.load(dataset_dir / "sequences.npz", allow_pickle=True)["X"]
    return samples, seq
