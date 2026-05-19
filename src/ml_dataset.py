from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS


@dataclass(frozen=True)
class DatasetConfig:
    window: int = 168
    horizon: int = 24
    classification_threshold: float = 0.001
    train_days: int = 90
    test_days: int = 30
    step_days: int = 30
    # If True, replace the fixed classification_threshold with a per-pair
    # threshold = label_scale_factor * std(|future_abs - current_abs|) computed
    # on a pair-specific training window. This normalises the 3-class label
    # distribution across pairs of different spread vol.
    #
    # Leakage note (caught in paranoia audit): an earlier version used the
    # first `label_train_fraction` of each pair's data, which extended past
    # most walk-forward test windows and so leaked future-of-test information
    # into the label threshold. The current implementation prefers an explicit
    # `label_train_end` cutoff timestamp: data with timestamp < label_train_end
    # is used for the threshold, everything after is held out. The fraction
    # fallback remains for backwards-compatibility, but the cutoff form is the
    # recommended one and is what scripts/run_first_branch.py passes when
    # --per-pair-label is set (label_train_end = t0).
    per_pair_label: bool = False
    label_scale_factor: float = 0.5
    label_train_fraction: float = 0.5  # used only when label_train_end is None
    label_train_end: Optional[pd.Timestamp] = None


TABULAR_COLUMNS = [
    "latest_spread_z",
    "mean_spread_z",
    "min_spread_z",
    "max_spread_z",
    "spread_slope",
    "latest_volume_ratio",
    "mean_volume_ratio",
    "latest_btc_return_24h",
    "latest_rolling_corr",
    "latest_realized_vol_a",
    "latest_realized_vol_b",
    "time_since_zero_crossing",
    "mean_pair_volume_sum",
    "latest_pair_volume_gmean",
]


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


def _time_since_zero_crossing(spread: np.ndarray) -> float:
    signs = np.sign(spread.astype(float))
    changes = np.where(signs[1:] * signs[:-1] <= 0)[0]
    if len(changes) == 0:
        return float(len(spread))
    return float(len(spread) - 1 - changes[-1])


def _class_label(current_abs: float, future_abs: float, threshold: float) -> int:
    delta = future_abs - current_abs
    if delta < -threshold:
        return 0  # revert
    if delta > threshold:
        return 2  # diverge
    return 1  # persist


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
            # Pick the label-training window for THIS pair.
            # Preferred: explicit timestamp cutoff (no leakage past it).
            # Fallback: fractional cutoff (kept for backwards-compat; may leak
            # if the fraction extends past walk-forward test windows).
            if cfg.label_train_end is not None:
                cutoff = pd.Timestamp(cfg.label_train_end)
                if cutoff.tzinfo is None:
                    cutoff = cutoff.tz_localize("UTC")
                pair_ts = pd.to_datetime(g["timestamp"], utc=True)
                label_train = g.loc[pair_ts < cutoff]
                # Need at least one full feature window to be meaningful.
                if len(label_train) < cfg.window:
                    label_train = g.iloc[: cfg.window]
            else:
                n_train_for_label = max(cfg.window, int(len(g) * cfg.label_train_fraction))
                label_train = g.iloc[:n_train_for_label]

            label_change = (
                (label_train["spread"] + label_train["target_spread_change_24h"]).abs()
                - label_train["spread"].abs()
            ).to_numpy(dtype=float)
            label_change = label_change[np.isfinite(label_change)]
            if len(label_change) > 1:
                pair_threshold = float(cfg.label_scale_factor * np.std(label_change, ddof=1))
            else:
                pair_threshold = cfg.classification_threshold
        else:
            pair_threshold = cfg.classification_threshold

        for end_pos in range(cfg.window - 1, len(g)):
            window_df = g.iloc[end_pos - cfg.window + 1 : end_pos + 1]
            latest = window_df.iloc[-1]
            seq = arr[end_pos - cfg.window + 1 : end_pos + 1]
            if not np.isfinite(seq).all():
                continue

            spread = window_df["spread"].to_numpy(dtype=float)
            spread_z = window_df["spread_z_168h"].to_numpy(dtype=float)
            vol_ratio = window_df["volume_ratio"].to_numpy(dtype=float)
            future_abs = abs(float(latest["spread"] + latest["target_spread_change_24h"]))
            current_abs = abs(float(latest["spread"]))

            row = {
                "sample_id": len(rows),
                "pair": pair,
                "timestamp": latest["timestamp"],
                "sym_a": latest["sym_a"],
                "sym_b": latest["sym_b"],
                "y_regression": float(latest["target_spread_change_24h"]),
                "y_reversion": int(float(latest["target_reversion_24h"])),
                "y_class": _class_label(current_abs, future_abs, pair_threshold),
                "latest_spread_z": float(spread_z[-1]),
                "mean_spread_z": float(np.mean(spread_z)),
                "min_spread_z": float(np.min(spread_z)),
                "max_spread_z": float(np.max(spread_z)),
                "spread_slope": _spread_slope(spread),
                "latest_volume_ratio": float(vol_ratio[-1]),
                "mean_volume_ratio": float(np.mean(vol_ratio)),
                "latest_btc_return_24h": float(latest["btc_return_24h"]),
                "latest_rolling_corr": float(latest["rolling_corr_24h"]),
                "latest_realized_vol_a": float(latest["realized_vol_a_24h"]),
                "latest_realized_vol_b": float(latest["realized_vol_b_24h"]),
                "time_since_zero_crossing": _time_since_zero_crossing(spread),
                "mean_pair_volume_sum": float(window_df["pair_volume_sum"].mean()),
                "latest_pair_volume_gmean": float(latest["pair_volume_gmean"]),
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
    samples: pd.DataFrame,
    sequences: np.ndarray,
    out_dir: Path,
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
    # Stringify Timestamp fields so json.dumps doesn't choke.
    cfg_dict = asdict(cfg)
    for k, v in list(cfg_dict.items()):
        if isinstance(v, pd.Timestamp):
            cfg_dict[k] = v.isoformat()
    (out_dir / "dataset_config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")


def load_dataset(dataset_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    samples = pd.read_parquet(dataset_dir / "samples.parquet", engine="pyarrow")
    seq = np.load(dataset_dir / "sequences.npz", allow_pickle=True)["X"]
    return samples, seq
