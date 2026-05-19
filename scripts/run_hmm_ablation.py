"""
Apply the HMM regime filter to saved walk-forward predictions and report a
side-by-side comparison: raw model metrics vs. HMM-filtered metrics.

The script fits a 2-state Gaussian HMM per pair on the training slice of each
walk-forward split, identifies the mean-reverting state, and suppresses
predictions outside that state to flat (class 1). Trading metrics are then
recomputed on the filtered predictions and compared to the original.

Inputs:
  --walk-forward-dir  directory containing walk_forward_predictions.parquet,
                      walk_forward_splits.csv, and walk_forward_metrics_by_split.csv
  --features-dir      directory containing all_pair_features.parquet
  --dataset-dir       directory containing samples.parquet (for pair/timestamp lookup)
  --out-dir           where to write the ablation results

Outputs:
  ablation_predictions.parquet   filtered predictions with regime mask per row
  ablation_metrics_by_split.csv  per-split metrics, raw vs filtered
  ablation_summary.csv           summary metrics per model, raw vs filtered
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hmm_filter import (
    HMMConfig,
    HMM_FEATURE_COLUMNS,
    apply_regime_filter,
    regime_mask_for_samples,
)
from src.ml_dataset import load_dataset
from src.modeling import evaluate_predictions, trading_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HMM regime filter ablation on walk-forward predictions.")
    p.add_argument("--walk-forward-dir", required=True)
    p.add_argument("--features-dir", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-states", type=int, default=2)
    p.add_argument("--n-iter", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-init", type=int, default=1, help="Multiple random starts per HMM fit (keep best converged LL)")
    p.add_argument("--no-require-converged", action="store_true", help="Accept non-converged fits (default: prefer converged)")
    return p.parse_args()


def _metrics_row(model: str, split: int, suffix: str, y_true: np.ndarray, pred_df: pd.DataFrame, pred: np.ndarray) -> Dict[str, float]:
    row = {"model": model, "walk_split": split, "variant": suffix}
    row.update(evaluate_predictions(y_true, pred))
    row.update(trading_metrics(pred_df, pred))
    return row


def main() -> None:
    args = parse_args()
    walk_dir = Path(args.walk_forward_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = walk_dir / "walk_forward_predictions.parquet"
    splits_path = walk_dir / "walk_forward_splits.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing {predictions_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing {splits_path}")

    preds = pd.read_parquet(predictions_path, engine="pyarrow")
    splits = pd.read_csv(splits_path, parse_dates=["train_start", "train_end", "test_end"])
    for col in ("train_start", "train_end", "test_end"):
        splits[col] = pd.to_datetime(splits[col], utc=True)

    features_path = Path(args.features_dir) / "all_pair_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}")
    features = pd.read_parquet(features_path, engine="pyarrow")
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)

    samples, _ = load_dataset(Path(args.dataset_dir))
    samples["timestamp"] = pd.to_datetime(samples["timestamp"], utc=True)

    cfg = HMMConfig(
        n_states=args.n_states,
        n_iter=args.n_iter,
        seed=args.seed,
        n_init=args.n_init,
        require_converged=not args.no_require_converged,
    )

    metric_rows: List[Dict[str, float]] = []
    filtered_pred_frames: List[pd.DataFrame] = []

    for split_id in sorted(preds["walk_split"].unique()):
        split_meta = splits[splits["walk_split"] == split_id].iloc[0]
        train_end = pd.Timestamp(split_meta["train_end"])
        if train_end.tzinfo is None:
            train_end = train_end.tz_localize("UTC")

        split_preds = preds[preds["walk_split"] == split_id].copy()
        split_preds["timestamp"] = pd.to_datetime(split_preds["timestamp"], utc=True)

        relevant_pairs = split_preds["pair"].unique().tolist()
        split_features = features[features["pair"].isin(relevant_pairs)]
        mask = regime_mask_for_samples(
            features=split_features,
            samples=split_preds[["sample_id", "pair", "timestamp"]].rename_axis("idx"),
            cfg=cfg,
            columns=HMM_FEATURE_COLUMNS,
            split_train_end=train_end,
        )
        split_preds["mean_reverting_state"] = mask.reindex(split_preds.index).fillna(False).to_numpy(dtype=bool)

        for model_name in split_preds["model"].unique():
            sub = split_preds[split_preds["model"] == model_name]
            raw_pred = sub["pred_class"].to_numpy(dtype=int)
            filt_pred = apply_regime_filter(raw_pred, sub["mean_reverting_state"].to_numpy(dtype=bool))
            y_true = sub["y_class"].to_numpy(dtype=int)

            metric_rows.append(_metrics_row(model_name, int(split_id), "raw", y_true, sub, raw_pred))
            metric_rows.append(_metrics_row(model_name, int(split_id), "hmm_filtered", y_true, sub, filt_pred))

            out_frame = sub.copy()
            out_frame["pred_class_filtered"] = filt_pred
            filtered_pred_frames.append(out_frame)

    if not metric_rows:
        raise RuntimeError("No metrics produced; check inputs.")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "ablation_metrics_by_split.csv", index=False)

    summary = (
        metrics_df.groupby(["model", "variant"], as_index=False)
        .agg(
            splits=("walk_split", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            trades_total=("trades", "sum"),
            total_pnl_sum=("total_pnl", "sum"),
            pnl_mean_to_std_mean=("pnl_mean_to_std", "mean"),
            max_drawdown_min=("max_drawdown", "min"),
            win_rate_mean=("win_rate", "mean"),
        )
        .sort_values(["model", "variant"])
    )
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)

    pd.concat(filtered_pred_frames, ignore_index=True).to_parquet(
        out_dir / "ablation_predictions.parquet", engine="pyarrow", index=False
    )

    print(summary.to_string(index=False))
    print(f"Saved HMM ablation outputs under {out_dir}")


if __name__ == "__main__":
    main()
