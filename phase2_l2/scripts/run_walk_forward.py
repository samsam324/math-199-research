"""
Walk-forward evaluation on the L2 bar dataset.

Mirrors phase 1's scripts/run_walk_forward.py shape: rolling train/test
windows in calendar time, per-split train-from-scratch, optional deep models.
Default cadence is days (5/1/1), suitable for 1s bars with multi-week data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_dataset import TABULAR_COLUMNS, load_dataset
from src.modeling import (
    TrainingConfig, _metrics_for_predictions,
    baseline_majority_class, baseline_persist_class, baseline_random_stratified, baseline_zscore_rule,
    train_booster_on_split, train_lstm_on_split, train_transformer_on_split,
    combine_results,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward evaluation on the L2 bar dataset.")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-days", type=float, default=5.0)
    p.add_argument("--test-days", type=float, default=1.0)
    p.add_argument("--step-days", type=float, default=1.0)
    p.add_argument("--max-train-samples", type=int, default=200000)
    p.add_argument("--max-test-samples", type=int, default=50000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--deep", action="store_true")
    p.add_argument("--dl-epochs", type=int, default=4)
    return p.parse_args()


def _tail_by_time(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sort_values("timestamp").tail(max_rows)


def _windows(samples: pd.DataFrame, train_days: float, test_days: float, step_days: float) -> Iterable[Tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    min_ts = samples["timestamp"].min().floor("D")
    max_ts = samples["timestamp"].max().ceil("D")
    train_delta = pd.Timedelta(days=train_days)
    test_delta = pd.Timedelta(days=test_days)
    step_delta = pd.Timedelta(days=step_days)
    split_id = 0
    train_start = min_ts
    while train_start + train_delta + test_delta <= max_ts:
        train_end = train_start + train_delta
        test_end = train_end + test_delta
        yield split_id, train_start, train_end, test_end
        split_id += 1
        train_start += step_delta


def _baselines(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> List[Tuple[pd.DataFrame, Dict[str, float]]]:
    return [
        baseline_persist_class(test),
        baseline_majority_class(train, test),
        baseline_random_stratified(train, test, seed=seed),
        baseline_zscore_rule(test),
    ]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    samples, sequences = load_dataset(Path(args.dataset_dir))
    samples = samples.sort_values("timestamp").copy()
    samples["timestamp"] = pd.to_datetime(samples["timestamp"], utc=True)
    cfg = TrainingConfig(
        max_train_samples=args.max_train_samples, max_test_samples=args.max_test_samples,
        dl_epochs=args.dl_epochs, seed=args.seed,
    )

    results: List[Tuple[pd.DataFrame, Dict[str, float]]] = []
    split_rows: List[Dict[str, object]] = []
    for split_id, train_start, train_end, test_end in _windows(samples, args.train_days, args.test_days, args.step_days):
        train = samples[(samples["timestamp"] >= train_start) & (samples["timestamp"] < train_end)]
        test = samples[(samples["timestamp"] >= train_end) & (samples["timestamp"] < test_end)]
        train = _tail_by_time(train, cfg.max_train_samples)
        test = _tail_by_time(test, cfg.max_test_samples)
        if train.empty or test.empty:
            continue

        split_results = _baselines(train, test, cfg.seed)
        split_results.append(train_booster_on_split(train, test, TABULAR_COLUMNS))
        if args.deep:
            print(f"  split {split_id}: training LSTM..."); split_results.append(train_lstm_on_split(train, test, sequences, cfg))
            print(f"  split {split_id}: training transformer..."); split_results.append(train_transformer_on_split(train, test, sequences, cfg))

        for pred_df, metrics in split_results:
            pred_df = pred_df.copy(); pred_df["walk_split"] = split_id
            metrics = dict(metrics); metrics.update({"walk_split": split_id, "train_start": train_start, "train_end": train_end, "test_end": test_end})
            results.append((pred_df, metrics))
        split_rows.append({"walk_split": split_id, "train_start": train_start, "train_end": train_end, "test_end": test_end, "train_samples": len(train), "test_samples": len(test)})

    if not results:
        raise RuntimeError("No valid walk-forward windows found.")

    preds, metrics = combine_results(results)
    preds.to_parquet(out_dir / "walk_forward_predictions.parquet", engine="pyarrow", index=False)
    metrics.to_csv(out_dir / "walk_forward_metrics_by_split.csv", index=False)
    pd.DataFrame(split_rows).to_csv(out_dir / "walk_forward_splits.csv", index=False)

    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            splits=("walk_split", "nunique"),
            train_samples_mean=("train_samples", "mean"),
            test_samples_total=("test_samples", "sum"),
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            trades_total=("trades", "sum"),
            total_pnl_sum=("total_pnl", "sum"),
            pnl_mean_to_std_mean=("pnl_mean_to_std", "mean"),
            max_drawdown_min=("max_drawdown", "min"),
            win_rate_mean=("win_rate", "mean"),
        )
        .sort_values(["total_pnl_sum", "macro_f1_mean"], ascending=False)
    )
    summary.to_csv(out_dir / "walk_forward_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved walk-forward outputs under {out_dir}")


if __name__ == "__main__":
    main()
