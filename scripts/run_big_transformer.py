"""
Train the bigger transformer on a saved dataset and report metrics.

Two modes:
  --mode single  : single chronological train/test split (fast sanity check)
  --mode walk    : walk-forward across all splits (the rigorous test)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_dataset import load_dataset
from src.modeling import TrainingConfig, _balanced_tail, _split_frame, combine_results
from src.modeling_big_transformer import BigTransformerConfig, train_big_transformer_on_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Big transformer (4 layers, 4 heads, d_model 128) walk-forward.")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["single", "walk"], default="walk")
    p.add_argument("--train-days", type=int, default=90)
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--step-days", type=int, default=30)
    p.add_argument("--max-train-samples", type=int, default=12000)
    p.add_argument("--max-test-samples", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--ff", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def _windows(samples: pd.DataFrame, train_days: int, test_days: int, step_days: int) -> Iterable[Tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    min_ts = samples["timestamp"].min().floor("D")
    max_ts = samples["timestamp"].max().ceil("D")
    train_delta = pd.Timedelta(days=train_days)
    test_delta = pd.Timedelta(days=test_days)
    step_delta = pd.Timedelta(days=step_days)
    split_id = 0
    ts = min_ts
    while ts + train_delta + test_delta <= max_ts:
        yield split_id, ts, ts + train_delta, ts + train_delta + test_delta
        split_id += 1
        ts += step_delta


def _tail_by_time(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sort_values("timestamp").tail(max_rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, sequences = load_dataset(Path(args.dataset_dir))
    samples = samples.sort_values("timestamp").copy()
    samples["timestamp"] = pd.to_datetime(samples["timestamp"], utc=True)

    big_cfg = BigTransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff=args.ff,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    base_cfg = TrainingConfig(
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )

    if args.mode == "single":
        train, test = _split_frame(samples)
        train = _balanced_tail(train, base_cfg.max_train_samples)
        test = _balanced_tail(test, base_cfg.max_test_samples)
        pred_df, metrics = train_big_transformer_on_split(train, test, sequences, big_cfg)
        pd.DataFrame([metrics]).to_csv(out_dir / "single_split_metrics.csv", index=False)
        pred_df.to_parquet(out_dir / "single_split_predictions.parquet", engine="pyarrow", index=False)
        print(pd.DataFrame([metrics]).to_string(index=False))
        print(f"Saved {out_dir}")
        return

    # walk-forward
    results: List[Tuple[pd.DataFrame, Dict[str, float]]] = []
    split_rows: List[Dict[str, object]] = []
    for split_id, train_start, train_end, test_end in _windows(samples, args.train_days, args.test_days, args.step_days):
        train = samples[(samples["timestamp"] >= train_start) & (samples["timestamp"] < train_end)]
        test = samples[(samples["timestamp"] >= train_end) & (samples["timestamp"] < test_end)]
        train = _tail_by_time(train, base_cfg.max_train_samples)
        test = _tail_by_time(test, base_cfg.max_test_samples)
        if train.empty or test.empty:
            continue
        print(f"split {split_id}: training big transformer (epochs<={big_cfg.epochs}, early stopping patience={big_cfg.patience})...", flush=True)
        pred_df, metrics = train_big_transformer_on_split(train, test, sequences, big_cfg)
        pred_df = pred_df.copy()
        pred_df["walk_split"] = split_id
        metrics = dict(metrics)
        metrics.update({
            "walk_split": split_id,
            "train_start": train_start,
            "train_end": train_end,
            "test_end": test_end,
        })
        results.append((pred_df, metrics))
        split_rows.append({
            "walk_split": split_id,
            "train_start": train_start,
            "train_end": train_end,
            "test_end": test_end,
            "train_samples": len(train),
            "test_samples": len(test),
        })

    if not results:
        raise RuntimeError("No walk-forward windows ran.")

    preds, metrics = combine_results(results)
    preds.to_parquet(out_dir / "walk_forward_predictions.parquet", engine="pyarrow", index=False)
    metrics.to_csv(out_dir / "walk_forward_metrics_by_split.csv", index=False)
    pd.DataFrame(split_rows).to_csv(out_dir / "walk_forward_splits.csv", index=False)

    summary = (
        metrics.groupby("model", as_index=False)
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
    )
    summary.to_csv(out_dir / "walk_forward_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved {out_dir}")


if __name__ == "__main__":
    main()
