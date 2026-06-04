from __future__ import annotations

import warnings
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_dataset import TABULAR_COLUMNS, load_dataset
from src.modeling import (
    TrainingConfig,
    _load_xgb_classifier,
    _metrics_for_predictions,
    _prediction_frame,
    combine_results,
    train_lstm_on_split,
    train_transformer_on_split,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run rolling walk-forward evaluation on a saved dataset.")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-days", type=int, default=90)
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--step-days", type=int, default=30)
    p.add_argument("--max-train-samples", type=int, default=30000)
    p.add_argument("--max-test-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--deep",
        action="store_true",
        help="Also train LSTM and transformer on each walk-forward split. Loads sequences.npz and is slow.",
    )
    p.add_argument("--dl-epochs", type=int, default=4, help="Epochs per deep model per split when --deep is set.")
    p.add_argument(
        "--with-micro",
        action="store_true",
        help="Include volume-as-information features (micro_* columns) in the XGBoost tabular model. "
        "Missing values are median-imputed per training window. No effect if the dataset has no micro_* columns.",
    )
    return p.parse_args()


def _micro_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("micro_")]


def _tail_by_time(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sort_values("timestamp").tail(max_rows)


def _windows(samples: pd.DataFrame, train_days: int, test_days: int, step_days: int) -> Iterable[Tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
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


def _train_tabular(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: TrainingConfig,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    cols = feature_cols or TABULAR_COLUMNS
    x_train = train[cols].to_numpy(dtype=float)
    y_train = train["y_class"].to_numpy(dtype=int)
    x_test = test[cols].to_numpy(dtype=float)

    # Median-impute (fit on train only -> no leakage). Columns that are entirely
    # NaN in the training window (e.g. micro features in a pre-L2 split) get 0,
    # which StandardScaler maps to a constant 0 column harmlessly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN micro col -> handled below
        medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, medians)
    x_test = np.where(np.isfinite(x_test), x_test, medians)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model, model_name = _load_xgb_classifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test).astype(int)
    return _metrics_for_predictions(model_name, train, test, pred)


def _baselines(train: pd.DataFrame, test: pd.DataFrame, cfg: TrainingConfig) -> List[Tuple[pd.DataFrame, Dict[str, float]]]:
    out: List[Tuple[pd.DataFrame, Dict[str, float]]] = []

    persist = np.ones(len(test), dtype=int)
    out.append(_metrics_for_predictions("persist_class", train.iloc[0:0], test, persist))

    majority = int(train["y_class"].mode().iloc[0]) if not train.empty else 1
    out.append(_metrics_for_predictions("majority_class", train, test, np.full(len(test), majority, dtype=int)))

    rng = np.random.default_rng(cfg.seed)
    probs = train["y_class"].value_counts(normalize=True).reindex([0, 1, 2], fill_value=0.0).to_numpy(dtype=float)
    probs = probs / probs.sum() if probs.sum() > 0 else np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    out.append(_metrics_for_predictions("random_stratified", train, test, rng.choice([0, 1, 2], size=len(test), p=probs)))

    z = test["latest_spread_z"].to_numpy(dtype=float)
    zpred = np.ones(len(test), dtype=int)
    zpred[np.abs(z) >= 1.5] = 0
    out.append(_metrics_for_predictions("zscore_rule", train.iloc[0:0], test, zpred))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, sequences = load_dataset(Path(args.dataset_dir))
    samples = samples.sort_values("timestamp").copy()
    samples["timestamp"] = pd.to_datetime(samples["timestamp"], utc=True)

    tabular_cols = list(TABULAR_COLUMNS)
    if args.with_micro:
        micro = _micro_columns(samples)
        if micro:
            tabular_cols += micro
            print(f"Volume-information features ON: +{len(micro)} micro columns -> {len(tabular_cols)} tabular features")
        else:
            print("--with-micro set but dataset has no micro_* columns; running base features only.")
    cfg = TrainingConfig(
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        dl_epochs=args.dl_epochs,
        seed=args.seed,
    )

    if args.deep:
        print(f"Deep models enabled. Sequences shape: {sequences.shape}, epochs/split: {cfg.dl_epochs}")

    results: List[Tuple[pd.DataFrame, Dict[str, float]]] = []
    split_rows: List[Dict[str, object]] = []
    for split_id, train_start, train_end, test_end in _windows(samples, args.train_days, args.test_days, args.step_days):
        train = samples[(samples["timestamp"] >= train_start) & (samples["timestamp"] < train_end)]
        test = samples[(samples["timestamp"] >= train_end) & (samples["timestamp"] < test_end)]
        train = _tail_by_time(train, cfg.max_train_samples)
        test = _tail_by_time(test, cfg.max_test_samples)
        if train.empty or test.empty:
            continue

        split_results = _baselines(train, test, cfg)
        split_results.append(_train_tabular(train, test, cfg, feature_cols=tabular_cols))

        if args.deep:
            print(f"  split {split_id}: training LSTM...")
            split_results.append(train_lstm_on_split(train, test, sequences, cfg))
            print(f"  split {split_id}: training transformer...")
            split_results.append(train_transformer_on_split(train, test, sequences, cfg))

        for pred_df, metrics in split_results:
            pred_df = pred_df.copy()
            pred_df["walk_split"] = split_id
            metrics = dict(metrics)
            metrics.update(
                {
                    "walk_split": split_id,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_end": test_end,
                }
            )
            results.append((pred_df, metrics))

        split_rows.append(
            {
                "walk_split": split_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_end": test_end,
                "train_samples": len(train),
                "test_samples": len(test),
            }
        )

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
            mean_pnl_mean=("mean_pnl", "mean"),
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
