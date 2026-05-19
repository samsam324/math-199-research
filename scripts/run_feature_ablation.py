"""
Drop one tabular feature at a time, retrain the booster on a saved dataset,
and report the delta in held-out metrics. Use to formally test whether a
suspected-leak feature like time_since_zero_crossing is actually carrying
real signal.

Single chronological train/test split is fine here: this is an ablation,
not a ranking estimator. For each variant we compare to a baseline trained
on all features with the same split and the same seed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_dataset import TABULAR_COLUMNS, load_dataset
from src.modeling import TrainingConfig, _balanced_tail, _load_xgb_classifier, _split_frame, trading_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-at-a-time feature ablation on the tabular booster.")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--max-train-samples", type=int, default=20000)
    p.add_argument("--max-test-samples", type=int, default=10000)
    p.add_argument("--features", nargs="+", default=TABULAR_COLUMNS, help="Features to ablate one-at-a-time.")
    return p.parse_args()


def _train_with_features(samples: pd.DataFrame, features: List[str], cfg: TrainingConfig) -> Dict[str, float]:
    train, test = _split_frame(samples)
    train = _balanced_tail(train, cfg.max_train_samples)
    test = _balanced_tail(test, cfg.max_test_samples)

    x_train = train[features].to_numpy(dtype=float)
    y_train = train["y_class"].to_numpy(dtype=int)
    x_test = test[features].to_numpy(dtype=float)
    y_test = test["y_class"].to_numpy(dtype=int)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model, _ = _load_xgb_classifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test).astype(int)

    from sklearn.metrics import accuracy_score, f1_score

    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    tm = trading_metrics(test, pred)

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "total_pnl": tm.get("total_pnl", 0.0),
        "win_rate": tm.get("win_rate", 0.0),
        "trades": tm.get("trades", 0),
    }


def main() -> None:
    args = parse_args()
    samples, _ = load_dataset(Path(args.dataset_dir))
    cfg = TrainingConfig(max_train_samples=args.max_train_samples, max_test_samples=args.max_test_samples)

    rows = []
    print("Training baseline (all features)...", flush=True)
    baseline = _train_with_features(samples, TABULAR_COLUMNS, cfg)
    rows.append({"dropped_feature": "(none, baseline)", **baseline})

    for feat in args.features:
        if feat not in TABULAR_COLUMNS:
            continue
        kept = [c for c in TABULAR_COLUMNS if c != feat]
        print(f"Training without {feat}...", flush=True)
        m = _train_with_features(samples, kept, cfg)
        m["dropped_feature"] = feat
        m["delta_accuracy"] = m["accuracy"] - baseline["accuracy"]
        m["delta_macro_f1"] = m["macro_f1"] - baseline["macro_f1"]
        m["delta_total_pnl"] = m["total_pnl"] - baseline["total_pnl"]
        m["delta_win_rate"] = m["win_rate"] - baseline["win_rate"]
        rows.append(m)

    out = pd.DataFrame(rows)
    cols = [
        "dropped_feature", "accuracy", "macro_f1", "total_pnl", "win_rate", "trades",
        "delta_accuracy", "delta_macro_f1", "delta_total_pnl", "delta_win_rate",
    ]
    out = out.reindex(columns=cols)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print()
    print(out.to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
