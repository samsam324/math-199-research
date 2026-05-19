"""
Train the tabular booster on the saved dataset, then plot feature importances.

Uses gain importance for XGBoost when available; falls back to permutation
importance for HistGradientBoosting.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_dataset import TABULAR_COLUMNS, load_dataset
from src.modeling import TrainingConfig, _load_xgb_classifier, _split_frame, _balanced_tail


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature importance plot for the tabular booster.")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--max-train-samples", type=int, default=20000)
    p.add_argument("--max-test-samples", type=int, default=10000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples, _ = load_dataset(Path(args.dataset_dir))
    cfg = TrainingConfig(max_train_samples=args.max_train_samples, max_test_samples=args.max_test_samples)

    train, test = _split_frame(samples)
    train = _balanced_tail(train, cfg.max_train_samples)
    test = _balanced_tail(test, cfg.max_test_samples)

    x_train = train[TABULAR_COLUMNS].to_numpy(dtype=float)
    y_train = train["y_class"].to_numpy(dtype=int)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    model, model_name = _load_xgb_classifier()
    model.fit(x_train, y_train)

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        kind = "feature_importances_"
    else:
        from sklearn.inspection import permutation_importance

        x_test = scaler.transform(test[TABULAR_COLUMNS].to_numpy(dtype=float))
        y_test = test["y_class"].to_numpy(dtype=int)
        result = permutation_importance(model, x_test, y_test, n_repeats=8, random_state=cfg.seed, n_jobs=-1)
        importances = result.importances_mean
        kind = "permutation_importance"

    order = np.argsort(importances)
    labels = np.array(TABULAR_COLUMNS)[order]
    values = importances[order]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(TABULAR_COLUMNS) + 1)))
    ax.barh(np.arange(len(labels)), values, color="#4c78a8")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"{kind} ({model_name})")
    ax.set_title("Tabular booster feature importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(pd.DataFrame({"feature": labels[::-1], "importance": values[::-1]}).to_string(index=False))


if __name__ == "__main__":
    main()
