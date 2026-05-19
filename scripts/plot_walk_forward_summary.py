"""
Per-model summary bar chart with bootstrap confidence intervals from walk-
forward metrics.

Reads `walk_forward_metrics_by_split.csv` (one row per model x split),
bootstraps each model's per-split metric values, and plots a horizontal bar
chart of the mean with 5%/95% CIs.

Default metric is `pnl_mean_to_std` (the placeholder Sharpe-like score). You
can also plot accuracy, macro_f1, win_rate, total_pnl by --metric.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward metric bar chart with bootstrap CIs.")
    p.add_argument("--metrics-csv", required=True, help="walk_forward_metrics_by_split.csv")
    p.add_argument("--out-path", required=True)
    p.add_argument("--metric", default="pnl_mean_to_std")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def bootstrap_ci(values: np.ndarray, n: int, lo: float, hi: float, rng: np.random.Generator) -> tuple:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    samples = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.mean(values)), float(np.quantile(samples, lo)), float(np.quantile(samples, hi))


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metrics_csv)
    if args.metric not in df.columns:
        raise ValueError(f"metric '{args.metric}' not in columns: {sorted(df.columns)}")

    rng = np.random.default_rng(args.seed)
    rows = []
    for model, g in df.groupby("model"):
        mean, lo, hi = bootstrap_ci(g[args.metric].to_numpy(dtype=float), args.n_bootstrap, 0.05, 0.95, rng)
        rows.append({"model": model, "mean": mean, "ci_lo": lo, "ci_hi": hi, "n_splits": len(g)})
    summary = pd.DataFrame(rows).sort_values("mean")

    fig, ax = plt.subplots(figsize=(9, max(3, 0.55 * len(summary) + 1.5)))
    y = np.arange(len(summary))
    means = summary["mean"].to_numpy()
    err_lo = means - summary["ci_lo"].to_numpy()
    err_hi = summary["ci_hi"].to_numpy() - means
    ax.barh(y, means, color="#4c78a8", alpha=0.85)
    ax.errorbar(means, y, xerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=4, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["model"])
    ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlabel(f"{args.metric} (mean, 5/95% bootstrap CI over splits)")
    ax.set_title(f"Walk-forward {args.metric} by model")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(summary.to_string(index=False))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
