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
    p.add_argument(
        "--block-size",
        type=int,
        default=3,
        help="Block-bootstrap size. Splits are time-ordered and consecutive "
             "windows share training data (90d train / 30d step means ~67% overlap); "
             "iid resampling overstates precision. Block size 3 matches the "
             "approximate dependence horizon at 30-day step. Set to 1 for the "
             "old iid bootstrap.",
    )
    return p.parse_args()


def block_bootstrap_ci(values: np.ndarray, n: int, lo: float, hi: float, block_size: int, rng: np.random.Generator) -> tuple:
    """
    Circular block bootstrap: resample contiguous blocks of length block_size
    (with wrap-around) until the synthetic series matches the original length.
    Reduces precision-inflation when consecutive values are autocorrelated
    (e.g., overlapping walk-forward splits).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    m = len(values)
    if m == 0:
        return float("nan"), float("nan"), float("nan")
    if m == 1:
        v = float(values[0])
        return v, v, v
    b = max(1, min(block_size, m))
    n_blocks = (m + b - 1) // b
    samples = np.empty(n, dtype=float)
    for i in range(n):
        starts = rng.integers(0, m, size=n_blocks)
        # circular indexing
        idx = (starts[:, None] + np.arange(b)[None, :]) % m
        flat = idx.reshape(-1)[:m]
        samples[i] = float(values[flat].mean())
    return float(np.mean(values)), float(np.quantile(samples, lo)), float(np.quantile(samples, hi))


def bootstrap_ci(values: np.ndarray, n: int, lo: float, hi: float, rng: np.random.Generator) -> tuple:
    """Backwards-compatible iid bootstrap wrapper (block_size=1)."""
    return block_bootstrap_ci(values, n, lo, hi, 1, rng)


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
        g_sorted = g.sort_values("walk_split") if "walk_split" in g.columns else g
        mean, lo, hi = block_bootstrap_ci(
            g_sorted[args.metric].to_numpy(dtype=float),
            args.n_bootstrap, 0.05, 0.95, args.block_size, rng,
        )
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
