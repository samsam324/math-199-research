"""
Side-by-side per-pair ADF p-values: static OLS vs Kalman dynamic, evaluated
out-of-sample on a held-out test slice. This is the visual for the central
finding (Kalman recovers cointegration that static OLS misses on the same
training data).

Input: kalman_oos_comparison.csv from scripts/run_kalman_oos.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static vs Kalman OOS ADF p-values per pair.")
    p.add_argument("--csv", required=True, help="kalman_oos_comparison.csv")
    p.add_argument("--out-path", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv).sort_values("kalman_adf_p_oos_honest")

    pairs = df["pair"].tolist()
    static_p = df["static_adf_p_oos"].to_numpy(dtype=float)
    kalman_p = df["kalman_adf_p_oos_honest"].to_numpy(dtype=float)

    y = np.arange(len(pairs))
    height = 0.4

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(pairs) + 1.5)))
    ax.barh(y - height / 2, static_p, height=height, label="static OLS", color="#888")
    ax.barh(y + height / 2, kalman_p, height=height, label="Kalman MLE", color="#4c78a8")
    ax.set_xscale("log")
    ax.set_xlim(1e-10, 1.0)
    ax.axvline(0.05, color="red", linestyle="--", linewidth=1, label="p = 0.05")

    ax.set_yticks(y)
    ax.set_yticklabels(pairs)
    ax.set_xlabel("ADF p-value on test-slice residuals (log scale, lower = more stationary)")
    ax.set_title("Out-of-sample cointegration: static OLS vs Kalman dynamic hedge\nparameters fit on 90d training, applied to 30d test")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3, which="both")
    fig.tight_layout()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
