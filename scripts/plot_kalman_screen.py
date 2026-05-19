"""
Visualize the OOS Kalman cointegration screen across the full pair universe.

Reads `kalman_pair_screen.csv` from `run_kalman_pair_screen.py` and produces
a CDF of static vs Kalman OOS ADF p-values, log x-axis, with the p=0.05
reference line. This is the visual for the "generalization" half of
finding 1: that Kalman's ability to recover cointegration is not a
property of the top-10 selected pairs but holds across the entire liquid
universe.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CDF of static vs Kalman OOS ADF p-values across the screened universe.")
    p.add_argument("--csv", required=True, help="kalman_pair_screen.csv from run_kalman_pair_screen.py")
    p.add_argument("--out-path", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    static_p = df["static_adf_pvalue_oos"].dropna().to_numpy(dtype=float)
    kalman_p = df["kalman_adf_pvalue_oos"].dropna().to_numpy(dtype=float)

    def cdf(values):
        v = np.sort(values)
        y = np.arange(1, len(v) + 1) / len(v)
        return v, y

    s_x, s_y = cdf(static_p)
    k_x, k_y = cdf(kalman_p)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s_x, s_y, color="#888", linewidth=2, label=f"static OLS (n={len(static_p)})")
    ax.plot(k_x, k_y, color="#4c78a8", linewidth=2, label=f"Kalman MLE (n={len(kalman_p)})")
    ax.axvline(0.05, color="red", linestyle="--", linewidth=1, label="p = 0.05")
    ax.set_xscale("log")
    ax.set_xlim(1e-11, 1.0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("OOS ADF p-value (log scale)")
    ax.set_ylabel("CDF (fraction of pairs at or below p)")
    ax.set_title(
        "OOS cointegration: static OLS vs Kalman dynamic hedge\n"
        f"{len(df)} pairs across the liquidity top-50 universe, 90d train / 30d held-out test"
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, which="both")

    for thr in [0.001, 0.05]:
        static_pass = float((static_p < thr).mean()) * 100
        kalman_pass = float((kalman_p < thr).mean()) * 100
        ax.text(thr, 0.02 if thr == 0.05 else 0.06, f"  static {static_pass:.1f}%\n  Kalman {kalman_pass:.1f}%",
                fontsize=8, alpha=0.8)

    fig.tight_layout()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
