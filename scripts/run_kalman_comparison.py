"""
Static OLS hedge ratio vs. Kalman dynamic hedge ratio comparison.

For each selected pair, compute both spreads on the training window and report:
  - ADF p-values on both spreads (kalman with a burn-in drop)
  - Residual std on both
  - Time-varying beta path

Also writes per-pair overlay plots: static beta as a horizontal line, kalman
beta path, and both spreads on a shared axis.

Inputs:
  --pairs-path  parquet of selected pairs (from run_first_branch.py)
  --data-dir    local Binance store
  --t0          as-of timestamp
  --train-days  training window length
  --out-dir     where to write metrics and plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_store import StoreConfig, build_close_panel
from src.kalman_hedge import KalmanConfig, adf_comparison, kalman_spread_series


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare static OLS vs Kalman dynamic hedge ratio on selected pairs.")
    p.add_argument("--pairs-path", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--t0", default="2026-01-01T00:00:00Z")
    p.add_argument("--train-days", type=int, default=180)
    p.add_argument("--top-pairs", type=int, default=10)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--burn-in", type=int, default=72, help="Bars to drop from start of kalman spread before ADF.")
    p.add_argument("--q-alpha", type=float, default=1e-7)
    p.add_argument("--q-beta", type=float, default=1e-5)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_parquet(args.pairs_path, engine="pyarrow")
    if pairs.empty:
        raise RuntimeError(f"No pairs in {args.pairs_path}")
    pairs = pairs.head(args.top_pairs)

    cfg = StoreConfig(interval="1h", data_dir=args.data_dir)
    t0 = pd.Timestamp(args.t0)
    train_start = (t0 - pd.Timedelta(days=args.train_days)).floor(cfg.interval)
    train_end = t0.floor(cfg.interval)

    symbols = sorted(set(pairs["sym_a"]).union(pairs["sym_b"]))
    close_panel = build_close_panel(cfg, symbols, train_start, train_end, min_symbol_coverage=0.80)
    if close_panel.empty:
        raise RuntimeError("Empty close panel; widen training window or check data.")

    kcfg = KalmanConfig(q_alpha=args.q_alpha, q_beta=args.q_beta)

    rows: List[Dict[str, float]] = []
    for _, pair_row in pairs.iterrows():
        sym_a, sym_b = pair_row["sym_a"], pair_row["sym_b"]
        if sym_a not in close_panel.columns or sym_b not in close_panel.columns:
            continue
        df = kalman_spread_series(close_panel, sym_a, sym_b, cfg=kcfg)
        if df.empty:
            continue

        diag = adf_comparison(
            df["static_spread"].to_numpy(dtype=float),
            df["kalman_spread"].to_numpy(dtype=float),
            burn_in=args.burn_in,
        )
        diag.update(
            {
                "pair": f"{sym_a}_{sym_b}",
                "sym_a": sym_a,
                "sym_b": sym_b,
                "static_beta": float(df["static_beta"].iloc[0]),
                "kalman_beta_mean": float(df["kalman_beta"].iloc[args.burn_in:].mean()),
                "kalman_beta_std": float(df["kalman_beta"].iloc[args.burn_in:].std(ddof=1)),
                "n_obs": int(len(df)),
            }
        )
        rows.append(diag)

        # Save per-pair time series for downstream plotting.
        df.to_parquet(out_dir / f"kalman_{sym_a}_{sym_b}.parquet", engine="pyarrow", index=False)

        if not args.no_plots:
            _plot_pair(df, sym_a, sym_b, out_dir, burn_in=args.burn_in)

    if not rows:
        raise RuntimeError("No pairs produced Kalman comparisons.")

    summary = pd.DataFrame(rows)
    summary = summary[
        [
            "pair",
            "sym_a",
            "sym_b",
            "static_beta",
            "kalman_beta_mean",
            "kalman_beta_std",
            "static_std",
            "kalman_std",
            "static_adf_stat",
            "static_adf_pvalue",
            "kalman_adf_stat",
            "kalman_adf_pvalue",
            "n_obs",
        ]
    ].sort_values("kalman_adf_pvalue")
    summary.to_csv(out_dir / "kalman_comparison.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved Kalman comparison outputs under {out_dir}")


def _plot_pair(df: pd.DataFrame, sym_a: str, sym_b: str, out_dir: Path, burn_in: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(df["timestamp"], df["kalman_beta"], color="#d62728", linewidth=0.9, label="kalman beta")
    axes[0].axhline(df["static_beta"].iloc[0], color="#1f77b4", linestyle="--", linewidth=1, label="static beta")
    axes[0].set_ylabel("beta")
    axes[0].set_title(f"{sym_a} on {sym_b}: hedge ratio over time")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["timestamp"], df["static_spread"], color="#1f77b4", linewidth=0.7, label="static spread")
    axes[1].plot(df["timestamp"].iloc[burn_in:], df["kalman_spread"].iloc[burn_in:], color="#d62728", linewidth=0.7, label="kalman spread")
    axes[1].axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    axes[1].set_ylabel("log-price residual")
    axes[1].set_xlabel("time (UTC)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / f"kalman_{sym_a}_{sym_b}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
