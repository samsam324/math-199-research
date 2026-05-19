"""
Cumulative dollar PnL by model over the walk-forward test horizon, read from
the per-model portfolio_returns_*.csv files written by run_portfolio_backtest.

Plots one line per model on a shared time axis. The cost wall shows up as
the gap between models that look identical at bar t=0 and diverge into deep
negative territory by t=N as the bar-by-bar cost drag accumulates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cumulative PnL curve per model from backtest outputs.")
    p.add_argument("--backtest-dir", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--title", default="Cumulative portfolio PnL by model (with 15 bps round-trip cost)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    backtest_dir = Path(args.backtest_dir)
    paths = sorted(backtest_dir.glob("portfolio_returns_*.csv"))
    if not paths:
        raise RuntimeError(f"No portfolio_returns_*.csv files in {backtest_dir}")

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.get_cmap("tab10").colors
    for i, path in enumerate(paths):
        model = path.stem.replace("portfolio_returns_", "")
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        equity = df["pnl"].cumsum()
        ax.plot(df["timestamp"], equity, label=model, color=colors[i % len(colors)], linewidth=1.2)

    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("cumulative PnL ($)")
    ax.set_title(args.title)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
