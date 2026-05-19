"""
Run the cost-aware portfolio backtester on saved walk-forward predictions.

Outputs a per-model metrics table comparable in shape to what minitron
reports for equity strategies (sharpe, total_return, max_drawdown,
n_trades, turnover, win_rate, bars).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import BacktestConfig, metrics_for_minitron, run_backtest
from src.data_store import StoreConfig, build_close_panel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cost-aware portfolio backtest on walk-forward predictions.")
    p.add_argument("--predictions-path", required=True, help="walk_forward_predictions.parquet or predictions.parquet")
    p.add_argument("--pairs-path", required=True, help="selected_pairs.parquet from run_first_branch")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--taker-fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--leg-notional", type=float, default=10_000.0)
    p.add_argument("--max-active-pairs", type=int, default=None)
    p.add_argument("--entry-z", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(args.predictions_path, engine="pyarrow")
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    pairs = pd.read_parquet(args.pairs_path, engine="pyarrow")

    if "pair" not in pairs.columns:
        pairs["pair"] = pairs["sym_a"] + "_" + pairs["sym_b"]

    needed_symbols = sorted(set(pairs["sym_a"]).union(pairs["sym_b"]))
    cfg_store = StoreConfig(interval="1h", data_dir=args.data_dir)
    start = predictions["timestamp"].min().floor("D")
    end = predictions["timestamp"].max().ceil("D") + pd.Timedelta(days=1)
    close_panel = build_close_panel(cfg_store, needed_symbols, start, end, min_symbol_coverage=0.50)
    if close_panel.empty:
        raise RuntimeError("Empty close panel; check data store coverage for the relevant timestamps.")

    cfg = BacktestConfig(
        taker_fee_bps=args.taker_fee_bps,
        slippage_bps=args.slippage_bps,
        leg_notional=args.leg_notional,
        max_active_pairs=args.max_active_pairs,
        entry_z_threshold=args.entry_z,
    )

    rows: List[Dict[str, float]] = []
    for model_name in predictions["model"].unique():
        model_preds = predictions[predictions["model"] == model_name].copy()
        try:
            result = run_backtest(model_preds, close_panel, pairs, cfg=cfg)
        except RuntimeError as exc:
            print(f"  skipping {model_name}: {exc}")
            continue
        flat = {"model": model_name, **metrics_for_minitron(result)}
        rows.append(flat)

        result.portfolio_returns.to_csv(out_dir / f"portfolio_returns_{model_name}.csv", header=["pnl"])
        result.positions.to_csv(out_dir / f"positions_{model_name}.csv")
        if not result.trades.empty:
            result.trades.to_csv(out_dir / f"trades_{model_name}.csv", index=False)

    if not rows:
        raise RuntimeError("No models produced backtest results.")

    summary = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    summary.to_csv(out_dir / "portfolio_metrics.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved portfolio backtest outputs under {out_dir}")


if __name__ == "__main__":
    main()
