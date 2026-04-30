"""
Plot spread series for top-scoring cointegrated pairs.

Usage:
    python scripts/plot_spreads.py [--top N] [--days D] [--out DIR]

Produces one PNG per pair showing:
  - Log-price spread over the training window
  - Mean and ±1, ±2 standard deviation bands
  - Half-life and ADF p-value annotation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from src.data_store import StoreConfig, list_local_symbols, build_close_panel
from src.universe import compute_universe_at_time
from src.pair_selection import PairConfig, score_pairs


def compute_spread(close_panel: pd.DataFrame, sym_a: str, sym_b: str) -> tuple[pd.Series, float, float]:
    logp = np.log(close_panel)
    y = logp[sym_a].values
    x = logp[sym_b].values
    X = add_constant(x)
    model = OLS(y, X).fit()
    alpha = float(model.params[0])
    beta = float(model.params[1])
    spread = pd.Series(y - (alpha + beta * x), index=close_panel.index, name=f"{sym_a}-{sym_b}")
    return spread, alpha, beta


def plot_pair(
    spread: pd.Series,
    sym_a: str,
    sym_b: str,
    beta: float,
    half_life: float,
    adf_pvalue: float,
    out_path: Path,
) -> None:
    mu = spread.mean()
    sd = spread.std(ddof=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(spread.index, spread.values, color="#1f77b4", linewidth=0.8, label="spread")
    ax.axhline(mu, color="black", linestyle="-", linewidth=1, alpha=0.6, label=f"mean={mu:.4f}")
    for k, color in [(1, "#888"), (2, "#aaa")]:
        ax.axhline(mu + k * sd, color=color, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(mu - k * sd, color=color, linestyle="--", linewidth=0.8, alpha=0.7)

    title = (
        f"{sym_a} - {beta:.3f}*{sym_b}  (log-price spread)\n"
        f"half-life: {half_life:.1f}h   ADF p-value: {adf_pvalue:.4f}   spread std: {sd:.4f}"
    )
    ax.set_title(title)
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("log-price residual")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=4, help="number of top pairs to plot")
    parser.add_argument("--days", type=int, default=180, help="training window days")
    parser.add_argument("--t0", type=str, default="2026-01-01T00:00:00Z", help="as-of timestamp")
    parser.add_argument("--out", type=str, default="figures", help="output directory")
    args = parser.parse_args()

    cfg = StoreConfig(interval="1h", data_dir="data")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_symbols = list_local_symbols(cfg)
    if not local_symbols:
        raise RuntimeError("No local data found. Run: python store_data.py")

    t0 = pd.Timestamp(args.t0)
    universe = compute_universe_at_time(cfg, local_symbols, t0)
    train_start = (t0 - pd.Timedelta(days=args.days)).floor(cfg.interval)
    train_end = t0.floor(cfg.interval)

    print(f"Universe at t0={t0}: {len(universe)} symbols")
    print(f"Training window: {train_start} -> {train_end}")

    panel = build_close_panel(cfg, universe, train_start, train_end, min_symbol_coverage=0.80)
    print(f"Panel shape: {panel.shape}")

    pcfg = PairConfig(
        min_corr=0.5,
        max_pairs_to_test=9999999999999999999,
        max_adf_pvalue=0.05,
        half_life_min_hours=4.0,
        half_life_max_hours=240.0,
        half_life_target_hours=48.0,
        beta_stability_segments=4,
        max_beta_instability=0.35,
        min_spread_vol=0.002,
        min_obs=1200,
    )

    print("Scoring pairs...")
    pairs = score_pairs(panel, pcfg)
    if pairs.empty:
        print("No pairs passed filters.")
        return

    top = pairs.head(args.top)
    print(f"\nPlotting top {len(top)} pairs to {out_dir}/")
    print(top[["pair", "corr", "adf_pvalue", "half_life_hours", "spread_vol", "score"]].to_string(index=False))

    for _, row in top.iterrows():
        sym_a, sym_b = row["sym_a"], row["sym_b"]
        spread, alpha, beta = compute_spread(panel, sym_a, sym_b)
        out_path = out_dir / f"spread_{sym_a}_{sym_b}.png"
        plot_pair(
            spread=spread,
            sym_a=sym_a,
            sym_b=sym_b,
            beta=beta,
            half_life=float(row["half_life_hours"]),
            adf_pvalue=float(row["adf_pvalue"]),
            out_path=out_path,
        )
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
