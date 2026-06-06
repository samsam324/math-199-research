"""
Phase 2 Kalman OOS cointegration screen on microprice spreads.

For every pair in the liquidity-filtered universe, fit Kalman MLE on a
training window of microprice log-series sampled at the chosen bar cadence
(default 1s) and forward-roll on a held-out test window. ADF on the OOS
residuals, report pass rates at standard p-value thresholds.

Replaces phase 1's hourly OHLCV close panel with bars-derived microprice
panel. Same protocol, same audit invariants.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.l2_store import L2Config, load_symbol_range as load_l2_range
from src.trade_store import TradeConfig
from src.bars import BarConfig, build_bars
from src.kalman_hedge import fit_kalman_mle, kalman_forward_residuals
from src.universe import compute_universe_at_time, filter_top_n_by_liquidity
from src.trade_store import load_symbol_range as load_trades_range


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOS Kalman cointegration screen on microprice spreads.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--t0", required=True, help="As-of UTC timestamp.")
    p.add_argument("--train-days", type=float, default=5.0)
    p.add_argument("--test-days", type=float, default=1.0)
    p.add_argument("--bar-size", default="1s")
    p.add_argument("--liquid-top-n", type=int, default=30)
    p.add_argument("--min-history-days", type=float, default=7.0)
    p.add_argument("--max-pvalue", type=float, default=0.05)
    p.add_argument("--top-pairs", type=int, default=20)
    p.add_argument("--min-corr-prescreen", type=float, default=0.0)
    p.add_argument("--levels", type=int, default=10)
    return p.parse_args()


def _adf(series: np.ndarray) -> tuple:
    if len(series) < 20 or not np.isfinite(series).all():
        return float("nan"), float("nan")
    try:
        stat, pvalue, *_ = adfuller(series, regression="c", autolag=None, maxlag=24)
        return float(stat), float(pvalue)
    except Exception:
        return float("nan"), float("nan")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cfg_l2 = L2Config(levels=args.levels, data_dir=str(Path(args.data_dir) / "l2"))
    cfg_trades = TradeConfig(data_dir=str(Path(args.data_dir) / "trades"))
    cfg_bars = BarConfig(bar_size=args.bar_size)

    t0 = pd.Timestamp(args.t0)
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    full_start = (t0 - pd.Timedelta(days=args.train_days + args.test_days)).floor("D")
    train_end = (t0 - pd.Timedelta(days=args.test_days)).floor("D")
    full_end = t0.floor("D")

    universe = compute_universe_at_time(
        t0=t0, l2_cfg=cfg_l2, trade_cfg=cfg_trades,
        min_history_days=args.min_history_days, require_both_feeds=True,
    )
    print(f"Universe at t0={t0}: {len(universe)} symbols", flush=True)
    liquid = filter_top_n_by_liquidity(universe, full_start, train_end, top_n=args.liquid_top_n, trade_cfg=cfg_trades)
    print(f"Liquidity top-{args.liquid_top_n}: {len(liquid)} symbols", flush=True)

    # Pre-build microprice panel at bar cadence over the full window
    micro: Dict[str, pd.Series] = {}
    for sym in tqdm(liquid, desc="Bars per symbol"):
        book = load_l2_range(sym, full_start, full_end, cfg=cfg_l2)
        trades = load_trades_range(sym, full_start, full_end, cfg=cfg_trades)
        if book.empty:
            continue
        bars = build_bars(book, trades, cfg=cfg_bars)
        if bars.empty:
            continue
        micro[sym] = bars["microprice"].astype(float)

    if len(micro) < 2:
        raise RuntimeError("Fewer than 2 symbols produced bars; cannot screen pairs.")

    panel = pd.concat(micro, axis=1).sort_index()
    panel = panel.dropna(how="any")
    train_panel = panel[panel.index < train_end]
    test_panel = panel[(panel.index >= train_end) & (panel.index < full_end)]
    print(f"Train bars: {len(train_panel)}, Test bars: {len(test_panel)}", flush=True)

    logp_train = np.log(train_panel)
    logp_test = np.log(test_panel)

    cols = list(panel.columns)
    pair_list = list(combinations(cols, 2))
    print(f"Screening {len(pair_list)} pairs...", flush=True)

    rows: List[Dict[str, float]] = []
    for sym_a, sym_b in tqdm(pair_list, desc="Kalman screen"):
        y_tr = logp_train[sym_a].to_numpy(dtype=float); x_tr = logp_train[sym_b].to_numpy(dtype=float)
        y_te = logp_test[sym_a].to_numpy(dtype=float); x_te = logp_test[sym_b].to_numpy(dtype=float)
        if np.isnan(y_tr).any() or np.isnan(x_tr).any() or len(y_tr) < 200 or len(y_te) < 100:
            continue
        try:
            c = float(np.corrcoef(np.diff(y_tr), np.diff(x_tr))[0, 1])
        except Exception:
            c = float("nan")
        if args.min_corr_prescreen > 0 and (not np.isfinite(c) or c < args.min_corr_prescreen):
            continue

        try:
            ols = OLS(y_tr, add_constant(x_tr)).fit()
            alpha_s = float(ols.params[0]); beta_s = float(ols.params[1])
        except Exception:
            continue
        static_oos = y_te - (alpha_s + beta_s * x_te)
        stat_s, p_s = _adf(static_oos)

        try:
            fitted = fit_kalman_mle(y_tr, x_tr)
        except Exception:
            continue
        if not fitted.get("converged", False):
            continue
        _, _, kalman_oos = kalman_forward_residuals(y_te, x_te, fitted)
        stat_k, p_k = _adf(kalman_oos)

        rows.append({
            "pair": f"{sym_a}_{sym_b}", "sym_a": sym_a, "sym_b": sym_b,
            "corr": c, "static_beta": beta_s, "kalman_beta_train_final": fitted["beta_train_final"],
            "q_alpha": fitted["q_alpha"], "q_beta": fitted["q_beta"], "r": fitted["r"],
            "static_oos_std": float(np.std(static_oos, ddof=1)),
            "kalman_oos_std": float(np.std(kalman_oos, ddof=1)) if len(kalman_oos) > 1 else float("nan"),
            "static_adf_stat_oos": stat_s, "static_adf_pvalue_oos": p_s,
            "kalman_adf_stat_oos": stat_k, "kalman_adf_pvalue_oos": p_k,
        })

    if not rows:
        raise RuntimeError("Screen produced no valid pairs.")

    out = pd.DataFrame(rows).sort_values("kalman_adf_pvalue_oos")
    out.to_csv(out_dir / "kalman_pair_screen.csv", index=False)

    n = len(out)
    print(f"\nPairs screened: {n}", flush=True)
    for p_thr in (0.001, 0.005, 0.01, 0.05, 0.10):
        n_static = int((out["static_adf_pvalue_oos"] < p_thr).sum())
        n_kalman = int((out["kalman_adf_pvalue_oos"] < p_thr).sum())
        print(f"  p < {p_thr:.3f}: static={n_static:5d} ({100*n_static/n:.1f}%)  kalman={n_kalman:5d} ({100*n_kalman/n:.1f}%)", flush=True)

    selected = out[out["kalman_adf_pvalue_oos"] < args.max_pvalue].head(args.top_pairs).copy()
    pipeline_rows = []
    for _, r in selected.iterrows():
        pipeline_rows.append({
            "pair": r["pair"], "sym_a": r["sym_a"], "sym_b": r["sym_b"],
            "corr": r["corr"], "alpha": 0.0, "beta_a_on_b": r["static_beta"],
            "beta_instability": float("nan"), "spread_vol": r["kalman_oos_std"],
            "adf_stat": r["kalman_adf_stat_oos"], "adf_pvalue": r["kalman_adf_pvalue_oos"],
            "half_life_bars": float("nan"), "score": -float(r["kalman_adf_pvalue_oos"]),
            "selection_method": "kalman_oos_cointegration_microprice",
        })
    pd.DataFrame(pipeline_rows).to_parquet(out_dir / "kalman_selected_pairs.parquet", engine="pyarrow", index=False)
    print(f"\nSaved {out_dir/'kalman_pair_screen.csv'} and kalman_selected_pairs.parquet", flush=True)


if __name__ == "__main__":
    main()
