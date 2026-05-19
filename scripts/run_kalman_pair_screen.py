"""
Screen all pairs in a liquidity-filtered universe by OOS Kalman cointegration.

Static OLS cointegration screening (src/pair_selection.py score_pairs) finds
zero passing pairs on the local Binance.US slice. Finding 1 in
docs/RESULTS.md establishes that Kalman dynamic hedge ratios DO recover
cointegration on the same data. This script makes that methodologically
consistent: for every C(N, 2) pair in the liquidity-filtered universe,
fit a Kalman filter by MLE on a training slice, forward-roll on a held-out
test slice, ADF test the OOS residuals, and report how many pairs pass at
various p-value thresholds.

Output:
  kalman_pair_screen.csv         all pairs with diagnostics, sorted by OOS p
  kalman_selected_pairs.parquet  top N pairs that pass --max-pvalue, in the
                                 same schema as src/pair_selection.py output
                                 so the rest of the pipeline can use them
                                 unchanged
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

from src.data_store import StoreConfig, build_close_panel, list_local_symbols
from src.kalman_hedge import fit_kalman_mle, kalman_forward_residuals
from src.universe import compute_universe_at_time, filter_top_n_by_liquidity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOS Kalman cointegration screen on every pair in the liquidity universe.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--t0", default="2024-01-01T00:00:00Z")
    p.add_argument("--train-days", type=int, default=90)
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--liquid-top-n", type=int, default=50, help="Liquidity universe size to screen across.")
    p.add_argument("--min-history-days", type=float, default=180.0)
    p.add_argument("--max-pvalue", type=float, default=0.05, help="OOS ADF p-value threshold for selected pairs.")
    p.add_argument("--top-pairs", type=int, default=20, help="Max pairs to write to kalman_selected_pairs.parquet.")
    p.add_argument("--min-corr-prescreen", type=float, default=0.0,
                   help="Optional: skip pairs whose return correlation is below this on the training window. "
                        "Default 0 means screen every pair.")
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_store = StoreConfig(interval="1h", data_dir=args.data_dir)
    t0 = pd.Timestamp(args.t0)
    total_days = args.train_days + args.test_days
    full_start = (t0 - pd.Timedelta(days=total_days)).floor(cfg_store.interval)
    train_end = (t0 - pd.Timedelta(days=args.test_days)).floor(cfg_store.interval)
    full_end = t0.floor(cfg_store.interval)

    local = list_local_symbols(cfg_store)
    universe = compute_universe_at_time(cfg_store, local, t0, min_history_days=args.min_history_days)
    print(f"Universe at t0={t0} with min_history={args.min_history_days}d: {len(universe)} symbols", flush=True)
    liquid = filter_top_n_by_liquidity(cfg_store, universe, full_start, train_end, top_n=args.liquid_top_n, min_coverage=0.80)
    print(f"Liquidity top-{args.liquid_top_n}: {len(liquid)} symbols", flush=True)

    panel = build_close_panel(cfg_store, liquid, full_start, full_end, min_symbol_coverage=0.50)
    if panel.empty:
        raise RuntimeError("Empty close panel.")
    logp = np.log(panel)
    train_logp = logp[logp.index < train_end]
    test_logp = logp[(logp.index >= train_end) & (logp.index < full_end)]
    print(f"Train window: {len(train_logp)} bars; Test window: {len(test_logp)} bars", flush=True)

    # Optional correlation prescreen to skip obvious noise pairs
    if args.min_corr_prescreen > 0:
        rets = train_logp.diff().dropna()
        corr = rets.corr()
    else:
        corr = None

    cols = list(panel.columns)
    pair_list = list(combinations(cols, 2))
    print(f"Screening {len(pair_list)} candidate pairs...", flush=True)

    rows: List[Dict[str, float]] = []
    for sym_a, sym_b in tqdm(pair_list, desc="Kalman screen"):
        if corr is not None:
            c = float(corr.loc[sym_a, sym_b])
            if not np.isfinite(c) or c < args.min_corr_prescreen:
                continue
        else:
            # cheap correlation compute for diagnostic regardless
            try:
                c = float(np.corrcoef(train_logp[sym_a].diff().dropna(), train_logp[sym_b].diff().dropna())[0, 1])
            except Exception:
                c = float("nan")

        y_tr = train_logp[sym_a].to_numpy(dtype=float)
        x_tr = train_logp[sym_b].to_numpy(dtype=float)
        y_te = test_logp[sym_a].to_numpy(dtype=float)
        x_te = test_logp[sym_b].to_numpy(dtype=float)
        if np.isnan(y_tr).any() or np.isnan(x_tr).any() or np.isnan(y_te).any() or np.isnan(x_te).any():
            continue
        if len(y_tr) < 200 or len(y_te) < 100:
            continue

        # Static OLS reference: fit on train, evaluate on test
        try:
            ols = OLS(y_tr, add_constant(x_tr)).fit()
            alpha_s = float(ols.params[0])
            beta_s = float(ols.params[1])
        except Exception:
            continue
        static_oos = y_te - (alpha_s + beta_s * x_te)
        stat_s, p_s = _adf(static_oos)

        # Kalman MLE on train, forward-roll on test
        try:
            fitted = fit_kalman_mle(y_tr, x_tr)
        except Exception:
            continue
        if not fitted.get("converged", False):
            continue
        _, _, kalman_oos = kalman_forward_residuals(y_te, x_te, fitted)
        stat_k, p_k = _adf(kalman_oos)

        rows.append({
            "pair": f"{sym_a}_{sym_b}",
            "sym_a": sym_a,
            "sym_b": sym_b,
            "corr": c,
            "static_beta": beta_s,
            "kalman_beta_train_final": fitted["beta_train_final"],
            "q_alpha": fitted["q_alpha"],
            "q_beta": fitted["q_beta"],
            "r": fitted["r"],
            "static_oos_std": float(np.std(static_oos, ddof=1)),
            "kalman_oos_std": float(np.std(kalman_oos, ddof=1)) if len(kalman_oos) > 1 else float("nan"),
            "static_adf_stat_oos": stat_s,
            "static_adf_pvalue_oos": p_s,
            "kalman_adf_stat_oos": stat_k,
            "kalman_adf_pvalue_oos": p_k,
        })

    if not rows:
        raise RuntimeError("Screen produced no valid pairs.")

    out = pd.DataFrame(rows).sort_values("kalman_adf_pvalue_oos")
    out.to_csv(out_dir / "kalman_pair_screen.csv", index=False)

    # Distribution summary
    n = len(out)
    print(f"\nPairs screened: {n}", flush=True)
    for p_thr in (0.001, 0.005, 0.01, 0.05, 0.10):
        n_static = int((out["static_adf_pvalue_oos"] < p_thr).sum())
        n_kalman = int((out["kalman_adf_pvalue_oos"] < p_thr).sum())
        print(f"  p < {p_thr:.3f}: static={n_static:5d} ({100*n_static/n:.1f}%)  kalman={n_kalman:5d} ({100*n_kalman/n:.1f}%)", flush=True)

    # Selected pairs in the standard schema downstream tooling expects
    selected = out[out["kalman_adf_pvalue_oos"] < args.max_pvalue].head(args.top_pairs).copy()
    print(f"\nSelected top {len(selected)} pairs (kalman OOS p < {args.max_pvalue}):", flush=True)
    print(selected[["pair", "corr", "static_adf_pvalue_oos", "kalman_adf_pvalue_oos", "kalman_oos_std"]].to_string(index=False), flush=True)

    # Write a parquet that matches src/pair_selection.py output schema so the
    # existing pipeline (run_first_branch.py) can consume it directly.
    pipeline_rows = []
    for _, r in selected.iterrows():
        pipeline_rows.append({
            "pair": r["pair"],
            "sym_a": r["sym_a"],
            "sym_b": r["sym_b"],
            "corr": r["corr"],
            "alpha": 0.0,  # not directly comparable; selection_method will note this
            "beta_a_on_b": r["static_beta"],  # used by backtester for contract sizing
            "beta_instability": float("nan"),
            "spread_vol": r["kalman_oos_std"],
            "adf_stat": r["kalman_adf_stat_oos"],
            "adf_pvalue": r["kalman_adf_pvalue_oos"],
            "half_life_hours": float("nan"),
            "score": -float(r["kalman_adf_pvalue_oos"]),  # higher = better
            "selection_method": "kalman_oos_cointegration",
        })
    pipeline = pd.DataFrame(pipeline_rows)
    pipeline.to_parquet(out_dir / "kalman_selected_pairs.parquet", engine="pyarrow", index=False)
    print(f"\nSaved {out_dir/'kalman_pair_screen.csv'} and {out_dir/'kalman_selected_pairs.parquet'}", flush=True)


if __name__ == "__main__":
    main()
