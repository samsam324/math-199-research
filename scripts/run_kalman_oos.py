"""
Honest out-of-sample evaluation of the Kalman dynamic hedge.

For each pair:
  1. Split the price series into a training slice and a held-out test slice
     (default 90 / 30 days).
  2. Fit static OLS on the training slice -> static alpha, beta.
  3. Fit Kalman by MLE on the training slice (Q_alpha, Q_beta, R) and record
     the final filtered state.
  4. On the test slice:
       static residual = y - (alpha_s + beta_s * x)
       kalman residual = forward-filter innovations starting from the
                         trained state, using the trained Q and R (no
                         re-fitting on test data).
  5. ADF on each test-slice residual series.

This avoids the in-sample whitening trap of the earlier comparison: both
methods now have their parameters fixed on training, and both are scored on
data they did not see.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_store import StoreConfig, build_close_panel
from src.kalman_hedge import fit_kalman_mle, kalman_forward_residuals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Out-of-sample static-vs-Kalman comparison.")
    p.add_argument("--pairs-path", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--t0", default="2026-01-01T00:00:00Z", help="End of full window.")
    p.add_argument("--train-days", type=int, default=90)
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--top-pairs", type=int, default=10)
    p.add_argument("--out-dir", required=True)
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

    pairs = pd.read_parquet(args.pairs_path, engine="pyarrow").head(args.top_pairs)
    cfg = StoreConfig(interval="1h", data_dir=args.data_dir)
    t0 = pd.Timestamp(args.t0)
    total_days = args.train_days + args.test_days
    full_start = (t0 - pd.Timedelta(days=total_days)).floor(cfg.interval)
    train_end = (t0 - pd.Timedelta(days=args.test_days)).floor(cfg.interval)
    full_end = t0.floor(cfg.interval)

    symbols = sorted(set(pairs["sym_a"]).union(pairs["sym_b"]))
    panel = build_close_panel(cfg, symbols, full_start, full_end, min_symbol_coverage=0.50)
    if panel.empty:
        raise RuntimeError("Empty close panel; check data store and date range.")

    rows: List[Dict[str, float]] = []
    for _, pair_row in pairs.iterrows():
        sym_a, sym_b = pair_row["sym_a"], pair_row["sym_b"]
        if sym_a not in panel.columns or sym_b not in panel.columns:
            continue
        sub = panel[[sym_a, sym_b]].dropna()
        if sub.empty:
            continue
        logp = np.log(sub)
        train = logp[logp.index < train_end]
        test = logp[(logp.index >= train_end) & (logp.index < full_end)]
        if len(train) < 200 or len(test) < 100:
            continue

        y_tr = train[sym_a].to_numpy(dtype=float)
        x_tr = train[sym_b].to_numpy(dtype=float)
        y_te = test[sym_a].to_numpy(dtype=float)
        x_te = test[sym_b].to_numpy(dtype=float)

        # Static OLS fit on train, evaluate residuals on test
        ols = OLS(y_tr, add_constant(x_tr)).fit()
        alpha_s = float(ols.params[0])
        beta_s = float(ols.params[1])
        static_oos = y_te - (alpha_s + beta_s * x_te)
        static_is = y_tr - (alpha_s + beta_s * x_tr)

        # Kalman MLE on train, forward-filter on test using trained params + state
        try:
            fitted = fit_kalman_mle(y_tr, x_tr)
        except Exception as exc:
            print(f"  {sym_a}_{sym_b}: Kalman MLE failed: {exc}")
            continue
        _, _, kalman_oos = kalman_forward_residuals(y_te, x_te, fitted)

        # For reference also report the in-sample Kalman residuals, computed
        # with the SAME fitted parameters as the OOS pass (not defaults). This
        # is the "cheating" comparison because the residuals are from the same
        # data the parameters were fit on; the contrast with the OOS column
        # below is the whole point of the test.
        kalman_is = np.asarray(fitted["train_residuals"], dtype=float)

        stat_is, p_is = _adf(static_is)
        stat_oos, p_oos = _adf(static_oos)
        kstat_is, kp_is = _adf(kalman_is[50:])
        kstat_oos, kp_oos = _adf(kalman_oos)

        rows.append(
            {
                "pair": f"{sym_a}_{sym_b}",
                "sym_a": sym_a,
                "sym_b": sym_b,
                "train_obs": int(len(train)),
                "test_obs": int(len(test)),
                "kalman_converged": bool(fitted["converged"]),
                "q_alpha": fitted["q_alpha"],
                "q_beta": fitted["q_beta"],
                "r": fitted["r"],
                "beta_train_final": fitted["beta_train_final"],
                "static_beta": beta_s,
                "static_std_oos": float(np.std(static_oos, ddof=1)),
                "kalman_std_oos": float(np.std(kalman_oos, ddof=1)) if len(kalman_oos) > 1 else float("nan"),
                "static_adf_p_is": p_is,
                "static_adf_p_oos": p_oos,
                "kalman_adf_p_is_cheating": kp_is,
                "kalman_adf_p_oos_honest": kp_oos,
            }
        )

    if not rows:
        raise RuntimeError("No pairs produced OOS comparisons.")

    out = pd.DataFrame(rows)
    out = out.sort_values("kalman_adf_p_oos_honest")
    cols = [
        "pair", "train_obs", "test_obs", "kalman_converged",
        "static_beta", "beta_train_final", "q_beta",
        "static_std_oos", "kalman_std_oos",
        "static_adf_p_is", "static_adf_p_oos",
        "kalman_adf_p_is_cheating", "kalman_adf_p_oos_honest",
    ]
    out[cols].to_csv(out_dir / "kalman_oos_comparison.csv", index=False)
    pd.options.display.float_format = "{:.4e}".format
    print(out[cols].to_string(index=False))
    pd.options.display.float_format = None
    print(f"\nSaved {out_dir/'kalman_oos_comparison.csv'}")


if __name__ == "__main__":
    main()
