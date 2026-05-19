"""
Compute Newey-West HAC-corrected pnl_mean_to_std on walk-forward predictions.

The standard pnl_mean_to_std = mean(signal * change_24h) / std(...) is biased
in its denominator because consecutive hourly samples have 23-of-24 hours of
target overlap. Replacing the sample std with the Newey-West long-run
standard deviation (bandwidth = 24) gives a denominator that is consistent
under autocorrelation, so the resulting ratio can be interpreted as a
Sharpe-like statistic that does not over-state precision.

For each model x split, compute:
  pnl = signal * change_24h   (per bar)
  signal = -spread_sign if pred_class==0; +spread_sign if pred_class==2; 0 if 1
  iid_ratio = mean(pnl) / std(pnl)
  hac_ratio = mean(pnl) / hac_lrv(pnl, lag=24).sqrt()

Inflation factor = iid_ratio / hac_ratio. If overlapping samples were truly
iid, factor would be 1.0; higher values mean iid was overstating.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HAC-corrected pnl_mean_to_std on walk-forward predictions.")
    p.add_argument("--predictions-path", required=True, help="walk_forward_predictions.parquet")
    p.add_argument("--out-path", required=True, help="Where to save the per-split CSV")
    p.add_argument("--lag", type=int, default=24, help="Newey-West bandwidth in bars (default 24, matching the 24h target horizon)")
    return p.parse_args()


def hac_long_run_var(x: np.ndarray, lag: int) -> float:
    """
    Newey-West long-run variance estimator with Bartlett kernel.
    Returns the long-run variance per observation (not divided by n).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    x = x - x.mean()
    gamma_0 = float(x @ x) / n
    var = gamma_0
    L = min(lag, n - 1)
    for j in range(1, L + 1):
        gamma_j = float(x[:-j] @ x[j:]) / n
        weight = 1.0 - j / (L + 1)
        var += 2.0 * weight * gamma_j
    return max(var, 1e-20)


def _signal(pred: np.ndarray, spread_sign: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=int)
    spread_sign = np.asarray(spread_sign, dtype=float)
    s = np.zeros_like(spread_sign, dtype=float)
    s[pred == 0] = -spread_sign[pred == 0]
    s[pred == 2] = spread_sign[pred == 2]
    s[spread_sign == 0] = 0.0
    return s


def main() -> None:
    args = parse_args()
    preds = pd.read_parquet(args.predictions_path, engine="pyarrow")
    if "walk_split" not in preds.columns:
        preds["walk_split"] = 0

    rows: List[Dict[str, float]] = []
    for (model, split), g in preds.groupby(["model", "walk_split"], sort=False):
        spread_sign = np.sign(g["current_spread"].to_numpy(dtype=float))
        sig = _signal(g["pred_class"].to_numpy(), spread_sign)
        change = g["y_regression"].to_numpy(dtype=float)
        pnl = sig * change

        # iid version
        if len(pnl) > 1:
            iid_var = float(np.var(pnl, ddof=1))
            iid_std = iid_var ** 0.5
        else:
            iid_std = 0.0
        iid_ratio = float(np.mean(pnl) / iid_std) if iid_std > 0 else 0.0

        # HAC version
        hac_var = hac_long_run_var(pnl, args.lag)
        hac_std = hac_var ** 0.5
        hac_ratio = float(np.mean(pnl) / hac_std) if hac_std > 0 else 0.0

        rows.append({
            "model": model,
            "walk_split": int(split),
            "n_bars": int(len(pnl)),
            "mean_pnl": float(np.mean(pnl)),
            "iid_std": iid_std,
            "hac_std_lag24": hac_std,
            "iid_pnl_mean_to_std": iid_ratio,
            "hac_pnl_mean_to_std_lag24": hac_ratio,
            "inflation_factor": (iid_ratio / hac_ratio) if hac_ratio != 0 else float("nan"),
        })

    df = pd.DataFrame(rows)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    summary = df.groupby("model", as_index=False).agg(
        n_splits=("walk_split", "nunique"),
        iid_mean=("iid_pnl_mean_to_std", "mean"),
        hac_mean=("hac_pnl_mean_to_std_lag24", "mean"),
        inflation_mean=("inflation_factor", "mean"),
        inflation_median=("inflation_factor", "median"),
    ).sort_values("hac_mean", ascending=False)
    print(summary.to_string(index=False))
    print(f"\nSaved per-split values to {out_path}")
    print(f"Mean inflation factor across models = {df['inflation_factor'].mean():.3f}")
    print(f"Median inflation factor = {df['inflation_factor'].median():.3f}")


if __name__ == "__main__":
    main()
