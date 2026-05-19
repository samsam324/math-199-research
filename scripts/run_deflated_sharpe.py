"""
Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

Corrects observed Sharpe ratios for:
  1. Selection bias from trying N strategies and reporting the best
  2. Non-normality of returns (skewness, kurtosis)

Formula:
                  (SR - SR_0) * sqrt(T - 1)
  DSR = Phi( -------------------------------- )
              sqrt( 1 - g3*SR + (g4-1)/4 * SR^2 )

where:
  SR_0 = sqrt(V) * [ (1 - gamma_e) * Z^{-1}(1 - 1/N) + gamma_e * Z^{-1}(1 - 1/(N*e)) ]
  V    = sample variance of the Sharpe ratios across the N trials
  gamma_e ~ 0.5772 (Euler-Mascheroni)
  T    = number of return observations
  g3   = skewness of returns
  g4   = kurtosis of returns

DSR is a p-value: probability that the true Sharpe is > 0 given the
observed SR, the number of trials, and the non-normality of returns.
A DSR of 0.95 means we can reject "true Sharpe is 0" at 5% after accounting
for trial bias.

Input: portfolio_returns_<model>.csv files written by run_portfolio_backtest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats


GAMMA_E = 0.5772156649015329
BARS_PER_YEAR = 24 * 365


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deflated Sharpe Ratio across a directory of portfolio_returns_*.csv files.")
    p.add_argument("--backtest-dir", required=True, help="Directory containing portfolio_returns_*.csv")
    p.add_argument("--n-trials", type=int, default=50,
                   help="Number of strategy configurations tried across the project. Conservative default 50.")
    p.add_argument("--label", default="", help="Label for this run in the output (e.g. '5 bps').")
    p.add_argument("--out-path", default=None, help="Optional CSV path to write per-model DSR.")
    return p.parse_args()


def per_bar_returns(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if "pnl" not in df.columns:
        raise ValueError(f"{csv_path} has no pnl column")
    # Convert $ pnl to per-bar return on the capital base; the absolute scale
    # cancels out of Sharpe, but skewness and kurtosis are scale-invariant.
    return df["pnl"].to_numpy(dtype=float)


def sharpe(r: np.ndarray) -> tuple:
    """Returns per-bar mean, per-bar std, per-bar Sharpe, annualized Sharpe."""
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0, 0.0, 0.0, 0.0
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    sr_bar = mu / sd if sd > 0 else 0.0
    sr_ann = sr_bar * np.sqrt(BARS_PER_YEAR)
    return mu, sd, sr_bar, sr_ann


def deflated_sharpe(sr_observed: float, sr_var_across_trials: float, n_trials: int,
                    skew: float, kurt: float, T: int) -> tuple:
    """
    Returns (sr_0, dsr_pvalue).

    sr_observed and sr_var_across_trials should be in PER-BAR units.
    sr_0 = expected max Sharpe under the null over n_trials.
    dsr = probability the true SR > 0 after correction.
    """
    sqrt_v = float(np.sqrt(max(sr_var_across_trials, 1e-30)))
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr_0 = sqrt_v * ((1 - GAMMA_E) * z1 + GAMMA_E * z2)

    # Bias-corrected Sharpe's variance accounting for skew + kurtosis
    denom = np.sqrt(max(1 - skew * sr_observed + (kurt - 1) / 4 * sr_observed * sr_observed, 1e-12))
    test_stat = (sr_observed - sr_0) * np.sqrt(T - 1) / denom
    dsr = float(stats.norm.cdf(test_stat))
    return float(sr_0), dsr


def main() -> None:
    args = parse_args()
    backtest_dir = Path(args.backtest_dir)
    paths = sorted(backtest_dir.glob("portfolio_returns_*.csv"))
    if not paths:
        raise RuntimeError(f"No portfolio_returns_*.csv in {backtest_dir}")

    # Step 1: per-model return series
    series = {}
    for path in paths:
        model = path.stem.replace("portfolio_returns_", "")
        series[model] = per_bar_returns(path)

    # Step 2: per-model Sharpe in per-bar and annualized units
    rows = []
    for model, r in series.items():
        mu, sd, sr_bar, sr_ann = sharpe(r)
        rows.append({
            "model": model,
            "T": int(len(r[np.isfinite(r)])),
            "mean_pnl": mu,
            "std_pnl": sd,
            "sr_per_bar": sr_bar,
            "sr_annualized": sr_ann,
            "skew": float(stats.skew(r[np.isfinite(r)])) if len(r) > 2 else 0.0,
            "kurt": float(stats.kurtosis(r[np.isfinite(r)], fisher=False)) if len(r) > 2 else 3.0,
        })
    df = pd.DataFrame(rows)

    # Step 3: variance of per-bar Sharpe across trials (use ALL models present)
    sr_var = float(np.var(df["sr_per_bar"].to_numpy(dtype=float), ddof=1))

    # Step 4: DSR per model
    n = args.n_trials
    for _, row in df.iterrows():
        sr_0, dsr = deflated_sharpe(
            sr_observed=row["sr_per_bar"],
            sr_var_across_trials=sr_var,
            n_trials=n,
            skew=row["skew"],
            kurt=row["kurt"],
            T=row["T"],
        )
        df.loc[df["model"] == row["model"], "sr_0_per_bar"] = sr_0
        df.loc[df["model"] == row["model"], "sr_0_annualized"] = sr_0 * np.sqrt(BARS_PER_YEAR)
        df.loc[df["model"] == row["model"], "DSR"] = dsr

    df = df.sort_values("sr_annualized", ascending=False)
    cols = ["model", "T", "sr_per_bar", "sr_annualized", "sr_0_per_bar", "sr_0_annualized", "DSR", "skew", "kurt"]
    if args.label:
        print(f"\n=== Deflated Sharpe, N_trials = {n}, label = '{args.label}' ===")
    else:
        print(f"\n=== Deflated Sharpe, N_trials = {n} ===")
    print(f"variance of per-bar SR across {len(df)} models: {sr_var:.3e}")
    print(f"sr_0 (expected max SR under null, annualized): {df['sr_0_annualized'].iloc[0]:.4f}")
    print()
    print(df[cols].to_string(index=False))

    print("\nInterpretation:")
    print(" - DSR > 0.95 => reject 'true Sharpe = 0' at 5% after N-trial correction")
    print(" - DSR ~ 0.5  => observed Sharpe is exactly what you'd expect from chance")
    print(" - DSR < 0.5  => observed Sharpe is BELOW the chance-best, very likely null")

    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df[cols].to_csv(out_path, index=False)
        print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
