# Leakage audit

A code-level audit of every place information could flow backward in time.
Two real bugs found and fixed, 10 leakage tests written, all pass.

## Method

For each piece of the pipeline: inject a known-future signal at time T,
verify pre-T outputs are byte-identical to a clean run. Identity is the
pass criterion.

Walked the codebase top-down:

1. `src/universe.py` - as-of universe construction
2. `src/pair_selection.py` - cointegration screen, correlation fallback
3. `src/features.py` - per-pair features (16 cols) and 3 targets
4. `src/ml_dataset.py` - sliding-window dataset and 3-class labels
5. `src/modeling.py` - standardization, train/test split, training
6. `src/backtest.py` - position model, cost model, Sharpe calc
7. `src/kalman_hedge.py` - MLE fit and forward residuals
8. `scripts/run_walk_forward.py` - walk-forward driver
9. `scripts/run_first_branch.py` - orchestration

## What was already clean

- Universe first-bar leak was fixed earlier; regression test in place
- Pair selection uses training-window close panel only
- Feature columns are all backward-rolling (verified by spike-injection)
- Targets equal `spread[t+horizon] - spread[t]` within float tolerance
- Deep model standardization uses x_train statistics only (static check)
- Backtester position at time t comes from signal at time t-1 (one-bar
  execution lag)
- Walk-forward driver carves strictly time-ordered windows; no overlap
- Kalman OOS comparison in `run_kalman_oos.py` already fit train and
  forward-rolled test correctly for the OOS column

## Bugs found and fixed

### 1. Kalman parameter discontinuity at the train/test boundary

- Site: `src/features.py` `build_kalman_spread_overrides()`
- What: training spread computed via `kalman_dynamic_hedge()` with default
  `Q_alpha=1e-7, Q_beta=1e-5, R`; test spread used MLE-fitted Q from
  `fit_kalman_mle`. Two different filters on the two halves of the same
  series.
- Why it matters: not strictly look-ahead, but a real distribution shift
  between train and test caused by the implementation. Headline Kalman
  pipeline ran on inconsistent spreads.
- Magnitude on synthetic data: training residual std 0.111, test 0.018
  (6.2x mismatch) -> after fix 0.026 / 0.018 (1.4x). On real Binance.US
  pairs the booster's `pnl_mean_to_std` moved 0.307 -> 0.356 with the
  fix (numbers got better, not worse).
- Fix: `fit_kalman_mle()` now returns `train_residuals` from the
  fitted-parameter forward pass that it already runs. `build_kalman_spread_overrides()`
  uses those instead of calling `kalman_dynamic_hedge()` with defaults.
- Verified by `test_kalman_no_parameter_discontinuity`: boundary std
  ratio 1.09x in the synthetic case.

### 2. Per-pair label threshold leak

- Site: `src/ml_dataset.py` `build_samples_from_features()` per-pair
  label branch
- What: with `--per-pair-label`, the threshold was fit on the first
  `label_train_fraction` (default 0.5) of each pair's data. On the long
  pipeline (180d train + 750d test = 930d total), first 50% extends 9
  months past t0, leaking future spread vol into the label distribution.
- Magnitude: threshold is a single scalar per pair; label distributions
  shift slightly. Small in magnitude but real.
- Fix: `DatasetConfig.label_train_end: Optional[pd.Timestamp]`. When set,
  only data with `timestamp < label_train_end` is used. `run_first_branch.py`
  passes `train_end` (= t0) when `--per-pair-label` is set.
- Verified by `test_per_pair_label_threshold_no_leakage`: injecting a
  post-cutoff spread spike leaves the pre-cutoff class distribution
  byte-identical.

## Considered, not fixed (documented limitations)

- Stale Kalman parameters across walk-forward splits: fit once at t0,
  not refit at each split boundary. Not a leak (params still fit before
  every test window), just suboptimal. The OOS comparison in
  `run_kalman_oos.py` refits per pair on each pair's training slice, so
  the headline finding doesn't depend on this.
- iid bootstrap on overlapping splits: replaced by block bootstrap;
  numbers basically unchanged because per-split metrics aren't strongly
  autocorrelated (models retrain from scratch each split).
- `pnl_mean_to_std` denominator under overlapping returns: replaced by
  HAC (Newey-West, lag 24). iid overstates by ~1.7x; rankings unchanged.

## Tests

`tests/test_leakage_audit.py`. Run: `python3 tests/test_leakage_audit.py`

10 tests, all pass:

1. Universe excludes post-t0 listings (regression for earlier fix)
2. `min_history_days` is a strict cutoff
3. Feature columns identical pre-spike for all 16 features
4. Target = spread[t+24] - spread[t] within float tolerance
5. Per-pair label threshold with `label_train_end` is unaffected by
   post-cutoff spike (pre-cutoff class distribution byte-identical)
6. Kalman boundary std ratio < 5x (was 6x before fix, 1.09x after)
7. Standardization uses x_train statistics only (static code check)
8. Backtester position uses signal at t-1 (injection test)
9. Walk-forward windows strictly time-ordered, no train/test overlap
10. Pair A features unaffected by mutating an unrelated pair B

Pairs with `tests/test_backtest_sanity.py` (6 hand-computable backtester
checks, all pass).

## Verification: headline findings hold post-audit

- Finding 1 (Kalman OOS): unchanged. OOS column never touched the buggy
  code path. Rerun in `artifacts/kalman_oos_post_fix/` confirms 10 of 10
  pairs hold p < 5e-5 OOS.
- Finding 2 (ML rankings): improved on the fixed pipeline. Booster
  pnl_mean_to_std 0.307 -> 0.356; win rate 0.62 -> 0.74. Booster CI
  [0.342, 0.368] is now strictly above z-score CI [0.274, 0.291].
  Z-score per-trade win rate 0.96 unchanged.
- Finding 3 (`latest_spread_z` carries the signal): pending ablation
  rerun on fixed dataset. Expected unchanged because spread_z depends on
  the spread series uniformly.
- Finding 4 (5 bps cost ceiling): pending. Depends on the walk-forward
  predictions in finding 2.

Any further movement folded into `docs/RESULTS.md`.
