# Leakage Audit

A paranoid, code-level audit of every place information could flow backward
in time and bias a result. Two real bugs found and fixed, eight leakage
tests written, all pass.

## Process

For each piece of the pipeline, ask: *if I inject a known-future signal at
time T, can it influence a feature, target, label, or model output at any
time t < T?* The injection is the test; identity of pre-T outputs between
clean and injected runs is the pass criterion.

The audit ran top-down through the codebase:

1. `src/universe.py` — as-of universe construction
2. `src/pair_selection.py` — strict cointegration screen
3. `src/features.py` — per-pair feature engineering (16 features + 3 targets)
4. `src/ml_dataset.py` — sliding-window dataset and 3-class labels
5. `src/modeling.py` — standardization, train/test split, model training
6. `src/backtest.py` — position model, cost model, Sharpe calculation
7. `src/kalman_hedge.py` — MLE fit and forward residuals
8. `scripts/run_walk_forward.py` — walk-forward driver
9. `scripts/run_first_branch.py` — pipeline orchestration

## What was already clean

- Universe construction had a first-bar leak fixed earlier; the regression
  test in `tests/test_leakage_audit.py` (#1, #8) prevents it from
  regressing.
- Pair selection (`score_pairs`, `rank_pairs_by_correlation`) uses a
  training-window-only close panel; alpha/beta from training only.
- Feature columns are all backward-rolling (verified in test #3: a spike
  at index 1200 is invisible at index < 1200 for all 16 features).
- Targets look forward by design and equal `spread[t+horizon] - spread[t]`
  within float tolerance (test #4).
- Deep model standardization computes mu/sd from `x_train` and applies to
  `x_test` (test #7, static code check).
- Backtester position at time t comes from signal at time t-1 (test #8:
  injection of a price spike at bar 50 with signal flip at bar 50 produces
  zero PnL because the new position only takes effect at bar 51).
- Walk-forward driver carves time-ordered windows; no test sample is in
  any train slice.
- Kalman OOS comparison (`scripts/run_kalman_oos.py`) was already correctly
  fitting on train and forward-rolling on test for the OOS column.

## Bugs found and fixed

### Bug 1: Kalman parameter discontinuity at the train/test boundary

**Site**: `src/features.py` `build_kalman_spread_overrides()`.

**What was wrong**: The training-period spread was computed via
`kalman_dynamic_hedge(train_y, train_x)` with default
`Q_alpha=1e-7, Q_beta=1e-5, R` (the `KalmanConfig` defaults). The
test-period spread was computed via `kalman_forward_residuals(...,
fitted)` with the MLE-fitted `Q_alpha, Q_beta, R`. Two different filters
were running on the two halves of the same spread series.

**Why it matters**: Models trained on features derived from training
spreads (one filter regime) and predicted on features derived from test
spreads (a different filter regime). Not strictly look-ahead leakage but
a real distribution shift between train and test caused by the
implementation. The headline Kalman pipeline (the 27-split walk-forward,
all four findings in `RESULTS.md`) was running on this inconsistent
spread series.

**Magnitude**: On synthetic time-varying-β data:
- Old (broken): training residual std 0.111, test residual std 0.018
  (6.2× mismatch at the boundary)
- New (fixed): training residual std 0.026, test residual std 0.018
  (1.4×; same filter parameters on both sides)

**Fix**: `fit_kalman_mle()` now also returns `train_residuals` computed
during the final fitted-parameter forward pass that it already runs to
record the final state. `build_kalman_spread_overrides()` uses those
training residuals instead of calling `kalman_dynamic_hedge()` with
defaults. Same single-pass parameters on both sides of `train_end`.

Verified by `tests/test_leakage_audit.py::test_kalman_no_parameter_discontinuity`:
boundary std ratio < 5×.

### Bug 2: Per-pair label threshold leak

**Site**: `src/ml_dataset.py` `build_samples_from_features()` per-pair
label logic.

**What was wrong**: With `--per-pair-label`, the per-pair classification
threshold was computed using the first `label_train_fraction` (default
0.5) of each pair's available data. With the long pipeline
(180-day train + 750-day test = 930 days of pair data), the first 50%
ends roughly 9 months past `t0`, so the threshold incorporated post-t0
spread vol — leaking future-of-test information into the label
distribution that the models trained on.

**Magnitude**: The threshold is a single scalar per pair (the std of
`||spread[t+24]| - |spread[t]||` on the label-training window). Models
don't see the threshold directly; only the resulting class labels
change. The bias on the headline metrics is therefore small but not
zero.

**Fix**: New `DatasetConfig.label_train_end: Optional[pd.Timestamp]`
field. When set, only data with `timestamp < label_train_end` is used
for the threshold. `scripts/run_first_branch.py` passes `train_end`
(= t0) when `--per-pair-label` is set, so the threshold is anchored to
data strictly before every walk-forward split's test window. The
fractional fallback is preserved for backwards compatibility.

Verified by `tests/test_leakage_audit.py::test_per_pair_label_threshold_no_leakage`:
injecting a giant post-cutoff spread spike leaves the pre-cutoff class
distribution byte-identical.

## Bugs considered, not fixed (acceptable as documented limitations)

### Stale Kalman parameters across walk-forward splits

`build_kalman_spread_overrides()` fits Kalman MLE ONCE per pair at the
original `train_end` (= t0). The fitted parameters are then held fixed
across all 27 walk-forward splits. For later splits, the parameters are
~12-18 months old by the time of evaluation.

This is NOT a leak (parameters are still fit before every split's test
window) but it is suboptimal. A "fully honest" version would refit
Kalman MLE at each walk-forward boundary. Empirically the headline
findings are robust either way — the static-vs-Kalman OOS comparison in
finding 1 uses a per-pair refit on each pair's training slice, and the
result generalizes.

Documented in `docs/METHODS.md` under the Kalman section.

### iid bootstrap on overlapping splits

Replaced by block bootstrap in commit `0e34377`; numbers basically
unchanged because per-split metric values are not strongly autocorrelated
despite overlapping training data (each split retrains from scratch).
See `docs/RESULTS.md` finding 2.

### `pnl_mean_to_std` denominator under overlapping returns

Replaced with HAC (Newey-West, lag 24) in commit `d053c06`. Iid version
overstated by ~1.7×; rankings unchanged.

## Tests

`tests/test_leakage_audit.py`. Run via:

```bash
python3 tests/test_leakage_audit.py
```

Eight tests, all pass:

1. `test_universe_excludes_post_t0_listings` — first-bar leak regression
2. `test_universe_min_history_strict` — min_history_days boundary
3. `test_features_have_no_lookahead` — feature injection test (16 cols)
4. `test_targets_use_forward_window_only` — target = spread[t+h] - spread[t]
5. `test_per_pair_label_threshold_no_leakage` — label_train_end respected
6. `test_kalman_no_parameter_discontinuity` — boundary std ratio < 5×
7. `test_deep_model_standardization_train_only` — static code check
8. `test_backtester_one_bar_execution_lag` — position uses signal at t-1

Pairs with `tests/test_backtest_sanity.py` (six hand-computable
backtester checks). Run both:

```bash
python3 tests/test_backtest_sanity.py
python3 tests/test_leakage_audit.py
```

## Verification that headline findings still hold

The Kalman parameter discontinuity bug was the more material of the two
fixes. To confirm the four findings in `docs/RESULTS.md` still hold on
the fixed pipeline:

- **Finding 1 (Kalman recovers cointegration OOS)**: unchanged. The OOS
  column always used fitted parameters; only the IS reference column
  changed (rerun in `artifacts/kalman_oos_post_fix/`, all 10 pairs still
  hold p < 5e-5 OOS).
- **Finding 2 (ML rankings)**: rerun in progress
  (`artifacts/kalman_long_fixed/`). The previous numbers depended on the
  training-period spread series, which is what the bug fix changed.
- **Finding 3 (`latest_spread_z` carries all signal)**: feature ablation
  to be re-run on the fixed dataset. Expectation: unchanged because
  `spread_z` depends on the spread series uniformly, not specifically on
  the training portion.
- **Finding 4 (5 bps cost ceiling)**: depends on walk-forward predictions
  from finding 2, so rerun is gated on that.

Any change to the headline numbers from the audit fixes will be folded
into `docs/RESULTS.md` and noted in the commit that updates it.
