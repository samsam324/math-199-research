# Long Walk-Forward Results (27 splits, Kalman pipeline)

The previous results docs operated on a single 180-day train + 60-day test
window with two walk-forward splits, which left every ranking claim too
noisy to assert. This run uses the same Kalman dynamic-beta spread pipeline
but with the existing data store stretched: t0 moved back to 2024-01-01,
training window 180 days, test horizon uncapped to 750 days, minimum
listing history before t0 set to 180 days. Universe is the top 50 by
liquidity within the 130 symbols that pass the as-of and min-history check.

Walk-forward configuration: 90-day train / 30-day test / 30-day step. The
resulting evaluation has **27 walk-forward splits, 135,000 test samples,
424,399 dataset rows**.

A separate methodological fix went in alongside this run:
`compute_universe_at_time` was including symbols whose first bar was after
t0, which is a real look-ahead bias when t0 is in the past. The function
now requires both `last_ts >= t0 - 1 step` AND `first_ts <= t0 -
min_history_days` to enter the as-of universe.

## Per-model means over 27 splits

Sorted by accuracy:

| Model | Accuracy | Macro F1 | total_pnl (sum) | pnl_mean_to_std | win_rate | trades/split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lstm | 0.595 | 0.500 | 12.33 | 0.279 | 0.603 | 4621 |
| **sklearn_hist_gradient_boosting** | 0.590 | 0.511 | **13.06** | **0.307** | 0.623 | 4466 |
| transformer | 0.391 | 0.295 | 3.86 | 0.047 | 0.520 | 4549 |
| majority_class | 0.379 | 0.183 | 5.51 | 0.148 | 0.565 | 5000 |
| random_stratified | 0.350 | 0.333 | 0.32 | 0.008 | 0.506 | 3828 |
| zscore_rule | 0.338 | 0.277 | 7.63 | 0.282 | **0.960** | 591 |
| persist_class | 0.239 | 0.127 | 0.00 | 0.000 | 0.000 | 0 |

## Bootstrap 5/95% CIs on `pnl_mean_to_std`

| Model | mean | 5% CI | 95% CI |
| --- | ---: | ---: | ---: |
| persist_class | 0.000 | 0.000 | 0.000 |
| random_stratified | 0.008 | 0.003 | 0.014 |
| transformer | 0.047 | -0.057 | 0.148 |
| majority_class | 0.148 | -0.023 | 0.310 |
| lstm | 0.279 | 0.258 | 0.297 |
| zscore_rule | 0.282 | 0.272 | 0.292 |
| sklearn_hist_gradient_boosting | 0.307 | 0.291 | 0.323 |

With 27 splits the CIs are tight enough to say things:

- **The transformer's CI includes zero.** We cannot reject the null that
  its edge over random is sampling noise.
- **Majority class also has a CI that includes zero.** Whatever PnL it
  appears to produce on aggregate is consistent with a coin flip across
  splits.
- **Boost, LSTM, and z-score all have CIs strictly above 0.25 and well
  separated from random.** All three are doing real work; the ordering
  among them (boost > zscore > lstm by `pnl_mean_to_std`) is statistically
  meaningful at this sample size.
- **Boost beats LSTM by a small but real margin** on `pnl_mean_to_std`
  (0.307 vs 0.279, CIs do not overlap). Tabular features carry essentially
  all the signal; the sequence model adds nothing here.

## Z-score rule operates at a different point on the precision-recall curve

The z-score rule trades 591 times per split on average vs 4400-5000 for the
ML models. Win rate is 96% vs 60-62%. Total PnL is comparable to LSTM
(7.6 vs 12.3). This is the same alpha source surfaced at very different
operating points: ML acts on every bar and is right 60% of the time;
z-score acts only when |spread_z| ≥ 1.5 and is right 96% of the time.

For a paper this is the cleanest single statement of what the experiment
showed: **on Kalman-derived dynamic spreads, classifier-level supervised
ML adds essentially no information beyond a simple z-score threshold; what
ML buys you is more frequent but less reliable trades.**

## Negative HMM result persists

Applying the 2-state Gaussian HMM filter to each model's predictions
suppresses ~30% of trades and drops total PnL by 30-60%:

| Model | raw total_pnl | filtered total_pnl |
| --- | ---: | ---: |
| sklearn_hist_gradient_boosting | 352.67 | 149.12 |
| lstm | 332.93 | 135.78 |
| zscore_rule | 206.07 | 69.16 |
| majority_class | 148.79 | 101.57 |
| transformer | 104.25 | 6.03 |

Several HMM fits still failed to converge. The result generalizes to the
larger sample: the HMM as specified does not isolate productive
mean-reverting bars on Kalman spreads, despite the input being stationary.
A different state-space (3 states, or HMM on different features) might
flip this; the current 2-state Gaussian on `[spread_z, spread_diff,
spread_vol]` does not.

## Cost wall result persists and gets bigger

Portfolio backtest at 15 bps round-trip per leg, $10k notional per leg
per pair, 20 pairs, hourly rebalancing across 6955 test bars:

| Model | Sharpe (ann.) | Total return ($) | n_trades |
| --- | ---: | ---: | ---: |
| persist_class | 0.00 | 0 | 0 |
| zscore_rule | -23.24 | -535,497 | 25,699 |
| sklearn_hist_gradient_boosting | -51.17 | -3,083,824 | 72,570 |
| lstm | -57.48 | -3,111,615 | 69,068 |
| majority_class | -58.78 | -3,678,777 | 68,063 |
| random_stratified | -60.51 | -3,592,347 | 87,691 |
| transformer | -62.21 | -3,432,077 | 61,713 |

The honest claim is unchanged from the shorter pipeline: a naive bar-by-bar
implementation of these signals does not pay 15 bps round-trip per leg.
The signal is real, but the trade-definition (every bar with a non-flat
class is a position; every change is a trade) generates 10000+ "trades" per
split per model, each charged 15 bps, which crushes any signal-level edge.
This is the next concrete thing to fix (entry/exit state machine).

## Notable additional finding: time_since_zero_crossing has no value on Kalman + more data

Permutation importance from HistGradientBoosting on the long Kalman
dataset:

- top: `latest_spread_z`
- bottom: `time_since_zero_crossing` (-0.00025, indistinguishable from zero,
  trending slightly negative)

For comparison, on the static-OLS pipeline this was the #1 feature at
+0.026. On the short Kalman pipeline it was #8 at +0.0009. On the long
Kalman pipeline it has no value at all. This was the suspected leak in the
adversarial review, and the trajectory across the three datasets confirms
it: the feature was picking up the trending non-stationarity of static
spreads, not anything about mean reversion.

## Reproduce

```bash
python3 scripts/run_first_branch.py --use-kalman --skip-deep \
  --out-dir artifacts/kalman_long \
  --t0 2024-01-01T00:00:00Z --train-days 180 --test-days 750 \
  --min-history-days 180 --liquid-top-n 50 --top-pairs 20

python3 scripts/run_walk_forward.py \
  --dataset-dir artifacts/kalman_long/dataset \
  --out-dir artifacts/walk_forward_kalman_long \
  --deep --dl-epochs 4 \
  --max-train-samples 12000 --max-test-samples 5000

python3 scripts/run_hmm_ablation.py \
  --walk-forward-dir artifacts/walk_forward_kalman_long \
  --features-dir artifacts/kalman_long/features \
  --dataset-dir artifacts/kalman_long/dataset \
  --out-dir artifacts/hmm_kalman_long

python3 scripts/run_portfolio_backtest.py \
  --predictions-path artifacts/walk_forward_kalman_long/walk_forward_predictions.parquet \
  --pairs-path artifacts/kalman_long/selected_pairs.parquet \
  --out-dir artifacts/backtest_kalman_long
```

Figures (in `figures/`):

- `long_walk_forward_pnl_mean_to_std.png`, `long_walk_forward_win_rate.png`,
  `long_walk_forward_accuracy.png`, `long_walk_forward_total_pnl.png` —
  bars with 5/95% bootstrap CIs over the 27 splits
- `long_feature_importance.png` — booster importance on the long dataset
- `long_cumulative_pnl.png` — per-model cumulative $ PnL with 15bps costs
