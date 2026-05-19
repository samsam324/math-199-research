# Kalman Pipeline Results

Full pipeline rerun with the spread input swapped from static-OLS residuals
to Kalman dynamic-beta residuals (MLE-fit on the training slice, forward-
rolled on test). All other components (feature set, ML architectures,
walk-forward split structure, HMM regime filter, portfolio backtest)
identical to the May 18 run, so the only thing varying between the two
docs is the spread input.

## Spread-signal evaluation, walk-forward (2 splits, 10000 test samples each)

| Model | Accuracy | Macro F1 | total_pnl | pnl_mean_to_std | win_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| **zscore_rule** | 0.307 | 0.253 | 18.33 | **0.290** | **0.954** |
| sklearn_hist_gradient_boosting | 0.570 | 0.449 | 19.85 | 0.163 | 0.541 |
| lstm | 0.569 | 0.440 | 20.06 | 0.154 | 0.514 |
| transformer | 0.406 | 0.312 | 8.92 | 0.134 | 0.528 |
| random_stratified | 0.377 | 0.333 | 1.80 | 0.019 | 0.515 |
| persist_class | 0.204 | 0.112 | 0.00 | 0.000 | 0.000 |
| majority_class | 0.391 | 0.187 | -12.89 | 0.048 | 0.508 |

The z-score rule on Kalman spreads hits 95.4% win rate at 1237 trades over
the test period; the ML models trade roughly 8x more often (~10000 trades)
and win 51-54% of those bars. ML accuracy is meaningfully higher on the
3-class label (0.57 vs 0.31 for z-score), but trade quality, conditional on
acting, is dramatically worse: more trades, more noise per trade. Same
underlying signal, different operating points on a precision-recall curve.

For comparison, the same models on the static-OLS pipeline (`docs/may18_run_results.md`):

| Model | Static `total_pnl` | Kalman `total_pnl` | Static `win_rate` | Kalman `win_rate` |
| --- | ---: | ---: | ---: | ---: |
| zscore_rule | 20.21 | 18.33 | 0.641 | **0.954** |
| sklearn_hist_gradient_boosting | 10.53 | 19.85 | 0.526 | 0.541 |
| lstm | 19.08 | 20.06 | 0.522 | 0.514 |
| transformer | 7.70 | 8.92 | 0.532 | 0.528 |

Spread-PnL is roughly comparable across pipelines, but win rate on the
classical rule goes from 64% to 95%. The ML rankings shuffle: boost catches
up to LSTM (was 10.5 vs 19.1, now 19.9 vs 20.1); transformer is still
fourth.

## HMM regime filter still hurts (negative result is robust)

Applying the 2-state Gaussian HMM filter to Kalman-pipeline predictions
suppresses 10-15% of trades and drops total_pnl by 30-60% for every model:

| Model | total_pnl raw | total_pnl filtered |
| --- | ---: | ---: |
| zscore_rule | 18.33 | 7.41 |
| sklearn_hist_gradient_boosting | 19.85 | 6.95 |
| lstm | 20.06 | 7.06 |
| transformer | 8.92 | -0.53 |
| majority_class | -12.89 | -5.08 |

Some HMM fits still did not converge on Kalman spreads. The negative result
generalizes: a 2-state Gaussian HMM on the same feature inputs doesn't help
even when fed stationary residuals, so the issue is the model class (or the
features driving it), not the stationarity of the input series.

## Portfolio backtest at 15 bps round-trip: still loses, more painfully

| Model | Static Sharpe | Kalman Sharpe | Kalman n_trades |
| --- | ---: | ---: | ---: |
| persist_class | 0.00 | 0.00 | 0 |
| zscore_rule | -4.83 | -9.26 | 1367 |
| transformer | -13.24 | -100.6 | 3328 |
| majority_class | -18.04 | -86.4 | 3520 |
| lstm | -26.43 | -110.8 | 4130 |
| sklearn_hist_gradient_boosting | -44.05 | -106.5 | 4114 |

Counter-intuitively the Kalman pipeline backtests WORSE in dollar terms,
because the more confident ML signals flip more often per bar, generating
more bar-edge "trades" in the bar-by-bar backtester. Net pre-cost PnL is
positive for the same models that were positive in the spread-signal table
above. The cost wall is bigger, not because the signal is worse, but
because the backtester's notion of "trade" is per-bar signal change rather
than open/close of a discrete position. The honest claim is unchanged:

  **A naive bar-by-bar implementation of these signals does not pay 15 bps
  round-trip per leg.**

The signal exists; what doesn't exist is an alpha source large enough to
trade by simply rebalancing every hour to the latest classifier output.
Closing this gap requires an entry/exit state machine in the backtester
(open at |z| >= 2, hold until |z| <= 0.5), which is the next concrete step.

## What changed because of Kalman, and what didn't

| Finding | Static pipeline | Kalman pipeline |
| --- | --- | --- |
| Z-score rule beats ML on win rate | Yes (0.64 vs ~0.52) | Yes, far more (0.95 vs ~0.52) |
| LSTM is the best ML model on `pnl_mean_to_std` | Yes (0.087) | Roughly tied with boost (0.154 vs 0.163) |
| Transformer wins single-split, loses walk-forward | Yes | Yes |
| 2-state HMM filter helps | No | No |
| Bar-by-bar backtest survives 15 bps | No | No (worse) |

## Reproduce

```bash
python3 scripts/run_first_branch.py --use-kalman --skip-deep --out-dir artifacts/kalman_branch
python3 scripts/run_walk_forward.py \
  --dataset-dir artifacts/kalman_branch/dataset \
  --out-dir artifacts/walk_forward_kalman \
  --deep --dl-epochs 4 \
  --max-train-samples 12000 --max-test-samples 5000
python3 scripts/run_hmm_ablation.py \
  --walk-forward-dir artifacts/walk_forward_kalman \
  --features-dir artifacts/kalman_branch/features \
  --dataset-dir artifacts/kalman_branch/dataset \
  --out-dir artifacts/hmm_kalman
python3 scripts/run_portfolio_backtest.py \
  --predictions-path artifacts/walk_forward_kalman/walk_forward_predictions.parquet \
  --pairs-path artifacts/kalman_branch/selected_pairs.parquet \
  --out-dir artifacts/backtest_kalman
```
