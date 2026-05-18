# Baseline Comparison Results

## Original Dataset, Fair Test Cap

This reruns only the modeling stage on the saved `artifacts/first_branch`
dataset, adding naive baselines and capping every evaluated model at the same
5,000 most recent test samples.

Output: `artifacts/first_branch_with_baselines/metrics.csv`

| Model | Accuracy | Macro F1 | Trades | Total PnL | PnL Mean/Std | Max Drawdown | Win Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| persist_class | 0.0748 | 0.0464 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| majority_class | 0.4440 | 0.2050 | 5000 | 2.4950 | 0.0319 | -10.0953 | 0.4914 |
| random_stratified | 0.4470 | 0.3407 | 4688 | 0.5498 | 0.0073 | -1.1465 | 0.5064 |
| zscore_rule | 0.1710 | 0.1538 | 1358 | -1.0876 | -0.0319 | -4.1756 | 0.4779 |
| sklearn_hist_gradient_boosting | 0.4856 | 0.3186 | 4996 | 5.1814 | 0.0663 | -5.4414 | 0.5222 |
| lstm | 0.5152 | 0.3476 | 5000 | 10.5653 | 0.1362 | -4.6387 | 0.5474 |
| transformer | 0.5708 | 0.3930 | 5000 | 18.8906 | 0.2486 | -2.9760 | 0.6048 |

The transformer remains strongest after adding naive baselines and enforcing a
common test cap.

## Expanded Data, Skip-Deep Baseline Pass

Run command:

```bash
python3 scripts/run_first_branch.py --out-dir artifacts/more_data_baselines --t0 2026-03-15T00:00:00Z --train-days 365 --liquid-top-n 75 --top-pairs 40 --max-pairs-to-test 0 --max-train-samples 30000 --max-test-samples 10000 --skip-deep
```

This pass uses a later cutoff, a 365-day lookback, 75 liquid symbols, and 40
correlation-fallback pairs. Strict ADF pair testing was intentionally disabled
for the quick pass because it was the runtime bottleneck on the larger panel.

- Local symbols: 204
- As-of universe at 2026-03-15 00:00:00 UTC: 201
- Liquidity-filtered universe: 75
- Training close panel: 7,117 hourly rows by 75 symbols
- Selected fallback pairs: 40
- Feature rows: 318,826
- ML samples: 312,146
- Sequence shape: 312,146 x 168 x 16

Output: `artifacts/more_data_baselines/metrics.csv`

| Model | Accuracy | Macro F1 | Trades | Total PnL | PnL Mean/Std | Max Drawdown | Win Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| persist_class | 0.0723 | 0.0450 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| majority_class | 0.4815 | 0.2167 | 10000 | 9.2850 | 0.0469 | -7.5284 | 0.5342 |
| random_stratified | 0.4392 | 0.3337 | 9354 | -1.2481 | -0.0065 | -2.0616 | 0.4975 |
| zscore_rule | 0.2358 | 0.1882 | 4070 | 2.1395 | 0.0160 | -2.5706 | 0.5233 |
| sklearn_hist_gradient_boosting | 0.4766 | 0.3299 | 9987 | 0.6713 | 0.0034 | -4.5043 | 0.5052 |

The larger dataset is available for a full deep-model run, but this quick pass
shows that the tabular model does not clearly beat simple baselines on
spread-signal PnL. That makes the full LSTM/transformer comparison the next
important check.

## Walk-Forward Evaluation

The earlier model runs used a single chronological train/test split. The saved
datasets did include `walk_split_ids`, but those IDs were annotations and were
not used by the training functions. A true walk-forward tabular evaluation was
added in `scripts/run_walk_forward.py`.

Each split trains on 90 days and tests on the next 30 days, stepping forward by
30 days. The run includes naive baselines, z-score, and the tabular boosted-tree
baseline. Deep models were not included in this walk-forward pass.

### Original Dataset

Run command:

```bash
python3 scripts/run_walk_forward.py --dataset-dir artifacts/first_branch/dataset --out-dir artifacts/walk_forward_first_branch --train-days 90 --test-days 30 --step-days 30 --max-train-samples 12000 --max-test-samples 5000
```

Output: `artifacts/walk_forward_first_branch/walk_forward_summary.csv`

| Model | Splits | Test Samples | Accuracy Mean | Macro F1 Mean | Trades | Total PnL | Win Rate Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zscore_rule | 2 | 10000 | 0.2131 | 0.1868 | 3032 | 20.2147 | 0.6410 |
| sklearn_hist_gradient_boosting | 2 | 10000 | 0.5227 | 0.3547 | 9981 | 10.5265 | 0.5262 |
| random_stratified | 2 | 10000 | 0.4390 | 0.3284 | 9422 | 2.2286 | 0.5029 |
| persist_class | 2 | 10000 | 0.0662 | 0.0414 | 0 | 0.0000 | 0.0000 |
| majority_class | 2 | 10000 | 0.4857 | 0.2179 | 10000 | -6.7347 | 0.4878 |

### Expanded Dataset

Run command:

```bash
python3 scripts/run_walk_forward.py --dataset-dir artifacts/more_data_baselines/dataset --out-dir artifacts/walk_forward_more_data --train-days 90 --test-days 30 --step-days 30 --max-train-samples 30000 --max-test-samples 10000
```

Output: `artifacts/walk_forward_more_data/walk_forward_summary.csv`

| Model | Splits | Test Samples | Accuracy Mean | Macro F1 Mean | Trades | Total PnL | Win Rate Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zscore_rule | 8 | 80000 | 0.1799 | 0.1589 | 21721 | 56.6598 | 0.5421 |
| sklearn_hist_gradient_boosting | 8 | 80000 | 0.4904 | 0.3376 | 79950 | 51.6168 | 0.5082 |
| majority_class | 8 | 80000 | 0.4787 | 0.2157 | 80000 | 25.0264 | 0.5064 |
| random_stratified | 8 | 80000 | 0.4478 | 0.3337 | 75670 | 0.3575 | 0.5012 |
| persist_class | 8 | 80000 | 0.0648 | 0.0405 | 0 | 0.0000 | 0.0000 |

Walk-forward changes the interpretation. The tabular model still has the best
classification metrics, but the z-score rule has slightly higher summed
spread-signal PnL on both walk-forward runs. Since these are still log-spread
signal metrics rather than portfolio returns, this should be treated as a
robustness check, not a trading-performance claim.
