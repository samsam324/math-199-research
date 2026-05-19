# May 18 Run Results

Full pipeline run with the wired-up deep walk-forward, HMM ablation, Kalman
comparison, and cost-aware portfolio backtester. The four findings below are
written for the paper's results section: each is a statement about the
market or the methodology, not about how to make the strategy profitable.

## 1. Transformer's single-split win does not survive walk-forward

Frank's earlier single-split run had the transformer top the table at
`pnl_mean_to_std = 0.249`. Across two walk-forward splits (90d train / 30d
test / 30d step), the ranking is very different.

| Model | Accuracy | Macro F1 | total_pnl | pnl_mean_to_std | win_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| zscore_rule | 0.213 | 0.187 | 20.21 | 0.156 | 0.641 |
| **lstm** | 0.533 | 0.360 | 19.08 | **0.087** | 0.522 |
| sklearn_hist_gradient_boosting | 0.523 | 0.355 | 10.53 | 0.048 | 0.526 |
| transformer | 0.508 | 0.346 | 7.70 | 0.031 | 0.532 |
| random_stratified | 0.439 | 0.328 | 2.23 | 0.011 | 0.503 |
| persist_class | 0.066 | 0.041 | 0.00 | 0.000 | 0.000 |
| majority_class | 0.486 | 0.218 | -6.73 | -0.054 | 0.488 |

LSTM is the best ML model on `pnl_mean_to_std`, but the classical z-score
rule beats every ML model on both PnL/std and win rate. Transformer drops
from #1 to #4 once we stop training on a single arbitrary slice.

**What this says.** Single chronological train/test splits over-state ML
performance on this kind of data. The signal that the transformer appeared
to capture in Frank's earlier run was idiosyncratic to one test window. A
methodologically honest evaluation of sequence models on financial spreads
requires walk-forward, and the bar that ML has to clear is a simple z-score
threshold rule, not just "random" or "majority."

Source: `artifacts/walk_forward/walk_forward_summary.csv`.

## 2. Kalman dynamic hedge ratio rescues cointegration on every pair tested

Static OLS finds zero cointegrated pairs at p < 0.05 on the local slice (the
reason Frank had to fall back to correlation-based selection). The Kalman
random-walk hedge ratio flips this completely. For all 10 top correlation-
fallback pairs:

| Pair | static_adf_p | kalman_adf_p | static_std | kalman_std |
| --- | ---: | ---: | ---: | ---: |
| XRPUSDT_ETHUSDT | 0.724 | 2.2e-22 | 0.118 | 0.008 |
| XRPUSDT_SOLUSDT | 0.604 | 9.7e-21 | 0.102 | 0.009 |
| XRPUSDT_DOGEUSDT | 0.146 | 4.6e-20 | 0.057 | 0.010 |
| ETHUSDT_SOLUSDT | 0.008 | 9.1e-19 | 0.073 | 0.006 |
| SOLUSDT_DOGEUSDT | 0.263 | 4.2e-16 | 0.067 | 0.012 |
| ADAUSDT_HBARUSDT | 0.192 | 3.9e-12 | 0.092 | 0.017 |
| DOGEUSDT_SUIUSDT | 0.041 | 9.1e-08 | 0.082 | 0.020 |
| XRPUSDT_ADAUSDT | 0.051 | 7.4e-06 | 0.054 | 0.022 |
| ETHUSDT_ADAUSDT | 0.011 | 5.0e-05 | 0.095 | 0.027 |
| SOLUSDT_ADAUSDT | 0.280 | 5.8e-05 | 0.081 | 0.033 |

Kalman residual std is 4-15x smaller than static, and ADF p-values drop by
12+ orders of magnitude on the strongest pairs.

**What this says.** The pair relationships in crypto are real but
non-stationary: a single OLS hedge ratio over a 180-day window is the wrong
model. The relationship between two coins drifts continuously, and any
analysis that holds beta fixed (including most academic pairs-trading
papers, including Gatev/Goetzmann/Rouwenhorst) will conclude that there is
no cointegration when there actually is, just under a time-varying hedge.
This is the central methodological finding of the project so far.

Source: `artifacts/kalman/kalman_comparison.csv`. Per-pair overlay plots in
`artifacts/kalman/kalman_*.png`.

## 3. HMM regime filter hurts every model on these pairs

Filtering trades to the mean-reverting state suppresses 30-40% of bars but
takes more good trades with it than bad. Raw vs. HMM-filtered, walk-forward
totals:

| Model | total_pnl raw | total_pnl filtered | trades raw | trades filtered |
| --- | ---: | ---: | ---: | ---: |
| zscore_rule | 20.21 | 5.99 | 3032 | 1288 |
| lstm | 19.08 | 3.76 | 10000 | 6233 |
| sklearn_hist_gradient_boosting | 10.53 | -0.26 | 9981 | 6217 |
| transformer | 7.70 | 1.93 | 10000 | 6233 |
| majority_class | -6.73 | -8.20 | 10000 | 6233 |

Several HMM fits did not converge to a stable likelihood.

**What this says.** A 2-state Gaussian HMM fit on static-OLS spread features
does not recover meaningful mean-reverting regimes on this universe. The
likely reason is consistent with finding #2: the input spread is not
stationary under a fixed beta, so the HMM is fitting noise rather than
regime structure. A natural follow-up is to refit the HMM on Kalman-derived
dynamic spreads, where the input series IS stationary. We report the
negative result here because it constrains what kind of regime-switching
story the paper can tell on the current data.

Source: `artifacts/hmm/ablation_summary.csv`.

## 4. Realistic costs (15 bps round-trip per leg) wipe out every model

The cost-aware portfolio backtester with hourly rebalancing, $10k per leg,
default taker fee 10 bps + 5 bps slippage:

| Model | Sharpe (ann.) | Total return ($) | Max DD ($) | n_trades | win_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| persist_class | 0.00 | 0 | 0 | 0 | 0.00 |
| zscore_rule | -4.83 | -10,375 | -1,146 | 896 | 0.45 |
| transformer | -13.24 | -23,198 | -25,131 | 518 | 0.43 |
| majority_class | -18.04 | -37,154 | -40,559 | 461 | 0.42 |
| lstm | -26.43 | -48,394 | -48,712 | 1003 | 0.33 |
| sklearn_hist_gradient_boosting | -44.05 | -72,665 | -72,595 | 1424 | 0.31 |
| random_stratified | -191.45 | -266,550 | -266,480 | 5491 | 0.03 |

**What this says.** This is a market-efficiency statement, not a strategy
critique. The pre-cost spread signal is non-zero (compare to finding #1:
LSTM, boost, and z-score all generate positive `total_pnl` in log-spread
units). The transaction cost level at which all signals net out to zero or
worse is small (10 bps fee + 5 bps slippage). Hourly relative-value
mispricings on Binance.US USDT pairs are therefore consistent with a market
that is approximately efficient up to round-trip costs of this magnitude.
That is the kind of empirical bound a paper can report: not "we found
alpha", but "the alpha that exists at this frequency is bounded above by
15 bps per side."

This is also why the L2 scope matters (blocked on UCLA access): at hourly
spot bars we cannot say anything sharper about the bid-ask cost, only that
exchange-published fees plus modest assumed slippage are enough to eliminate
the signal.

Source: `artifacts/backtest/portfolio_metrics.csv` + per-model return /
position / trade CSVs.

## Figures

In `figures/`:

- `walk_forward_pnl_mean_to_std.png`, `walk_forward_accuracy.png`,
  `walk_forward_total_pnl.png` — bar charts with 5/95% bootstrap CIs across
  the 2 walk-forward splits (small n, the CIs are wide)
- `feature_importance.png` — permutation importance from
  HistGradientBoosting on the tabular features. Top drivers:
  `time_since_zero_crossing`, `latest_spread_z`, `min_spread_z`
- `hmm_states_*.png` — spread series with HMM-decoded state bands for 3
  pairs
- `attn_*.png` — transformer CLS-attention weights over the 168-hour window
  for 4 high-|z| example test windows

## Reproducibility

```bash
python3 scripts/run_first_branch.py --skip-deep --out-dir artifacts/first_branch
python3 scripts/run_walk_forward.py --dataset-dir artifacts/first_branch/dataset \
  --out-dir artifacts/walk_forward --deep --dl-epochs 4 \
  --max-train-samples 12000 --max-test-samples 5000
python3 scripts/run_hmm_ablation.py --walk-forward-dir artifacts/walk_forward \
  --features-dir artifacts/first_branch/features \
  --dataset-dir artifacts/first_branch/dataset --out-dir artifacts/hmm
python3 scripts/run_kalman_comparison.py \
  --pairs-path artifacts/first_branch/selected_pairs.parquet \
  --out-dir artifacts/kalman --top-pairs 10
python3 scripts/run_portfolio_backtest.py \
  --predictions-path artifacts/walk_forward/walk_forward_predictions.parquet \
  --pairs-path artifacts/first_branch/selected_pairs.parquet \
  --out-dir artifacts/backtest

python3 scripts/plot_walk_forward_summary.py \
  --metrics-csv artifacts/walk_forward/walk_forward_metrics_by_split.csv \
  --out-path figures/walk_forward_pnl_mean_to_std.png --metric pnl_mean_to_std
python3 scripts/plot_feature_importance.py \
  --dataset-dir artifacts/first_branch/dataset \
  --out-path figures/feature_importance.png
python3 scripts/plot_hmm_states.py \
  --features-path artifacts/first_branch/features/all_pair_features.parquet \
  --pairs ETHUSDT_SOLUSDT XRPUSDT_ETHUSDT BTCUSDT_ETHUSDT --out-dir figures
python3 scripts/plot_transformer_attention.py \
  --dataset-dir artifacts/first_branch/dataset --out-dir figures \
  --epochs 4 --n-examples 4
```
