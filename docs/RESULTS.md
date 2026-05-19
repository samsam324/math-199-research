# Results

Consolidated findings on the long Kalman pipeline (27 walk-forward splits,
424,399 dataset samples, 20 top-liquidity USDT pairs, t0 = 2024-01-01,
180-day training, 750-day test horizon stepped 90/30/30).

## Finding 1: Static OLS hedge ratios fail out-of-sample; Kalman dynamic β recovers cointegration

The strict cointegration screen finds zero passing pairs on the local
Binance.US slice. Pair selection therefore uses a correlation fallback.
The natural diagnostic question is whether the underlying relationships
are non-stationary or simply non-cointegrated.

For each of the top 10 correlation-ranked pairs, both methods were fitted
on a 90-day training slice and evaluated on a held-out 30-day test slice:

- **Static**: OLS hedge ratio on training, residuals `y_test - (α + β·x_test)`.
- **Kalman**: filter parameters `(Q_α, Q_β, R)` MLE-fitted on training only.
  Filter forward-rolled across the test slice starting from the trained
  final state. No re-fitting on test data.

ADF test on the resulting test-slice residuals:

| Pair | static OOS p | Kalman OOS p | Kalman β std (test) |
| --- | ---: | ---: | ---: |
| DOGEUSDT_SUIUSDT  | 0.923 | 1.5e-7 | 7.4e-6 |
| SOLUSDT_DOGEUSDT  | 0.985 | 2.2e-7 | 4.6e-6 |
| XRPUSDT_SOLUSDT   | 0.373 | 3.0e-7 | 6.8e-8 |
| XRPUSDT_DOGEUSDT  | 0.092 | 2.5e-6 | 3.5e-6 |
| ADAUSDT_HBARUSDT  | 0.094 | 7.3e-6 | 3.9e-6 |
| SOLUSDT_ADAUSDT   | 0.823 | 2.9e-5 | 2.5e-5 |
| ETHUSDT_ADAUSDT   | 0.411 | 3.4e-5 | 2.9e-5 |
| XRPUSDT_ADAUSDT   | 0.011 | 1.5e-4 | 2.6e-5 |
| XRPUSDT_ETHUSDT   | 0.029 | 2.1e-4 | 2.8e-5 |
| ETHUSDT_SOLUSDT   | 0.013 | 8.0e-4 | 2.0e-5 |

**Static OLS holds cointegration at p < 0.05 on 2 of 10 pairs out of sample.
Kalman holds it on 10 of 10 at p < 0.001 OOS**, worst pair p = 8e-4. The
MLE-fitted `Q_β` lands in [1e-8, 3e-5], meaning the filter chooses very
slow β drift — slow enough that it could not plausibly be whitening
arbitrary noise. The relationships are real but non-stationary; a fixed
hedge ratio is the wrong model.

This finding required adversarial review (the original in-sample ADF
comparison ran the filter forward over the whole window and tested its
own residuals — circular). The OOS protocol above resolves that. See
`docs/long_pipeline_results.md` and `scripts/run_kalman_oos.py`.

Source: `artifacts/kalman_oos/kalman_oos_comparison.csv`.

## Finding 2: Once Kalman spreads are used, classical z-score matches ML on signal quality, dominates on trade selectivity

Walk-forward across 27 splits (135,000 test samples total). Block-bootstrap
5/95% CIs on per-split `pnl_mean_to_std` with block size 3 (the natural
choice for 90-day training windows stepped by 30 days, which gives roughly
3-step training overlap):

| Model | mean | 5/95% CI | per-trade win rate | trades/split |
| --- | ---: | ---: | ---: | ---: |
| **sklearn_hist_gradient_boosting** | **0.307** | [0.292, 0.323] | 0.62 | 4,466 |
| zscore_rule  | 0.282 | [0.274, 0.291] | **0.96** | 591 |
| lstm         | 0.279 | [0.253, 0.300] | 0.60 | 4,621 |
| majority_class | 0.148 | [0.034, 0.264] | 0.56 | 5,000 |
| transformer  | 0.047 | [-0.060, 0.149] | 0.52 | 4,549 |
| random_stratified | 0.008 | [0.003, 0.013] | 0.51 | 3,828 |
| persist_class | 0.000 | [0.000, 0.000] | 0.00 | 0 |

Boost edges LSTM and z-score by a small but real margin (CIs do not
overlap). LSTM and z-score are statistically tied. **Transformer's CI
includes zero** — we cannot reject the null that its edge over random is
sampling noise. Majority class CI does not include zero under block
bootstrap (it did under iid), but its 0.15 mean and wide 0.034-0.264 CI
make it inferior to all the trading-aware models.

iid bootstrap (block_size=1) was checked as a robustness comparison;
results are essentially identical for the trading models because models
retrain from scratch each split, so per-split metric values are not
strongly autocorrelated despite overlapping training data. The block
bootstrap is reported here as the more defensible default.

Same signal, different operating point: z-score acts only when
|spread_z| ≥ 1.5 and is right 96% of the time on the trades it makes; ML
acts on every bar and is right 60% of the time. ML buys more frequent
but less reliable trades; z-score buys very selective and very reliable
ones.

Source: `artifacts/walk_forward_kalman_long/walk_forward_summary.csv`.

## Finding 3: The ML signal reduces to one feature

Feature ablation: drop one tabular feature at a time, retrain the booster,
report the delta in held-out metrics. Single chronological split for the
ablation comparison (it is an ablation, not a ranking).

| Feature dropped | Δ accuracy | Δ macro F1 | Δ total_pnl |
| --- | ---: | ---: | ---: |
| (none, baseline) | 0.597 | 0.512 | 26.19 |
| **`latest_spread_z`** | **-0.183** | **-0.151** | **-23.07** |
| `time_since_zero_crossing` | +0.002 | +0.006 | -0.21 |
| `mean_volume_ratio` | +0.006 | +0.013 | -0.20 |
| `latest_btc_return_24h` | 0.000 | -0.003 | +0.23 |
| `latest_rolling_corr` | -0.002 | -0.008 | -0.00 |
| `latest_realized_vol_a` | -0.002 | -0.011 | +0.03 |
| (every other feature) | ≤ ±0.006 | ≤ ±0.013 | ≤ ±0.5 |

`latest_spread_z` carries essentially all of the booster's signal. Drop
it and the model loses 18 points of accuracy, 15 points of F1, and 23 of
26 PnL units. Every other feature is at noise level. **The ML models are
learning a fancy version of the z-score threshold rule**, consistent
with finding 2.

`time_since_zero_crossing` was suspected as a soft label leak in an
earlier adversarial review. Its trajectory across pipelines is decisive:

| Spread input | `time_since_zero_crossing` importance |
| --- | ---: |
| Static OLS spread (short dataset) | +0.026 (rank #1) |
| Kalman spread (short dataset) | +0.001 (rank #8) |
| Kalman spread (long dataset) | -0.000 (no signal) |

On static spreads the feature picked up trending non-stationarity, not
mean reversion. On the right (stationary) input it is dead. The
adversarial-review concern is confirmed and resolved.

Source: `docs/feature_ablation_kalman_long.csv`.

## Finding 4: Hourly USDT pair-spread alpha has a cost ceiling of ~5 bps round-trip per leg

Backtester: cost-aware, dollar-hedged, with proper entry/exit state
machine. Open when flat AND `pred_class == 0` (revert) AND
|spread_z| ≥ entry_z. Close when |spread_z| ≤ exit_z OR `pred_class == 2`
(diverge) OR spread sign flips. Run on walk-forward predictions across 27
splits, 6955 test bars, 20 pairs, $10k notional per leg per pair, entry_z=2,
exit_z=0.5.

| Round-trip cost per leg | LSTM Sharpe (ann.) | LSTM total $ | zscore Sharpe | zscore total $ |
| --- | ---: | ---: | ---: | ---: |
| **0 bps** | **8.06** | **+125,340** | 7.83 | +125,116 |
| **5 bps** (maker)  | 0.69 | +10,604 | 0.74 | +11,654 |
| **15 bps** (Binance.US taker) | -14.25 | -218,870 | -13.73 | -215,270 |

The break-even point is approximately 5 bps round-trip per leg. Below
that, the strategy makes money; above it, the spread reversion is not
strong enough to overcome the cost drag. The pre-cost Sharpe ≈ 8 with
61% per-trade win rate confirms that the underlying signal is real.

The market-efficiency statement: at hourly cadence with a textbook
entry/exit state machine on Kalman dynamic spreads,
**USDT pair-spread alpha on Binance.US is bounded above by approximately
5 bps in round-trip transaction cost per leg.** Binance.US default taker
fees (10 bps per side, 20 bps round-trip) exceed this threshold by 4x;
maker fees (~2.5 bps per side, 5 bps round-trip) sit at the boundary.

Source: `artifacts/backtest_sm_zerocost/`, `artifacts/backtest_sm_5bps/`,
`artifacts/backtest_kalman_long_sm/`. Full table in
`docs/state_machine_results.md`.

The backtester is independently sanity-checked by six hand-computable
tests in `tests/test_backtest_sanity.py` (all pass), including a
Sharpe annualization test that would have failed under the earlier
denominator bug.

## Negative results (both robust across spread definitions and walk-forward splits)

### Negative 1: 2-state Gaussian HMM regime filter does not help

Filtering predictions to the inferred "mean-reverting" state suppresses
30% of bars and drops total PnL by 30-60% on every model, on both static
and Kalman spreads. Several fits failed to converge.

| Model | total_pnl raw | total_pnl HMM-filtered |
| --- | ---: | ---: |
| sklearn_hist_gradient_boosting | 352.67 | 149.12 |
| lstm | 332.93 | 135.78 |
| zscore_rule | 206.07 | 69.16 |
| transformer | 104.25 | 6.03 |

The 2-state Gaussian model on `[spread_z, spread_diff, spread_vol]` does
not isolate productive bars. A 3-state HMM or one on different feature
combinations might flip this; the current configuration does not.

Source: `artifacts/hmm_kalman_long/ablation_summary.csv`.

### Negative 2: Single chronological train/test splits dramatically over-state ML

An earlier single-split run had the transformer leading at
`pnl_mean_to_std = 0.25`. With 27 walk-forward splits the transformer
drops to #4 with a 5/95% CI of [-0.057, 0.148] — indistinguishable from
random. The single-split #1 finish was a sampling artifact.

This affects any conclusion about model rankings drawn from a single
chronological split, including most published crypto-pairs ML results.
**Walk-forward with bootstrap CIs is the minimum standard.**

## Methodological caveat on the transformer

The walk-forward result above used a small transformer matched in
parameter count to the LSTM (2 layers, 2 heads, d_model=32, 4 epochs).
A larger transformer (4 layers, 4 heads, d_model=128, 20 epochs with
cosine LR schedule and validation-based early stopping) is currently
in the rerun queue. A single-split sanity check shows the larger model
recovers strongly on this data:

| Model | accuracy (single split) | win_rate (single split) | `pnl_mean_to_std` |
| --- | ---: | ---: | ---: |
| small transformer | 0.584 | 0.156 | -0.009 |
| big transformer | 0.715 | 0.913 | 0.380 |

The big-transformer walk-forward result is pending. If it confirms the
single-split improvement, the "transformer is no better than random"
conclusion in finding 2 should be revised to "the small transformer
matched to LSTM capacity is no better than random; a properly-sized and
properly-trained transformer is competitive with boost and LSTM."

If the big-transformer walk-forward does not confirm the single-split
recovery, the finding 2 conclusion stands robustly.

Either outcome is honest and useful.

## Limitations

- **Single exchange.** All findings are Binance.US-specific. The 5 bps
  cost ceiling depends on Binance.US fee structure; Coinbase or Kraken
  would give different numbers.
- **Single quote currency.** USDT pairs only.
- **Single time interval.** Hourly bars. Lower-frequency alpha (4h,
  daily) may exist with different cost characteristics.
- **Survivorship bias.** Quantified at 6.7% upper bound (14 delistings
  in window), ~5% realistic after the liquidity filter. See
  `docs/survivorship_bias.md` and `docs/binance_us_delistings.md`.
- **Funding cost on synthetic shorts not modeled.** Spot USDT shorting
  on Binance.US in practice requires margin or perp futures with their
  own funding rates. The backtester treats short legs as costless to
  hold.
- **Flat slippage, no market impact.** Until L2 data lands (Phase 2),
  slippage is a constant in bps not a function of trade size.
- **t0 sensitivity not exhaustively tested.** All results above use
  t0 = 2024-01-01. Other choices (2023, 2025) may give different
  pictures driven by different market regimes; this is on the
  Phase-2 list.
- **Three-class label is a heuristic.** A direct regression target
  (next 24h spread change) is an obvious alternative; not run here.
- **HMM tried only one configuration.** 3-state, multiple random
  starts, or different feature combinations might flip the negative
  result.
- **`pnl_mean_to_std` is not a proper Sharpe.** Samples are overlapping
  24h spread changes so they are not iid; HAC-corrected SE or block
  bootstrap on per-trade returns is a more defensible statistic.
  Numbers reported with this caveat noted.

## Phase 2 (pending L2 access)

- L2 microstructure feature engineering (microprice, quoted spread, OBI)
  on top of the existing pipeline. Schema and feature functions are in
  `src/l2_store.py`; only the ingestion parser needs to be filled in
  once the UCLA data format is known.
- Real market impact model in the backtester instead of flat slippage.
- Cross-exchange validation (Coinbase, Kraken) to test whether the
  5 bps cost ceiling generalizes.
- 3-state HMM and HMM with different feature combinations.
- Block-bootstrap CIs on overlapping walk-forward splits.
- Multi-seed deep model runs for ML reproducibility.
