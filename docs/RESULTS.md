# Results

Long Kalman pipeline. 27 walk-forward splits. 424k samples. 20 top-liquidity
USDT pairs. t0 = 2024-01-01. 180d train, 750d test. Walk-forward 90/30/30.

## 1. Kalman vs static OLS out-of-sample

Per pair: 90d train, 30d held-out test. Static OLS on train, residuals on
test. Kalman MLE-fits Q on train, forward-rolls on test using trained final
state. No re-fitting on test.

| Pair | static OOS p | Kalman OOS p | Kalman q_beta |
| --- | ---: | ---: | ---: |
| DOGEUSDT_SUIUSDT | 0.923 | 1.5e-7 | 7.4e-6 |
| SOLUSDT_DOGEUSDT | 0.985 | 2.2e-7 | 4.6e-6 |
| XRPUSDT_SOLUSDT | 0.373 | 3.0e-7 | 6.8e-8 |
| XRPUSDT_DOGEUSDT | 0.092 | 2.5e-6 | 3.5e-6 |
| ADAUSDT_HBARUSDT | 0.094 | 7.3e-6 | 3.9e-6 |
| SOLUSDT_ADAUSDT | 0.823 | 2.9e-5 | 2.5e-5 |
| ETHUSDT_ADAUSDT | 0.411 | 3.4e-5 | 2.9e-5 |
| XRPUSDT_ADAUSDT | 0.011 | 1.5e-4 | 2.6e-5 |
| XRPUSDT_ETHUSDT | 0.029 | 2.1e-4 | 2.8e-5 |
| ETHUSDT_SOLUSDT | 0.013 | 8.0e-4 | 2.0e-5 |

- Static OLS: 2/10 at p<0.05 OOS
- Kalman: 10/10 at p<0.001 OOS, worst 8e-4
- MLE q_beta in 1e-8 to 3e-5 (slow drift, not whitening)
- Relationships real but non-stationary
- Source: `artifacts/kalman_oos/kalman_oos_comparison.csv`

## 2. Walk-forward, post-audit Kalman pipeline

27 splits, block-bootstrap 5/95% CIs (block_size=3):

| Model | pnl_mean_to_std | 5/95% CI | win rate | win rate CI | trades/split |
| --- | ---: | ---: | ---: | ---: | ---: |
| booster | 0.356 | [0.342, 0.368] | 0.74 | [0.72, 0.76] | 2,355 |
| zscore | 0.282 | [0.274, 0.291] | 0.96 | [0.959, 0.967] | 588 |
| majority | 0.021 | [0.000, 0.063] | 0.03 | [0.00, 0.083] | 185 |
| random | 0.001 | [-0.003, 0.005] | 0.50 | [0.500, 0.505] | 2,689 |
| persist | 0.000 | [0.000, 0.000] | 0.00 | [0.00, 0.00] | 0 |

- Booster CI strictly above zscore CI
- Zscore: 4x fewer trades, 96% per-trade win rate
- LSTM and transformer pending (deep walk-forward running on fixed dataset)

HAC (Newey-West lag=24):

| Model | iid | HAC | inflation |
| --- | ---: | ---: | ---: |
| booster | 0.356 | 0.188 | 1.91 |
| zscore | 0.282 | 0.138 | 2.07 |
| majority | 0.021 | 0.012 | 1.70 |
| random | 0.001 | 0.001 | 0.99 |

- iid overstates by ~1.9x (23/24h target overlap)
- Random inflation 1.00 (sanity)
- Ordering unchanged

## 3. ML signal is one feature

Single-split feature ablation:

| Dropped | dAcc | dF1 | dPnL |
| --- | ---: | ---: | ---: |
| baseline | 0.597 | 0.512 | 26.19 |
| latest_spread_z | -0.183 | -0.151 | -23.07 |
| time_since_zero_crossing | +0.002 | +0.006 | -0.21 |
| every other | <=+/-0.006 | <=+/-0.013 | <=+/-0.5 |

- `latest_spread_z` carries the signal
- ML = fancy z-score
- `time_since_zero_crossing` trajectory: +0.026 (static) -> +0.001 (Kalman short) -> 0 (Kalman long)
- Confirms adversarial-review leak suspicion
- Source: `docs/feature_ablation_kalman_long.csv`

## 4. Cost ceiling

Entry/exit state machine. Open: flat AND pred=0 AND |z|>=2. Close:
|z|<=0.5 OR pred=2 OR sign flip. 27 splits, $10k/leg/pair, 20 pairs.

| Round-trip cost per leg | LSTM Sharpe | LSTM $ | zscore Sharpe | zscore $ |
| --- | ---: | ---: | ---: | ---: |
| 0 bps | 8.06 | +125,340 | 7.83 | +125,116 |
| 5 bps (maker) | 0.69 | +10,604 | 0.74 | +11,654 |
| 15 bps (Binance.US taker) | -14.25 | -218,870 | -13.73 | -215,270 |

- Pre-cost Sharpe ~8, 61% per-trade win rate
- Break-even ~5 bps round-trip per leg
- Binance.US taker: destroyed
- Binance.US maker: at the boundary
- Backtester sanity-checked (6 hand-computable tests, all pass)
- Source: `artifacts/backtest_sm_{0,5,15}bps/portfolio_metrics.csv`

## Negative results

### HMM filter

| Model | raw total_pnl | HMM-filtered |
| --- | ---: | ---: |
| booster | 352.67 | 149.12 |
| LSTM | 332.93 | 135.78 |
| zscore | 206.07 | 69.16 |
| transformer | 104.25 | 6.03 |

- 30% bars suppressed, 30-60% PnL gone
- Same negative result with 3 states + 3 random starts
- Source: `artifacts/hmm_kalman_long/`, `artifacts/hmm_kalman_long_3state/`

### Single-split ML over-states

- Earlier single-split run had transformer at #1 (pnl_mean_to_std=0.25)
- 27 splits drop it to #4 with CI [-0.057, 0.148]
- Walk-forward + bootstrap CIs = minimum standard

## Big transformer caveat

Headline transformer was small (2 layers, 2 heads, d_model=32, 4 epochs)
matched to LSTM param count. Larger transformer (4 layers, 4 heads,
d_model=128, 20 epochs, cosine LR, val early stopping) on per-pair-label
single split:

| Model | acc | win rate | pnl_mean_to_std |
| --- | ---: | ---: | ---: |
| small | 0.584 | 0.156 | -0.009 |
| big | 0.715 | 0.913 | 0.380 |

- Walk-forward 27-split rerun in progress
- If confirmed: small transformer is undertrained, big version competitive
- If not: original finding stands

## Limitations

- Single exchange (Binance.US)
- Single quote currency (USDT)
- Single interval (hourly)
- Survivorship: 6.7% upper, ~5% realistic (see `binance_us_delistings.md`)
- No funding cost on synthetic shorts
- Flat slippage, no market impact
- t0 only at 2024-01-01 (sensitivity pending)
- 3-class label only (regression head pending)
- Walk-forward bootstrap iid over overlapping splits (block ~identical)

## Phase 2 (L2)

- Microprice, quoted spread, OBI
- Market impact in backtester
- Cross-exchange validation
- 3-state HMM on different feature combos
- Multi-seed deep model runs

## Audit

- 2 leakage bugs found and fixed (Kalman boundary, per-pair label)
- 10 leakage tests pass (`tests/test_leakage_audit.py`)
- 6 backtester sanity tests pass (`tests/test_backtest_sanity.py`)
- Finding 1 verified unchanged on fixed pipeline
- Findings 2-4 re-running, booster numbers already improved (0.307 -> 0.356)
- See `docs/leakage_audit.md`
