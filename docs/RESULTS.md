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

### Generalization to the full universe

Same protocol on all 1219 pairs in the liquidity top-50 universe:

| p threshold | static OLS passes | Kalman passes |
| --- | ---: | ---: |
| p < 0.001 | 0 (0.0%) | 1,174 (96.3%) |
| p < 0.005 | 1 (0.1%) | 1,188 (97.5%) |
| p < 0.010 | 10 (0.8%) | 1,198 (98.3%) |
| p < 0.050 | 66 (5.4%) | 1,215 (99.7%) |
| p < 0.100 | 142 (11.6%) | 1,217 (99.8%) |

- Static finds cointegration on 5.4% of pairs at p<0.05 OOS
- Kalman finds it on 99.7%
- Kalman OOS p-values reach 5e-10 for the strongest pairs
- The finding is not a property of the top-10 selected pairs; it holds
  across essentially the entire liquidity universe
- Source: `artifacts/kalman_screen/kalman_pair_screen.csv`

## 2. Walk-forward, post-audit Kalman pipeline

27 splits, deep models included. Block-bootstrap 5/95% CIs (block_size=3):

| Model | pnl_mean_to_std | 5/95% CI | win rate | win rate CI | trades/split |
| --- | ---: | ---: | ---: | ---: | ---: |
| **lstm** | **0.376** | [0.362, 0.388] | **0.795** | [0.768, 0.824] | 2,083 |
| booster | 0.356 | [0.342, 0.368] | 0.741 | [0.720, 0.760] | 2,355 |
| zscore | 0.282 | [0.274, 0.291] | 0.963 | [0.959, 0.966] | 588 |
| majority | 0.021 | [0.000, 0.063] | 0.028 | [0.000, 0.083] | 185 |
| transformer | 0.018 | [-0.025, 0.064] | 0.437 | [0.354, 0.519] | 691 |
| random | 0.001 | [-0.003, 0.005] | 0.502 | [0.500, 0.505] | 2,689 |
| persist | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 0 |

- LSTM is now the leader. LSTM CI [0.362, 0.388] and booster CI [0.342, 0.368]
  overlap slightly but LSTM win rate CI [0.768, 0.824] is strictly above
  booster CI [0.720, 0.760]. LSTM dominates on per-trade quality.
- Booster CI strictly above zscore CI on pnl_mean_to_std
- Zscore still wins on per-trade win rate at 4x fewer trades
- Transformer (small, param-matched to LSTM) CI includes zero
- Audit fixes monotonically improved ML quality:
  - LSTM pnl_mean_to_std: 0.279 (pre-audit) -> 0.376 (post-audit)
  - LSTM win rate: 0.603 -> 0.795
  - Booster pnl_mean_to_std: 0.307 -> 0.356
  - Booster win rate: 0.623 -> 0.741
  - Zscore unchanged (doesn't use Kalman features or per-pair labels)
- Source: `artifacts/walk_forward_kalman_long_fixed_deep/walk_forward_summary.csv`

HAC (Newey-West lag=24):

| Model | iid | HAC | inflation |
| --- | ---: | ---: | ---: |
| lstm | 0.376 | 0.197 | 1.92 |
| booster | 0.356 | 0.188 | 1.91 |
| zscore | 0.282 | 0.138 | 2.07 |
| majority | 0.021 | 0.012 | 1.70 |
| transformer | 0.018 | 0.012 | 1.29 |
| random | 0.001 | 0.001 | 0.99 |

- iid overstates by ~1.9x (23/24h target overlap)
- random inflation 1.00 (sanity)
- transformer inflation 1.29 is close to random; consistent with the
  transformer mostly producing uncorrelated noise predictions
- Source: `docs/hac_sharpe_fixed_deep.csv`

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
Post-audit pipeline.

| Round-trip cost per leg | LSTM Sharpe | LSTM $ | zscore Sharpe | zscore $ | booster Sharpe | booster $ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 bps | 8.08 | +130,385 | 8.03 | +130,798 | 8.27 | +131,810 |
| 5 bps (maker) | 0.80 | +12,656 | 0.83 | +13,268 | 0.90 | +14,076 |
| 15 bps (Binance.US taker) | -14.04 | -222,802 | -13.88 | -221,791 | -14.17 | -221,393 |

- Pre-cost Sharpe ~8 on all three top models, 62% per-trade win rate
- Break-even ~5 bps round-trip per leg (LSTM/booster/zscore all just positive)
- Binance.US taker (10 bps/side, 20 bps round-trip per leg): destroyed
- Binance.US maker (~5 bps round-trip per leg): at the boundary
- Backtester sanity-checked (6 hand-computable tests, all pass)
- Source: `artifacts/backtest_fixed_sm_{0,5,15}bps/portfolio_metrics.csv`

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
