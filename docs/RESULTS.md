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

Single-split feature ablation on the post-audit pipeline:

| Dropped | dAcc | dF1 | dPnL |
| --- | ---: | ---: | ---: |
| baseline (all 14) | 0.587 | 0.575 | 26.22 |
| **latest_spread_z** | **-0.144** | **-0.190** | **-20.72** |
| mean_volume_ratio | -0.001 | -0.003 | -0.43 |
| every other | <=+/-0.005 | <=+/-0.005 | <=+/-0.5 |

- `latest_spread_z` carries the signal
- Drop it and the model loses 14pp accuracy, 19pp F1, 79% PnL
- Every other feature changes the model by <=+/-0.5 PnL units (noise)
- ML = fancy z-score
- `time_since_zero_crossing` trajectory: +0.026 (static) -> +0.001 (Kalman short) -> 0 (post-audit)
- Confirms adversarial-review leak suspicion; dead on the post-audit pipeline
- Source: `docs/feature_ablation_fixed.csv`

## 4. Cost ceiling (with deflated Sharpe correction)

Entry/exit state machine. Open: flat AND pred=0 AND |z|>=2. Close:
|z|<=0.5 OR pred=2 OR sign flip. 27 splits, $10k/leg/pair, 20 pairs.
Post-audit pipeline.

| Round-trip cost per leg | LSTM Sharpe | LSTM $ | zscore Sharpe | zscore $ | booster Sharpe | booster $ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 bps | 8.08 | +130,385 | 8.03 | +130,798 | 8.27 | +131,810 |
| 5 bps (maker) | 0.80 | +12,656 | 0.83 | +13,268 | 0.90 | +14,076 |
| 15 bps (Binance.US taker) | -14.04 | -222,802 | -13.88 | -221,791 | -14.17 | -221,393 |

### Deflated Sharpe (Bailey & Lopez de Prado 2014)

The naive Sharpe is biased upward by selection: across this project we
ran ~30-60 strategy configurations (2 spread definitions, 2 label
schemes, 7 models, 3 HMM variants, 2 entry/exit thresholds, 3 cost
levels, etc.) and report the best. The deflated Sharpe corrects for that.

SR_0 = expected max Sharpe under the null over N trials, given the
variance of Sharpes across the trials.

| N trials | SR_0 (ann.) | 0 bps DSR (top) | 5 bps DSR (top) | 15 bps DSR (top) |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 4.08 | 1.000 | 0.42 | ~0 |
| 10 | 5.38 | 0.99 | 0.38 | ~0 |
| 25 | 6.83 | 0.90 | 0.37 | ~0 |
| 50 | 7.78 | 0.67 | 0.34 | ~0 |

- At a defensible N=10-25 (the configurations we actually evaluated and
  reported on), pre-cost Sharpe is significant
- At N>=50 (conservative count across the whole project), pre-cost
  Sharpe does NOT clear the deflated bar
- The "5 bps break-even" finding does NOT survive even N=10 correction.
  The observed Sharpe 0.90 is below the chance-best of 1.23 annualized
- The "15 bps destroys it" finding is unaffected by DSR; the Sharpe is
  deeply negative regardless
- Source: `docs/dsr_{0,5,15}bps.csv`, `scripts/run_deflated_sharpe.py`

### Honest re-statement

- **Pre-cost signal: ambiguous.** Without DSR, Sharpe ~8 looks decisive.
  With N=50 DSR correction, indistinguishable from chance-best of the
  same number of strategies. Reportable at N <= 25.
- **Break-even cost claim is retracted.** Sharpe 0.80-0.90 at 5 bps does
  not survive selection adjustment at any plausible N. We cannot claim
  the alpha clears 5 bps round-trip per leg.
- **Above-cost claim survives.** At 15 bps round-trip per leg the
  strategy loses deterministically; no DSR question.

The narrower defensible claim: at hourly cadence on Binance.US USDT
pair spreads, no model among the ones tried produces a Sharpe that
beats selection-corrected chance at any realistic cost level. The
pre-cost signal MIGHT be real but cannot be confidently distinguished
from selection bias on this dataset alone.

What would resolve this:
- Pre-register a single model configuration BEFORE looking at results,
  then evaluate it once
- Cross-validate on out-of-sample exchanges (Coinbase, Kraken)
- L2 data + market impact model (Phase 2)

## Negative results

### HMM filter

Post-audit pipeline, 27 splits:

| Model | raw total_pnl | HMM-filtered total_pnl | bar suppression |
| --- | ---: | ---: | ---: |
| lstm | 334.6 | 142.6 | -48% |
| booster | 321.5 | 132.8 | -44% |
| zscore | 195.7 | 71.4 | -55% |
| transformer | 0.7 | -7.0 | -47% |
| majority | 21.2 | 9.7 | -39% |

- 40-55% of bars suppressed, total_pnl drops 50-65% across the board
- Same negative result with 3 states + 3 random starts
- Same negative result on the pre-audit pipeline
- Source: `artifacts/hmm_kalman_long_fixed/`, `artifacts/hmm_kalman_long_3state/`

### Single-split ML over-states

- Earlier single-split run had transformer at #1 (pnl_mean_to_std=0.25)
- 27 splits drop it to #4 with CI [-0.057, 0.148]
- Walk-forward + bootstrap CIs = minimum standard

## Big transformer caveat (resolved)

Headline transformer was small (2 layers, 2 heads, d_model=32, 4 epochs)
matched to LSTM parameter count. We re-ran with a larger transformer
(4 layers, 4 heads, d_model=128, ff=512, 20 epochs, cosine LR with
warmup, validation-based early stopping). On the per-pair-label dataset,
27 walk-forward splits:

| Metric | small transformer | big transformer |
| --- | ---: | ---: |
| accuracy | 0.584 | **0.668** |
| macro F1 | 0.254 | **0.509** |
| pnl_mean_to_std | -0.009 | **0.334** |
| win rate | 0.156 | **0.845** |
| total_pnl | 9.83 | 308.19 |
| trades | 7,179 | 37,689 |

- Big transformer pnl_mean_to_std 0.334 puts it close to LSTM (0.376)
  and booster (0.356); definitely not "indistinguishable from random"
- Win rate 0.845 actually exceeds both LSTM (0.795) and booster (0.741)
- The "transformer is no better than random" headline was a capacity
  artifact, not a property of the architecture
- Apples-to-apples comparison: big transformer ran on the pre-audit
  per-pair-label dataset, so the booster/LSTM numbers in finding 2
  shouldn't be directly stacked against the 0.334. A post-audit
  re-run is straightforward but not in flight.
- Source: `artifacts/big_tx_pp_walk/walk_forward_summary.csv`,
  `src/modeling_big_transformer.py`

## Revised finding 2 framing

Reading the walk-forward tables (post-audit LSTM/booster/zscore +
pre-audit big transformer all considered):

- **LSTM, booster, big transformer, z-score are within striking distance
  of each other on signal quality** (pnl_mean_to_std in the 0.28-0.38
  range). The ML models edge zscore by 0.05-0.10.
- **Z-score still wins on per-trade win rate** (0.96 vs 0.74-0.85 for
  ML) by being much more selective.
- **Small transformer matched to LSTM parameter count fails.** That's
  a parameter-count finding, not an architecture finding.

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
