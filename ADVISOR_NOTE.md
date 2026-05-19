# Note to advisor

Snapshot of the MATH 199 stat-arb project. Phase 1: methodology + first-pass
results on hourly Binance.US spot. Phase 2 = L2 microstructure, blocked on
UCLA access. L2 storage schema is scaffolded in `src/l2_store.py`.

Read README first.

## Headlines

- Kalman dynamic hedge recovers cointegration that static OLS misses. OOS
  with MLE-fit Q on train, forward-rolled on test: 10/10 pairs at p<0.001,
  static OLS holds only 2/10 at p<0.05.
- 27-split walk-forward on post-audit Kalman pipeline: LSTM
  pnl_mean_to_std 0.376 [0.362, 0.388] > booster 0.356 [0.342, 0.368]
  > zscore 0.282 [0.274, 0.291] > transformer 0.018 [-0.025, 0.064].
  Per-trade win rate: zscore 0.96, LSTM 0.80, booster 0.74.
- Pre-cost Sharpe ~8 with proper entry/exit state machine. Break-even
  ~5 bps round-trip per leg. Binance.US taker (15 bps) destroys it.
- 2-state Gaussian HMM filter doesn't help. Generalizes to 3-state.
- Single chronological train/test splits over-state ML. Transformer's
  earlier #1 finish was a sampling fluke.

## Methodological cleanups (worth flagging)

- Universe construction was leaking post-t0 listings. Fixed (`first_ts <= t0 - min_history`).
- Backtester Sharpe denominator could give positive Sharpe on negative
  return when active-pair count varied. Fixed (fixed capital base).
- In-sample Kalman ADF was circular (filter whitening its own residuals).
  Replaced with proper train/test split.
- Per-pair label threshold was fit on first 50% of pair data, extending
  past test windows. Fixed (label_train_end = t0).
- Kalman spread had a parameter discontinuity at train_end (training
  used defaults, test used MLE). Fixed.
- `time_since_zero_crossing` confirmed as a soft leak via ablation; dead
  on the post-audit pipeline.

## What we have not done

- Single exchange, single quote currency, single bar interval
- Survivorship: 14 delistings in window, 6.7% upper / ~5% realistic
- No funding cost on synthetic shorts
- No market impact (flat slippage)
- t0 only at 2024-01-01
- 3-class label only (no regression head)
- HMM tried 2-state and 3-state, both negative

## Open questions

- Is the Kalman vs OLS OOS protocol defensible as the central finding?
- Want a systematic entry/exit threshold sweep for the cost ceiling?
- Should we rerun pair selection using the Kalman screen?
- L2 data: any specific format/schema you know we'll be getting?

## Tests

- `tests/test_backtest_sanity.py` (6 cases, all pass)
- `tests/test_leakage_audit.py` (10 cases, all pass)
