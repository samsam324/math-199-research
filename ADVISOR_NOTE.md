# Note to Advisor

Snapshot of the MATH 199 stat-arb project (Jack Lutz, Sammy Adham,
Frank Kronewitter). Phase 1: methodology + first-pass results on hourly
Binance.US spot. Phase 2 deepens with L2 order-book data once UCLA access
lands. The L2 storage schema is scaffolded in `src/l2_store.py`.

## Reading order

1. `README.md` - orientation, status, headline findings, reproducibility
2. `docs/RESULTS.md` - four headline findings with tables and CIs
3. `docs/METHODS.md` - every methodological choice with the reason
4. `docs/leakage_audit.md` - paranoia audit, what was found and fixed
5. `proposal.md` - original proposal

## What we are confident in

- Kalman dynamic hedge ratio recovers cointegration that static OLS misses.
  Out-of-sample, MLE-fit on training only, forward-rolled on test: 10 of 10
  pairs hold p < 0.001 ADF; static OLS holds p < 0.05 on only 2 of 10.
- Across 27 walk-forward splits on the post-audit pipeline: booster
  pnl_mean_to_std 0.356 (CI [0.342, 0.368]) strictly above z-score
  0.282 (CI [0.274, 0.291]); z-score wins on per-trade win rate (96% vs
  74%) by being 4x more selective.
- Pre-cost Sharpe ~8 with a proper entry/exit state machine. Break-even
  is ~5 bps round-trip per leg. Binance.US taker (~15 bps round-trip per
  leg) destroys it. This is the market-efficiency bound the paper makes.
- 2-state Gaussian HMM regime filter does NOT help on either static or
  Kalman spreads. Generalizes to 3-state + 3 random starts.
- Single chronological train/test splits dramatically over-state ML.
  Earlier transformer #1 finish was a sampling artifact.

## Methodological cleanups during the project (worth flagging)

- Universe construction: `compute_universe_at_time` was including symbols
  whose first bar was AFTER t0, a real look-ahead when t0 is in the past.
  Fixed; requires first_ts <= t0 - min_history_days now.
- Backtester Sharpe denominator: was dividing per-bar PnL by time-varying
  deployed capital, which could produce positive Sharpe on negative total
  return. Fixed; uses fixed capital base.
- In-sample Kalman ADF: original version ran the filter forward over the
  full window and tested its own innovations (circular). Replaced with
  train/test split + MLE-fit Q + forward-roll on test.
- Per-pair label threshold: was fit on first 50% of pair data, extending
  past walk-forward test windows. Fixed via explicit label_train_end
  anchored to t0.
- Kalman spread parameter discontinuity at train_end: training period
  used default KalmanConfig, test period used MLE-fit. Fixed; both halves
  now use the same fitted params.
- `time_since_zero_crossing`: suspected leak in adversarial review,
  confirmed by feature ablation (top feature on static spreads, dead on
  Kalman + more data).

## Open questions for you

- Is the Kalman-vs-OLS OOS protocol defensible as the central finding?
  (MLE on train, forward-roll on test, parameters held fixed.)
- The 5 bps cost ceiling depends on entry/exit thresholds (z=2 in, z=0.5
  out). We checked entry z=2.5 too. Want a systematic threshold sweep?
- Pair selection is correlation-fallback. We could rerun selection using
  the Kalman cointegration screen (1225-pair sweep is running now).
- L2 data: any specific format or schema you know we'll be getting?

## What we have not done

- Single exchange (Binance.US only)
- Single quote currency (USDT)
- Single bar interval (hourly)
- Survivorship quantified at 6.7% upper, ~5% realistic after liquidity
  filter; not zero
- Funding cost on synthetic shorts not modeled
- Flat slippage, no market impact
- t0 sensitivity not exhaustively tested
- 3-class label is a heuristic; regression head pending
- HMM tried 2-state and 3-state, both negative
- pnl_mean_to_std reported with HAC correction (overlapping samples);
  iid version overstated by ~1.7x

## Reproducibility

Every number in RESULTS.md comes from a script in this repo. Full command
list in README.md. Two test files prevent regression:

- `tests/test_backtest_sanity.py` - 6 hand-computable backtester checks
- `tests/test_leakage_audit.py` - 10 leakage checks (universe, features,
  targets, labels, Kalman boundary, standardization, backtester lag,
  walk-forward ordering, cross-pair isolation)

Both pass on the current code. Run via `python3 tests/<file>`.

Thank you for the time.
