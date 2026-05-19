# Notes

Running scratchpad of questions, concerns, open threads, and things that
should land in the discussion section but haven't yet.

## Open questions for the advisor

- Is the Kalman vs OLS OOS protocol defensible as the central finding?
  MLE on train, forward-roll on test, parameters held fixed. We think
  yes; want a sanity check from you.
- The deflated Sharpe correction (Bailey & Lopez de Prado 2014) is
  brutal on our finding 4 (5 bps cost ceiling). At N=10 trials the
  5 bps Sharpe sits below chance-best; at N=50 even the pre-cost
  Sharpe doesn't clear. We've retracted the break-even claim. Is this
  the right framing for the paper, or are we being too conservative?
- Pair selection is currently correlation-fallback for the downstream
  pipeline. Should we rerun on the Kalman-screened pairs (we have a
  1219-pair OOS screen ready)?
- The 1219-pair Kalman screen result (99.7% cointegrated OOS) is the
  strongest single finding. Worth making it the lead in the paper?
- L2 data: any format/schema you know we'll get from Tardis?

## Concerns

- Selection bias is real and probably bigger than our N=50 estimate.
  Honestly we tried more like 30-60 configurations across spread
  definitions, label schemes, models, HMM variants, entry/exit
  thresholds, cost levels, hyperparams. We're not pre-registered.
- 27 walk-forward splits is a lot more than 2 but the splits share
  overlapping training windows (67% at 90d train / 30d step). Block
  bootstrap basically matched iid; we may be underestimating CIs.
- pnl_mean_to_std isn't a real Sharpe. HAC correction at lag 24
  shrinks it ~2x. The block-bootstrap CIs are on the HAC-inflated
  number. Should be reporting the HAC-corrected number in the headline.
- Backtester treats short leg as costless; spot USDT shorting on
  Binance.US isn't free in practice.
- Flat slippage, no market impact. The L2 phase will resolve this.
- Single exchange, single quote currency, single bar interval. Cross-
  exchange validation is the cleanest way to address selection bias.

## What we have NOT done (proposal scope that's punted)

- Pre-register a single strategy and evaluate once. Would resolve the
  DSR concern.
- Cross-exchange validation (Coinbase, Kraken).
- Funding-cost model on synthetic short legs.
- Market impact in backtester (pending L2).
- t0 sensitivity (all results at t0 = 2024-01-01).
- Multi-seed deep model runs.
- Regression head instead of 3-class label.
- HMM with non-Gaussian emissions or different feature combinations.

## Phase 2 (L2)

- Tardis L2 ingestion → `src/l2_store.py` (schema + feature stubs ready,
  ingestion parsers are NotImplementedError until format known)
- Microprice, quoted spread bps, order book imbalance features
- Real market impact model in the backtester
- Pre-registration: pick one model config based on phase-1 work, evaluate
  once on L2 features, no further tuning. That's the only honest way
  to defang the selection-bias concern.

## Methodological cleanups done during the project (worth flagging)

- Universe construction was leaking post-t0 listings. Fixed
  (`first_ts <= t0 - min_history`).
- Backtester Sharpe denominator could give positive Sharpe on negative
  return when active-pair count varied. Fixed (fixed capital base).
- In-sample Kalman ADF was circular (filter whitening its own
  residuals). Replaced with proper train/test split.
- Per-pair label threshold was fit on first 50% of pair data, extending
  past test windows. Fixed (`label_train_end` = t0).
- Kalman spread had a parameter discontinuity at train_end (training
  used defaults, test used MLE). Fixed.
- `time_since_zero_crossing` confirmed as a soft leak via ablation;
  dead on the post-audit pipeline.
- Iid bootstrap on overlapping splits replaced by block bootstrap
  (numbers essentially unchanged).
- `pnl_mean_to_std` replaced with HAC-corrected Newey-West lag-24
  (~1.7x inflation noted).
- Deflated Sharpe (Bailey & Lopez de Prado) added; finding 4 retracted
  at 5 bps under any plausible N.

## Tests

- `tests/test_backtest_sanity.py` (6 cases, all pass)
- `tests/test_leakage_audit.py` (10 cases, all pass)
