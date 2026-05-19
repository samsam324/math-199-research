# Note to Advisor

This is a snapshot of the MATH 199 statistical-arbitrage project (Jack
Lutz + Sammy Salameh + collaborator Frank Kronewitter) representing
**Phase 1: methodology and first-pass results on hourly Binance.US spot
data**.

Phase 2 will deepen the analysis with L2 order-book data once UCLA
access comes through. Until then, the L2 storage schema and feature
stubs are scaffolded (`src/l2_store.py`) and ready for the data sample.

We wanted you to see the methodology and first-pass numbers now rather
than wait for Phase 2.

## Suggested reading order

1. **README.md** — orientation, status, headline findings, repository
   layout, reproducibility commands.
2. **docs/RESULTS.md** — the four headline findings with tables and
   bootstrap CIs.
3. **docs/METHODS.md** — every methodological choice with the reason.
4. **proposal.md** — the original project proposal (the "Remaining Work"
   section there is now mostly done; RESULTS supersedes it).

## What we are confident in

- **The Kalman dynamic hedge ratio recovers cointegration in crypto
  pairs that static OLS misses.** Out-of-sample, with parameters
  MLE-fit on training only and forward-rolled on a 30-day held-out test
  slice: 10 of 10 pairs hold p < 0.001 ADF on the OOS residuals; static
  OLS holds p < 0.05 on only 2 of 10. The early in-sample version of
  this comparison was unsound (the filter was effectively whitening its
  own residuals); the OOS protocol resolves that concern.
- **Across 27 walk-forward splits, ML provides no meaningful edge over
  a classical z-score rule** once the Kalman spread is used. Bootstrap
  CIs on `pnl_mean_to_std` are tight enough to assert this: boost
  [0.291, 0.323] vs z-score [0.272, 0.292] vs LSTM [0.258, 0.297]. The
  z-score rule reaches 96% per-trade win rate by being much more
  selective.
- **The cost ceiling on hourly USDT pair-spread alpha is approximately
  5 bps round-trip per leg.** Pre-cost Sharpe ~8 with a proper entry/exit
  state machine; maker fees (~5 bps round-trip) leave it barely positive;
  Binance.US default taker fees (~15-20 bps round-trip) eliminate it.
  This is the market-efficiency-bound claim of the paper.
- The 2-state Gaussian HMM regime filter as configured does NOT help on
  either spread definition. Reported as a negative result; might be a
  3-state HMM is needed, or different features.

## Methodological cleanups done during the project

Listed because they may be relevant to your reading:

- **Look-ahead leak in universe construction**: `compute_universe_at_time`
  was including symbols whose first bar was AFTER `t0`, leaking
  post-t0 listings into the as-of universe whenever the local store had
  data through some later date. Fixed; now requires first-bar before
  t0 with a configurable minimum history.
- **Sharpe denominator bug in backtester**: was dividing per-bar PnL by
  time-varying deployed capital, which could produce positive Sharpe
  on negative total return. Fixed; uses a fixed capital base now.
- **In-sample Kalman ADF**: original version ran the filter forward over
  the full window and tested its own innovations — circular by
  construction (the filter is built to whiten). Replaced with proper
  train/test split + MLE-fit Q + forward-roll, where parameters are
  held fixed across the test slice.
- **`time_since_zero_crossing` suspected leak**: was the top feature on
  static spreads at +0.026 importance, suggesting it was picking up
  trending non-stationarity rather than mean reversion. Now formally
  ablated: drop it from the feature set, model improves slightly. Dead
  feature on Kalman spreads, removed from discussion.

## What we know we have not done

Explicit in `docs/RESULTS.md` ("Limitations") and `docs/METHODS.md`:

- **Single exchange.** All findings are Binance.US-specific.
- **Single quote currency.** USDT only.
- **Single interval.** Hourly bars; lower-frequency analysis may differ.
- **Survivorship bound is approximate** but is quantified at 6.7% upper
  bound from the public Binance.US delisting log (14 tokens delisted in
  the window, ~3-6 candidates in the liquidity-filtered universe).
- **Funding cost on synthetic short legs is not modeled**, only fees and
  slippage.
- **The walk-forward bootstrap is iid over overlapping splits**, which
  modestly overstates precision. A block bootstrap is on the list.
- **A larger / longer-trained transformer is currently running in the
  background.** Single-split result on the per-pair-label dataset
  jumped accuracy from 0.58 to 0.72 and per-trade win rate from 16% to
  91%, suggesting the small transformer in the headline tables was
  capacity-limited. The walk-forward rerun is the rigorous test of
  whether that recovery is real across the 27 splits; if it is, the
  "transformer is no better than random" line in finding 2 should be
  qualified as "the small transformer matched to LSTM capacity is no
  better than random."

## Questions we have for you

- **Is the Kalman-vs-OLS comparison protocol defensible as the central
  methodological finding?** We believe yes (MLE on train, forward-roll
  on test, fixed parameters); want a check from you.
- **The 5 bps cost ceiling depends on the specific entry/exit thresholds
  (z=2 enter, z=0.5 exit)**. We tested entry z = 2.5 and the ceiling is
  similar but Sharpe shape changes. Would you want a more systematic
  threshold sweep?
- **Pair selection is correlation-fallback**. We could rerun pair selection
  using the Kalman cointegration screen (since finding 1 says it works
  OOS) to make the methodology internally consistent. We have not yet;
  worth doing for Phase 2.
- **L2 integration plan**: the storage schema in `src/l2_store.py` is
  designed to handle a Binance-style WebSocket capture OR a vendor CSV
  dump. Anything else we should plan for given what UCLA is providing?

Thank you for the time on this. We will keep the L2 ingestion ready so
we can move quickly when access lands.
