# Notes (Phase 2, L2 rebuild)

## Pre-registered hypothesis (locked 2026-06-03, before any L2 data is in hand)

> **H1:** On Kalman-cointegrated USDT-perpetual pairs on Binance.com, net
> signed institutional-bucket buy volume on the over-leg (the leg currently
> expensive relative to its hedge) predicts mean reversion of the microprice
> spread within K seconds.
>
> Operationally: let `inst_buy_imbalance_over_leg(t)` be the rolling-mean
> over the last W bars of the per-bar institutional buy imbalance on the
> over-leg. The hypothesis predicts a negative correlation between
> `inst_buy_imbalance_over_leg(t)` and `target_spread_change(t, t+K)` for
> bars where `|spread_z(t)| >= 2`, evaluated on a single held-out tail of
> the data after the entire pipeline has been frozen.

**Parameters that are part of the pre-registration (cannot be tuned post-hoc):**
- Universe: top 10 Kalman-cointegrated pairs by OOS ADF p-value at p < 0.05
- Cadence: 1-second bars
- Window W (rolling mean): 60 bars (1 minute)
- Horizon K (target prediction): 60 bars (1 minute)
- Institutional bucket: trades with notional >= $100,000 USDT
- Entry filter: `|spread_z| >= 2` using `z_window_bars = 3600`
- Pre-registration date: 2026-06-03
- Pre-registration commit: (to fill in when frozen)

**The held-out tail (will not be touched until everything else is locked):**
- We will NOT define the tail until the data window is in hand, but we
  commit now: it will be the LAST 20% of the available time range, no
  cherry-picking by performance.

**Pass criterion:**
- One-sided p-value on the Spearman correlation between
  `inst_buy_imbalance_over_leg(t)` and `target_spread_change(t, t+K)` on the
  held-out tail, computed once, with HAC standard errors (Newey-West lag K).
- Pass = p < 0.05 in the predicted direction.

**Why this kills the DSR concern:** the phase 1 deflated Sharpe penalty was
brutal because we tried ~30-60 strategy configurations and reported the best.
By locking ONE hypothesis with ONE held-out evaluation, the DSR correction
becomes N=1 and the test is exactly what it claims to be.

## Open questions for the team / Mihai

- Confirm Tardis API key is live, or get a new one. Without it we are blocked
  past the synthetic-data validation stage.
- Get `data.py` reference script from the teammate. It is the authoritative
  source for the Tardis CSV column layout; we have written the parsing seam
  against Tardis's public docs but the real layout may differ in
  edge cases. The parsing seam is the only place that needs to change:
  `src/tardis_ingest._tardis_book_snapshot_25_to_canonical` and
  `_tardis_trades_to_canonical`.
- Confirm we can use Mihai's math server (or Hoffman2) for bulk ingest and
  compute. Local does not scale to the chosen universe size.
- Confirm the institutional bucket definition: $100k+ is a reasonable
  default for Binance perps but Mihai may have a sharper threshold tied to
  ADV percentile.

## What is NOT done in phase 2

- No real Tardis pulls yet (gated on API key + reference script).
- No bulk ingest yet.
- No real walk-forward run yet.
- No queue-position model in the backtester (book-walk impact only).
- No latency model.
- Multi-seed deep model runs.
- Funding cost on perps (we have the data type planned but it is not wired
  into PnL yet).

## What carried over from phase 1

See `docs/PROVENANCE.md`.

## What is NEW in phase 2

- Tardis ingestion + canonical L2 / trades stores.
- Microstructure features: microprice, OBI, quoted spread, signed flow,
  size buckets (retail / mid / large / institutional), VPIN, sweep
  detection.
- 1-second bar aggregation joining L2 state + trade flow.
- Microprice-based pair selection and Kalman screen.
- Book-walking impact backtester (no flat bps cost assumption).
- 8 leakage tests adapted for the L2 pipeline.
- 6 backtester sanity tests rewritten for the new cost model.
