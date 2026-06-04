# Methods (Phase 2)

## Data

- Tardis daily CSVs over (exchange, symbol, data_type, day).
- Canonical schemas: see `src/l2_store.py` and `src/trade_store.py`.
- Layout: `data/l2/{symbol}/{YYYY-MM-DD}.parquet`,
  `data/trades/{symbol}/{YYYY-MM-DD}.parquet`.
- Ingestion seam: `src/tardis_ingest._tardis_*_to_canonical`. Update only
  these when the real CSV layout is in hand.

## Universe (as-of-time)

`src/universe.compute_universe_at_time(t0, min_history_days, require_both_feeds=True)`.
A symbol qualifies iff its local store has every day in
`[t0 - min_history_days, t0)` for both L2 and trades feeds. Liquidity
top-N is by mean daily notional volume from the trades feed over the
training window only.

## Bars

`src/bars.build_bars(book, trades, cfg)` at default `bar_size="1s"`.
- Book-state columns (microprice, midprice, quoted_spread_bps, obi_5) take
  the LAST value within the bar.
- Trade-flow columns sum within the bar.
- Trades are signed first (exchange aggressor side, then quote rule, then
  tick rule fallback), then bucketed by notional ($1k / $10k / $100k cuts),
  then aggregated per bucket per side per bar.

## Spread

Default: `spread = log(microprice_a) - alpha - beta * log(microprice_b)`,
with (alpha, beta) from OLS on the training window in
`src/pair_selection.score_pairs`.

Kalman override: `src/features.build_kalman_spread_overrides` fits MLE on
bars < train_end and forward-rolls bars >= train_end. Same parameters across
the boundary (audit invariant carried over from phase 1).

## Features (per pair, at bar cadence)

`src/features.FEATURE_COLUMNS`. Cadence-agnostic lookbacks expressed in
bars. At 1s default:
- spread, spread_z (1h lookback), spread_diff_{1, 60, 600 bars},
  spread_vol_short/long (60/600 bars).
- Book microstructure per leg: quoted_spread_bps, obi_5.
- Trade flow per leg: signed_notional, trade_notional.
- Institutional / retail signed flow per leg + buy imbalance ratio.
- The headline pre-reg target: `inst_buy_imbalance_over_leg`.

All columns are backward-rolling. Verified by spike-injection test
`test_pair_features_have_no_lookahead`.

## Labels (3-class)

Same convention as phase 1:
- 0 = revert (|future| - |current| < -tau)
- 2 = diverge (|future| - |current| > +tau)
- 1 = persist

Default `tau = 1e-5` log-microprice units. Per-pair mode anchors on
`label_train_end` to avoid leak past it (audit invariant).

## ML models

Same families as phase 1, transplanted with new input shapes:
- 4 baselines (persist / majority / random_stratified / zscore_rule)
- Booster (XGBoost or HistGradientBoosting fallback)
- LSTM (2-layer, hidden 64)
- Small transformer (2-layer, d_model=32, CLS pooling)
- Big transformer (4-layer, d_model=128, cosine LR, early stopping)

## Walk-forward

`scripts/run_walk_forward.py`. Default at 1s bars: 5d train / 1d test /
1d step. Per-split train-from-scratch.

## Cost model (book-walking impact)

`src/backtest.walk_book_fill(notional, side, book_row, levels)` walks the
contemporaneous book in the trade direction and returns VWAP. The
backtester computes slippage = (VWAP - mid) per leg per fill and adds the
taker fee. No flat bps assumption when book columns are present; falls
back to flat bps when they are not (used by unit tests).

## Statistical corrections

`src/statistics.py`:
- `hac_long_run_var` / `hac_sharpe` (Newey-West, Bartlett kernel)
- `deflated_sharpe` (Bailey & Lopez de Prado 2014). Pre-registration moves
  the effective N to 1, but DSR remains available for diagnostic comparison
  to phase 1.
- `block_bootstrap_ci` for walk-forward metric CIs.

## Reproducibility

All commands ship in `phase2_l2/scripts/`. Outputs land under
`phase2_l2/artifacts/` (gitignored). Random seeds default to 7 per script.
