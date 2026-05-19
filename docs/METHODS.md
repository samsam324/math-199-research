# Methods

Every methodological choice and why. Pair with `docs/RESULTS.md`.

## Data

- Binance.US REST `api.binance.us/api/v3/klines`. Hourly OHLCV. USDT quote only.
- One parquet per symbol in `data/spot_1h/`. Built by `store_data.py` ->
  `sync_store_all` in `src/data_store.py`. Incremental on re-run.
- Snapshot ends 2026-01-22.

## Universe (as-of-time)

`src/universe.py` `compute_universe_at_time(t0, min_history_days)`. A symbol
qualifies iff both:

- `first_obs <= t0 - min_history_days`
- `last_obs >= t0 - 1 step`

The first-obs check is required; without it, symbols listed AFTER t0 leak
into the universe whenever the store has data through some future date.
This was a real bug, fixed mid-project. Regression test in
`test_universe_excludes_post_t0_listings`.

## Liquidity filter

`filter_top_n_by_liquidity`: rank by mean `close * volume` over the
training window only. Drop symbols below `min_coverage` (default 80%).
Top 50 by default.

## Survivorship

Local store has only currently-listed symbols. External Binance.US
delisting log: 14 delistings in window 2024-01-01 to 2026-01-22.
After liquidity filter: 3-6 candidates missing. 6.7% upper bound,
~5% realistic. See `binance_us_delistings.md`.

## Pair selection

`src/pair_selection.py`:

- `score_pairs`: corr prescreen (>=0.5) -> OLS hedge -> ADF on residual
  (p<=0.05) -> AR(1) half-life [4h, 240h] -> beta stability across 4
  segments -> composite score
- Finds 0 passing pairs on the local slice. Finding 1 in RESULTS shows why
  (static OLS is wrong model).
- `rank_pairs_by_correlation`: fallback when strict screen returns none.
  Top N by Pearson corr of log returns. Not claimed to be cointegrated by
  the static screen.

## Spread

Two modes via `--use-kalman`:

- Static (default): `spread_t = log(p_a) - alpha - beta * log(p_b)`,
  (alpha, beta) fit on training window via OLS in `score_pairs`.
- Kalman: see below.

## Kalman dynamic hedge

`src/kalman_hedge.py`. 2-state random-walk state-space model:

```
[alpha_t, beta_t] = [alpha_{t-1}, beta_{t-1}] + w_t,  w_t ~ N(0, Q)
y_t = [1, x_t] @ state_t + v_t,                       v_t ~ N(0, R)
```

- `fit_kalman_mle(y_train, x_train)`: L-BFGS-B over Kalman innovations
  log-likelihood. Returns Q, R, final state, AND train_residuals from the
  fitted-parameter forward pass.
- `kalman_forward_residuals(y_test, x_test, fitted)`: forward-roll on test
  with parameters held fixed and trained final state as initial condition.
  No re-fitting on test.
- Pure numpy, no `pykalman` dependency.

`build_kalman_spread_overrides(close, pairs, train_end)` in
`src/features.py`: per pair, fit MLE on data with `t < train_end`, build
combined spread series = fitted train_residuals ++ forward_residuals(test).
Same parameters on both sides of the boundary (caught in audit).

## Features (16 cols)

`src/features.py`:

- spread, spread_z_168h, spread_diff_{1h,4h,24h}, spread_vol_{24h,168h}
- volume_ratio, volume_z_{a,b}_168h, pair_volume_sum, pair_volume_gmean
- btc_return_24h
- rolling_corr_24h, realized_vol_{a,b}_24h

All backward-rolling. Verified by spike-injection test
(`test_features_have_no_lookahead`).

Targets:
- target_spread_change_24h = spread.shift(-24) - spread
- target_reversion_24h = `|future_spread| < |current_spread|`
- y_class = revert (0) / persist (1) / diverge (2) by threshold

## Labels (3-class)

`src/ml_dataset.py` `_class_label(current_abs, future_abs, threshold)`:

- 0 if `(future_abs - current_abs) < -threshold`
- 2 if `(future_abs - current_abs) > +threshold`
- 1 otherwise

Two threshold modes:

- Fixed: `classification_threshold = 0.001` in log-spread units (default)
- Per-pair: threshold = `label_scale_factor * std(|future_abs - current_abs|)`
  on each pair's data with `timestamp < label_train_end`. When
  `label_train_end` is set, no leakage past it (verified by
  `test_per_pair_label_threshold_no_leakage`). Earlier version used a
  fractional cutoff which leaked into test windows; fixed in audit.

## ML models (`src/modeling.py`)

Baselines:
- persist_class: always predict 1
- majority_class: predict training mode
- random_stratified: sample classes with training prior
- zscore_rule: class 0 if `|spread_z| >= 1.5`, else class 1

Tabular:
- sklearn_hist_gradient_boosting: 300 trees, lr=0.05, max_leaf_nodes=31
- xgboost preferred when available; sklearn fallback on Mac (libomp issue)

Sequence (small, param-matched to ~30k):
- LSTM: 2 layers, hidden 64, dropout 0.2
- transformer: 2 layers, 2 heads, d_model 32, ff 64, CLS pooling
- Adam, lr 1e-3, wd 1e-5, batch 256, 4 epochs
- Features standardized per-fit using x_train mu/sd only

Large (3M, separate file `src/modeling_big_transformer.py`):
- 4 layers, 4 heads, d_model 128, ff 512, GELU, pre-norm
- AdamW, lr 5e-4, wd 1e-4, cosine LR with 1-epoch warmup, grad clip 1.0
- Validation slice = last 10% of training, time-ordered
- Early stopping patience 5 on val loss

## Walk-forward (`scripts/run_walk_forward.py`)

- 90d train, 30d test, 30d step
- Long Kalman pipeline t0=2024-01-01 produces 27 splits / 135k test samples
- Per-split: filter samples by timestamp, cap via `--max-train-samples` /
  `--max-test-samples` (most recent first), train from scratch, predict on test
- Outputs: predictions.parquet, metrics_by_split.csv, summary.csv
- CIs via block bootstrap (block_size=3 for 90d/30d step overlap)

## HMM (`src/hmm_filter.py`)

- 2-state Gaussian on `[spread_z_168h, spread_diff_1h, spread_vol_24h]`
- Fit per pair on training slice (timestamps < split_train_end)
- Viterbi decode full pair history
- Mean-reverting state = state with lower spread_diff std
- Filter: suppress non-mean-reverting bars to class 1 (flat)
- `n_init` multi-start with convergence-preferred selection
- Library: `hmmlearn`. Some fits don't converge; reported as-is.

## Backtester (`src/backtest.py`)

Position:
- long $L of A, short $L*beta of B (dollar-neutral by construction)
- per-bar pnl: `held_position_{t-1} * L * (r_a_t - beta * r_b_t)`
- beta = static OLS hedge from pair selection (contract sizing, not spread)

Signal -> position:
- bar-by-bar (default): pred 0 -> -spread_sign, pred 2 -> +spread_sign, pred 1 -> 0
- state machine: open when flat AND pred=0 AND |spread_z|>=entry_z; close
  when |spread_z|<=exit_z OR pred=2 OR spread sign flips. Hold on pred=1.

Cost:
- taker_fee_bps (default 10) + slippage_bps (default 5) per side
- cost = `|delta_position| * leg_notional * (1 + |beta|) * total_bps / 1e4`

Sharpe:
- bar return = pnl / fixed capital base `N_pairs * leg_notional * 2`
- annualized = `mean / std * sqrt(24 * 365)`
- Fixed capital base required; time-varying deployed denominator could
  give positive Sharpe on negative return. Fixed in audit.

Sanity: 6 hand-computable tests in `tests/test_backtest_sanity.py`.

## L2 (Phase 2 scaffold)

`src/l2_store.py`:

- One file per symbol per UTC day: `data/l2/{SYMBOL}/{YYYY-MM-DD}.parquet`
- Schema: timestamp (UTC, ns), seq (uint64), {bid,ask}_{px,sz}_{1..10},
  is_snapshot, is_delta
- Feature stubs: microprice, quoted_spread_bps, order_book_imbalance
- Ingestion = `NotImplementedError` until UCLA sample arrives

## Reproducibility

Random seeds set per-script (`--seed` flag; default 7). All commands in
`README.md`. Outputs in `artifacts/` (gitignored). Deep models use a
single seed for the headline runs; multi-seed pending.
