# Methods

Every methodological choice and why. Pair this with `docs/RESULTS.md` for
the findings.

## Data

### Source

Binance.US public REST API (`api.binance.us/api/v3/klines`). Hourly
candles, OHLCV. Quote currency: USDT only. Local storage as one parquet
file per symbol in `data/spot_1h/`.

The store is built by `store_data.py` which calls `sync_store_all` in
`src/data_store.py`. Subsequent syncs are incremental (last-bar-onward).
The current snapshot in the local store ends 2026-01-22.

### Universe construction (as-of-time, with first-bar leak fix)

`src/universe.py` `compute_universe_at_time(t0, min_history_days)`:

A symbol is in the universe at `t0` iff:
- `first_observation <= t0 - min_history_days` (was listed by t0 with
  the required history)
- `last_observation >= t0 - 1 step` (was still trading near t0)

The first-observation check is necessary to prevent look-ahead leakage
when `t0` is in the past. Without it, symbols listed AFTER t0 enter the
"as-of t0" universe whenever the local store contains data through some
future date. This was a real bug in the inherited code; fixed mid-project.

### Liquidity filter

`src/universe.py` `filter_top_n_by_liquidity`: rank symbols by mean
`close * volume` (USDT volume) over the training window only, take the
top N. Symbols with less than `min_coverage` (default 80%) of expected
bars in the window are dropped before ranking to prevent noise from
sparse symbols.

Default: top 50 by liquidity.

### Survivorship

The local store contains only currently-listed symbols. An external
delisting log (Binance.US Help Center announcements) shows 14 tokens
delisted between 2024-01-01 and 2026-01-22. After accounting for the
liquidity filter (most delistees were small-cap and would fail it
anyway), an estimated 3-6 candidates out of the ~50-pair candidate set
are missing from the panel. See `docs/survivorship_bias.md` and
`docs/binance_us_delistings.md`.

## Pair selection

### Strict cointegration screen (`src/pair_selection.py` `score_pairs`)

1. Pearson correlation prescreen on log returns; keep pairs with
   `corr >= 0.5`.
2. OLS hedge ratio on log prices; residual is the spread.
3. ADF stationarity test on the residual; keep pairs with
   `adf_pvalue <= 0.05`.
4. AR(1) half-life estimate; keep pairs with `half_life ∈ [4h, 240h]`.
5. Beta stability across 4 contiguous training segments; keep pairs
   with `std(β) / |mean(β)| <= 0.35`.
6. Composite score: z-weighted (ADF strength, half-life proximity to
   48h, spread vol, beta stability, correlation).

This screen finds zero passing pairs on the local Binance.US slice. See
finding 1 in RESULTS.md for why (static OLS hedge ratios fail OOS).

### Correlation fallback (`rank_pairs_by_correlation`)

When the strict screen returns no pairs, fall back to ranking the same
correlation prescreen output and taking the top N. These pairs are
explicitly NOT claimed to be cointegrated by the static screen; finding
1 in RESULTS.md establishes that they ARE cointegrated under a Kalman
dynamic hedge.

Default: top 20 pairs.

## Spread computation

Two modes, switched by `--use-kalman` in `scripts/run_first_branch.py`:

### Static (default)

`spread_t = log(close_a_t) - α - β * log(close_b_t)` with `(α, β)` from
a single OLS fit over the training window. Used in the first-pass
results; superseded by Kalman for all advisor-facing tables.

### Kalman dynamic hedge (`src/kalman_hedge.py`)

State-space model:

```
state_t = [α_t, β_t]
state_t = state_{t-1} + w_t,         w_t ~ N(0, Q)
y_t     = [1, x_t] @ state_t + v_t,  v_t ~ N(0, R)
```

Where `y_t = log(close_a_t)`, `x_t = log(close_b_t)`. Parameters
`(Q_α, Q_β, R)` are fitted by MLE on training data only (L-BFGS-B over
the Kalman innovations log-likelihood). The filter is then run forward
on the test slice with parameters held fixed and the trained final state
as the initial condition. Residuals on the test slice are the
out-of-sample spread used in finding 1.

Implementation: pure numpy, no `pykalman` dependency. ~50 lines for the
filter, ~30 for the MLE wrapper.

Forward-rolled per pair via `build_kalman_spread_overrides` in
`src/features.py`; the resulting series is plugged into the standard
feature pipeline as the `spread` column.

## Feature engineering (`src/features.py`)

Per-bar, per-pair features (16 total):

- Spread: `spread`, `spread_z_168h`, `spread_diff_1h/4h/24h`,
  `spread_vol_24h/168h`
- Volume: `volume_ratio`, `volume_z_a_168h`, `volume_z_b_168h`,
  `pair_volume_sum`, `pair_volume_gmean`
- Market context: `btc_return_24h`, `rolling_corr_24h`,
  `realized_vol_a_24h`, `realized_vol_b_24h`

Targets:
- `target_spread_change_24h` (next 24h spread change, regression target)
- `target_reversion_24h` (binary, will |spread| be smaller in 24h)
- `y_class` (3-class: revert / persist / diverge)

The 3-class label uses either a fixed threshold of 0.001 in log-spread
units (default) or a per-pair-z-scored threshold (with
`--per-pair-label`): `threshold = label_scale_factor *
std(|future_abs| - |current_abs|)` over the first half of each pair's
data. The per-pair option normalizes the label distribution across pairs
of different spread vol; see `docs/long_pipeline_results.md` and the
`pp` artifacts directory for the comparison.

## ML models (`src/modeling.py`)

All models train on the same dataset, same splits, same seed.

### Baselines

- `persist_class`: always predict class 1 (persist)
- `majority_class`: predict the modal class from training
- `random_stratified`: sample classes with training prior probabilities
- `zscore_rule`: class 0 (revert) if `|spread_z| >= 1.5`, else class 1
- `sklearn_hist_gradient_boosting`: 300 trees, lr=0.05, max_leaf_nodes=31

(`xgboost` is preferred when available; falls back to sklearn when not.
The fallback path is what was used for the headline results because the
local install of xgboost was missing OpenMP at runtime.)

### LSTM (`train_lstm`)

2-layer LSTM, hidden dim 64, dropout 0.2 between layers, linear head to
3 logits. Trained with Adam, lr=1e-3, weight decay 1e-5, batch size 256,
4 epochs per fit. Features standardized per-fit using training statistics.

### Transformer (`train_transformer`)

2-layer encoder, 2 heads, d_model=32, ff=64, dropout 0.2,
batch_first=True. CLS-token pooling. Linear head to 3 logits. Trained
with the same optimizer/schedule as LSTM. Parameter count is matched to
the LSTM at ~30k parameters.

### Big transformer (`src/modeling_big_transformer.py`)

Separate, larger architecture introduced after adversarial review flagged
the small transformer as undertrained: 4 layers, 4 heads, d_model=128,
ff=512, sinusoidal positional encoding, pre-norm, GELU. Trained with
AdamW, lr=5e-4, weight decay 1e-4, cosine LR with 1-epoch warmup,
gradient clipping at 1.0, validation-based early stopping (patience 5).
~3M parameters, ~100x larger than the small transformer.

Single-split sanity result confirmed the small transformer was capacity-
limited; walk-forward result pending at time of writing.

## Walk-forward evaluation (`scripts/run_walk_forward.py`)

Rolling time-ordered splits. Defaults:
- 90-day training window
- 30-day test window
- 30-day step

With t0 = 2024-01-01, train_days = 180, test_days = 750, this produces
27 walk-forward splits with 135,000 total test samples.

Each split:
1. Filter samples to `train_start <= timestamp < train_end` for training,
   `train_end <= timestamp < test_end` for test.
2. Cap training and test sizes via `--max-train-samples` and
   `--max-test-samples` (most recent first).
3. Train each model from scratch on training set.
4. Predict on test set.
5. Compute per-bar metrics: accuracy, macro F1, trade-count, total PnL
   (in log-spread units), pnl_mean_to_std, win_rate, max drawdown.

Outputs:
- `walk_forward_predictions.parquet`: per-row predictions across all
  models and splits.
- `walk_forward_metrics_by_split.csv`: one row per (model, split).
- `walk_forward_summary.csv`: per-model means across splits.

Bootstrap CIs are computed by `scripts/plot_walk_forward_summary.py`
using iid resampling over splits. **Caveat: splits share training data
(67% overlap between consecutive trains), so iid bootstrap overstates
precision.** A block bootstrap is the planned fix; the current
narrow-CI rankings are conservative if anything (real CIs would be wider).

## HMM regime filter (`src/hmm_filter.py`)

2-state Gaussian HMM on per-pair feature input
`[spread_z_168h, spread_diff_1h, spread_vol_24h]`. Fitted per pair on
the training slice (strictly before `split_train_end`), decoded via
Viterbi over the full pair history.

The mean-reverting state is identified post-fit as the state with the
lower std of `spread_diff_1h` (more contained moves). Predictions in the
non-mean-reverting state are suppressed to class 1 (flat) before
trading-metrics recomputation.

Library dependency: `hmmlearn`. Fits to numerical-instability convergence
warnings on roughly 20% of pair-split fits; reported as-is. A version
with multiple random starts and a convergence filter is on the future-work
list.

## Backtester (`src/backtest.py`)

### Position model

Long $L of A leg, short $L*β of B leg in dollar terms. Per-bar PnL:

```
pnl_t = held_position_{t-1} * L * (r_a_t - β * r_b_t)
```

where `r_a`, `r_b` are arithmetic returns. β is the static OLS hedge
ratio from pair selection (constant across the backtest); this matters
because the pair-selection β is what determines the contract size on
each leg, not the time-varying Kalman β (which is used for spread
computation only, not contract sizing).

### Signal-to-position

Two modes, switched by `--state-machine` in
`scripts/run_portfolio_backtest.py`:

**Bar-by-bar** (default, used for the original first-pass result):
desired position is `_class_to_signal(pred_class, spread_sign)`:
- class 0 (revert) → position = -spread_sign
- class 2 (diverge) → position = +spread_sign
- class 1 (persist) → position = 0

Every bar the position is reset to the desired position; every change
charges cost. Generates many flickering "trades" from noisy classifiers,
inflating cost drag.

**State machine** (`use_state_machine=True`, used for the cost
sensitivity in finding 4):
- Open when flat AND `pred_class == 0` AND `|spread_z| >= entry_z`;
  position = `-sign(spread_z)`.
- Close (go flat) when `|spread_z| <= exit_z` OR `pred_class == 2` OR
  spread sign flips relative to entry.
- Hold when `pred_class == 1`.

Default entry_z=2.0, exit_z=0.5. Cuts trade frequency by 5-10x vs
bar-by-bar.

### Cost model

Per-side cost in basis points of notional traded:
- `taker_fee_bps` (default 10)
- `slippage_bps` (default 5)

Total cost on a position change of `Δ` is:
```
notional_traded = |Δ| * leg_notional * (1 + |β|)
cost = notional_traded * (taker_fee_bps + slippage_bps) / 1e4
```

Round-trip cost per leg = 2 * (taker_fee_bps + slippage_bps). For
Binance.US defaults: 2 * 15 = 30 bps round-trip per round-trip, or
"15 bps round-trip per leg" in the simpler language used in the results.

### Sharpe denominator (mid-project fix)

Originally divided per-bar PnL by time-varying deployed capital
(`active_pairs * leg_notional * 2`), which could produce positive Sharpe
on negative total return when the active count varied a lot. Now divides
by a fixed capital base (`N_pairs * leg_notional * 2`). This is the
standard portfolio convention and is what makes the headline 5 bps cost
ceiling number reliable.

### Sanity tests (`tests/test_backtest_sanity.py`)

Six hand-computable tests:
1. Flat signal → zero PnL, zero trades, zero cost.
2. Constant prices with one open at t=1 → cost equals
   `L * (1 + |β|) * cost_bps / 1e4` exactly.
3. Known +1% return on leg A, leg B flat, held position +1 → exactly
   `L * 0.01 = $100`.
4. Random signal on random walk → Sharpe near zero (got 2.66 with
   n=2000, well within sampling tolerance).
5. State machine across a |z|≥entry block bounded by |z|≤exit → exactly
   2 transitions.
6. Deterministic per-bar return with known mean/std → annualized Sharpe
   matches `(μ/σ) * sqrt(bars_per_year)` within ~25% sampling
   tolerance for n=1000.

All six pass on the current code. The earlier Sharpe-denominator bug
would have failed several. Run with:

```bash
python3 tests/test_backtest_sanity.py
```

## L2 store scaffold (`src/l2_store.py`)

Phase-2 placeholder. Schema is locked:
- One file per symbol per UTC day: `data/l2/{SYMBOL}/{YYYY-MM-DD}.parquet`
- Columns: timestamp (UTC, nanosecond), seq (uint64), bid_px_1..K,
  bid_sz_1..K, ask_px_1..K, ask_sz_1..K, is_snapshot, is_delta
- Default K=10 levels

Feature stubs (microprice, quoted spread bps, order book imbalance) are
implemented and tested with synthetic input. The ingestion parsers
(`_parse_raw_websocket_dump`, `_parse_raw_csv_dump`) are
`NotImplementedError` placeholders that will be filled in when the UCLA
data sample arrives. Tests for the parsers and downstream feature math
will run on the synthetic dataset until then.

## Reproducibility

All commands to reproduce every number in `docs/RESULTS.md` are in the
top-level `README.md` under "Reproducibility". Outputs land in
`artifacts/` (gitignored).

Random seeds are set per-script (`--seed` flag; default 7) where
applicable. The deep model trainings use a single seed for the headline
results; multi-seed runs are on the future-work list.
