# MATH 199 — Statistical Arbitrage in Crypto Pair Spreads

UCLA Math 199 research project: do crypto pair-spreads exhibit measurable
mean reversion, and can it be characterized rigorously enough to draw
inferences about market efficiency? Authors: Jack Lutz, Sammy Adham,
Frank Kronewitter.

The proposal (`proposal.md`) follows Gatev, Goetzmann, and Rouwenhorst
(2006) on equity pairs trading, adapted to crypto. This README is the
entry point an advisor should read first; detailed results in
`docs/RESULTS.md` and methodology in `docs/METHODS.md`.

## Status

**Phase 1 (this branch): methodology + first-pass results on hourly
Binance.US spot.** Complete. Findings below.

**Phase 2: deeper microstructure analysis with L2 order-book data.**
Blocked on UCLA data access. Storage schema and feature stubs are
scaffolded in `src/l2_store.py`; ingestion parsers are placeholders
pending the data sample.

## Headline findings (long Kalman pipeline, 27 walk-forward splits)

1. **Static OLS hedge ratios fail out-of-sample on this market.** Of 10
   top correlation-ranked pairs, only 2 maintain ADF p < 0.05 on the
   held-out test slice when the hedge ratio is fitted in-sample.
2. **Kalman dynamic hedge ratios recover cointegration.** With Kalman
   parameters MLE-fitted on training only and forward-rolled on test
   (no peeking), 10 of 10 pairs hold cointegration at p < 0.001 OOS.
   This is the central methodological result.
3. **Once the right spread is used, classical z-score matches ML on
   signal quality and dominates on trade selectivity.** Across 27
   walk-forward splits: boost `pnl_mean_to_std` 0.307 [0.291, 0.323]
   vs z-score 0.282 [0.272, 0.292] vs LSTM 0.279 [0.258, 0.297] vs
   transformer 0.047 [-0.057, 0.148] (CI includes zero). Z-score
   per-trade win rate is 96.0% [95.7%, 96.4%] at 6× fewer trades.
4. **Hourly USDT pair-spread alpha on Binance.US has a cost ceiling of
   roughly 5 bps round-trip per leg.** Pre-cost Sharpe ≈ 8 with proper
   entry/exit state machine; maker fees (~5 bps round-trip) leave it
   barely positive; taker fees (~15 bps round-trip) eliminate it.

Two negative results, both robust across pipelines:

- 2-state Gaussian HMM regime filter does not isolate productive bars
  on either static or Kalman spreads (filtering 30-60% of bars cuts
  PnL by 30-60% on every model).
- Single chronological train/test splits dramatically over-state ML.
  An earlier single-split run had the transformer at #1 with
  `pnl_mean_to_std = 0.25`; with 27 walk-forward splits it drops to
  #4 with a CI that includes zero.

Quantified limitations:

- **Survivorship bound: 6.7% upper, ~5% realistic.** 14 tokens were
  delisted from Binance.US between t0 (2024-01-01) and the snapshot
  date (2026-01-22). After the liquidity top-50 filter, an estimated
  3-6 candidates are missing from the panel. Full delisting table in
  `docs/binance_us_delistings.md`.
- **Single exchange.** Cost ceiling result is Binance.US-specific.
- **Single quote currency.** USDT pairs only.
- **No microstructure**, pending L2 data (Phase 2).

## Reproducibility

All numbers in `docs/RESULTS.md` are produced by scripts in this repo.
Outputs land under `artifacts/` (gitignored). The local Binance.US
parquet store under `data/spot_1h/` is also gitignored (≈200 MB of
hourly candles for 195 symbols). To reproduce end-to-end:

```bash
# 0. Install deps
pip install -r requirements.txt   # numpy pandas statsmodels sklearn torch hmmlearn xgboost

# 0a. Build the local OHLCV store from Binance.US (one-time, ~30 min for full history).
#     Incremental on re-run; safe to interrupt and resume.
python3 store_data.py             # populates data/spot_1h/{SYMBOL}.parquet for all listed USDT pairs

# 1. Build the long Kalman dataset (uses local data store from store_data.py)
python3 scripts/run_first_branch.py --use-kalman --skip-deep \
  --out-dir artifacts/kalman_long \
  --t0 2024-01-01T00:00:00Z --train-days 180 --test-days 750 \
  --min-history-days 180 --liquid-top-n 50 --top-pairs 20

# 2. Walk-forward across 27 splits with deep models
python3 scripts/run_walk_forward.py \
  --dataset-dir artifacts/kalman_long/dataset \
  --out-dir artifacts/walk_forward_kalman_long \
  --deep --dl-epochs 4 --max-train-samples 12000 --max-test-samples 5000

# 3. Out-of-sample Kalman validation (the central finding)
python3 scripts/run_kalman_oos.py \
  --pairs-path artifacts/kalman_long/selected_pairs.parquet \
  --out-dir artifacts/kalman_oos --top-pairs 10

# 4. HMM ablation
python3 scripts/run_hmm_ablation.py \
  --walk-forward-dir artifacts/walk_forward_kalman_long \
  --features-dir artifacts/kalman_long/features \
  --dataset-dir artifacts/kalman_long/dataset \
  --out-dir artifacts/hmm_kalman_long

# 5. Portfolio backtest with proper entry/exit state machine
python3 scripts/run_portfolio_backtest.py \
  --predictions-path artifacts/walk_forward_kalman_long/walk_forward_predictions.parquet \
  --pairs-path artifacts/kalman_long/selected_pairs.parquet \
  --dataset-dir artifacts/kalman_long/dataset \
  --out-dir artifacts/backtest_kalman_long_sm \
  --state-machine --sm-entry-z 2.0 --sm-exit-z 0.5

# 6. Backtester sanity tests
python3 tests/test_backtest_sanity.py

# 7. Feature ablation (closes the time_since_zero_crossing leak suspicion)
python3 scripts/run_feature_ablation.py \
  --dataset-dir artifacts/kalman_long/dataset \
  --out-path docs/feature_ablation_kalman_long.csv

# 8. Figures
python3 scripts/plot_kalman_oos.py --csv artifacts/kalman_oos/kalman_oos_comparison.csv --out-path figures/kalman_oos_comparison.png
python3 scripts/plot_walk_forward_summary.py --metrics-csv artifacts/walk_forward_kalman_long/walk_forward_metrics_by_split.csv --out-path figures/long_walk_forward_pnl_mean_to_std.png --metric pnl_mean_to_std
python3 scripts/plot_feature_importance.py --dataset-dir artifacts/kalman_long/dataset --out-path figures/long_feature_importance.png
python3 scripts/plot_cumulative_pnl.py --backtest-dir artifacts/backtest_kalman_long_sm --out-path figures/long_cumulative_pnl.png --title "Cumulative PnL, state machine, 15bps round-trip"
```

Step 2 takes ~30 minutes wall-clock on an M-series Mac (deep models on
MPS). Everything else is ≤5 minutes.

## Repository layout

```
src/                       Library code
  data_store.py             Local Binance.US OHLCV store (parquet per symbol)
  universe.py               As-of-time universe with first-bar look-ahead fix
  pair_selection.py         Strict cointegration screen + correlation fallback
  features.py               Per-pair feature engineering with optional Kalman spread override
  kalman_hedge.py           Random-walk Kalman filter, MLE fit, OOS forward residuals
  ml_dataset.py             Sliding-window dataset builder with per-pair labels option
  modeling.py               Baselines, sklearn booster, LSTM, transformer (matched param count)
  modeling_big_transformer.py  Larger transformer with cosine LR + early stopping
  hmm_filter.py             2-state Gaussian HMM regime filter
  backtest.py               Cost-aware portfolio backtester (bar-by-bar + state-machine)
  l2_store.py               Phase-2 L2 schema and microstructure feature stubs

scripts/                   One driver per stage; everything is reproducible from here
  run_first_branch.py       Single-pass: features, dataset, baselines, booster, deep models
  run_walk_forward.py       Rolling walk-forward across saved dataset
  run_kalman_oos.py         Honest static-vs-Kalman OOS ADF comparison
  run_kalman_comparison.py  In-sample Kalman comparison + per-pair plots
  run_hmm_ablation.py       HMM regime filter raw-vs-filtered metrics
  run_portfolio_backtest.py Backtester runner (bar-by-bar or state machine)
  run_feature_ablation.py   One-feature-out ablation
  run_big_transformer.py    Properly-sized transformer with early stopping
  plot_*.py                 Figure scripts (one per plot type)

tests/test_backtest_sanity.py  Six hand-computable backtester sanity checks

docs/                      Detailed writeups (see RESULTS.md for headline)
  RESULTS.md                *** Headline findings, read first ***
  METHODS.md                *** Methodology details ***
  long_pipeline_results.md   Full 27-split tables
  state_machine_results.md   Cost-sensitivity sweep
  backtest_notes.md          Backtester architecture and what it does NOT model
  survivorship_bias.md       Survivorship analysis with external bound
  binance_us_delistings.md   Public Binance.US delisting log used for the bound
  archive/                   Earlier-pass results (superseded but preserved)

figures/                   PNG outputs from plot scripts (gitignored)
artifacts/                 Pipeline outputs: datasets, predictions, metrics (gitignored)
data/spot_1h/              Local Binance.US hourly OHLCV parquet store (gitignored)
minitron/                  Sibling project (Candella Quant backtester) (gitignored)
```

## Pointers

- `docs/RESULTS.md` — all four headline findings with reproducible tables.
- `docs/METHODS.md` — every methodological choice and why.
- `proposal.md` — original proposal; section "Remaining Work" is now mostly
  done and superseded by RESULTS.md.
