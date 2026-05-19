# MATH 199 — Statistical Arbitrage in Crypto Pair Spreads

UCLA Math 199 research project: do crypto pair-spreads exhibit measurable
mean reversion, and can it be characterized rigorously enough to draw
inferences about market efficiency? Authors: Jack Lutz, Sammy Adham,
Frank Kronewitter.

The proposal (`proposal.md`) follows Gatev, Goetzmann, and Rouwenhorst
(2006) on equity pairs trading, adapted to crypto. Read this README
first, then `docs/RESULTS.md` for the headline findings and
`docs/NOTES.md` for open questions, concerns, and what's still
unresolved. Methodology details in `docs/METHODS.md`.

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
   held-out test slice. Generalizes to the full 1,219-pair universe:
   static OLS holds cointegration at p < 0.05 OOS on only 5.4% of pairs.
2. **Kalman dynamic hedge ratios recover cointegration on essentially
   every pair.** Parameters MLE-fit on training only, forward-rolled on
   test, no peeking. 10/10 selected pairs hold p < 0.001 OOS. Across
   the full universe: **99.7% of pairs hold p < 0.05 OOS** (1,215 / 1,219).
   This is the central methodological result.
3. **LSTM leads on signal quality, z-score dominates on trade
   selectivity.** Across 27 walk-forward splits on the post-audit
   pipeline: LSTM `pnl_mean_to_std` 0.376 [0.362, 0.388] > booster
   0.356 [0.342, 0.368] > z-score 0.282 [0.274, 0.291] > transformer
   0.018 [-0.025, 0.064] (CI includes zero). Per-trade win rate:
   z-score 96.3% [95.9%, 96.6%], LSTM 79.5% [76.8%, 82.4%], booster
   74.1% [72.0%, 76.0%]. Z-score makes 4× fewer trades and wins on
   selectivity; LSTM wins on edge per bar.
4. **Cost ceiling: weaker than naive Sharpe suggests.** Pre-cost Sharpe
   ≈ 8 with proper entry/exit state machine, but Bailey & López de
   Prado's deflated Sharpe (N≈25-50 trials across the project)
   indicates this is at most marginally above the chance-best of N
   random strategies. At 5 bps round-trip per leg the Sharpe falls to
   ~0.9 and does NOT survive DSR correction at any plausible N. At
   15 bps everything is deeply negative. **Defensible claim**: at
   hourly cadence on Binance.US USDT pairs, no tried model produces a
   selection-corrected Sharpe that beats chance at realistic cost
   levels. The pre-cost signal may be real but cannot be confidently
   distinguished from selection bias on this dataset alone.

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

# 1. Build the long Kalman dataset with per-pair labels (post-audit)
python3 scripts/run_first_branch.py --use-kalman --skip-deep --per-pair-label \
  --out-dir artifacts/kalman_long \
  --t0 2024-01-01T00:00:00Z --train-days 180 --test-days 750 \
  --min-history-days 180 --liquid-top-n 50 --top-pairs 20

# 2. Walk-forward across 27 splits with deep models
python3 scripts/run_walk_forward.py \
  --dataset-dir artifacts/kalman_long/dataset \
  --out-dir artifacts/walk_forward_kalman_long \
  --deep --dl-epochs 4 --max-train-samples 12000 --max-test-samples 5000

# 3. Out-of-sample Kalman validation on 10 selected pairs
python3 scripts/run_kalman_oos.py \
  --pairs-path artifacts/kalman_long/selected_pairs.parquet \
  --out-dir artifacts/kalman_oos --top-pairs 10

# 4. Full-universe Kalman cointegration screen (1,219 pairs)
python3 scripts/run_kalman_pair_screen.py \
  --out-dir artifacts/kalman_screen \
  --t0 2024-01-01T00:00:00Z --train-days 90 --test-days 30 \
  --liquid-top-n 50 --min-history-days 180

# 5. HMM ablation (negative result, robust)
python3 scripts/run_hmm_ablation.py \
  --walk-forward-dir artifacts/walk_forward_kalman_long \
  --features-dir artifacts/kalman_long/features \
  --dataset-dir artifacts/kalman_long/dataset \
  --out-dir artifacts/hmm_kalman_long

# 6. Cost-aware portfolio backtest with entry/exit state machine
python3 scripts/run_portfolio_backtest.py \
  --predictions-path artifacts/walk_forward_kalman_long/walk_forward_predictions.parquet \
  --pairs-path artifacts/kalman_long/selected_pairs.parquet \
  --dataset-dir artifacts/kalman_long/dataset \
  --out-dir artifacts/backtest_sm_15bps \
  --state-machine --sm-entry-z 2.0 --sm-exit-z 0.5
#  ... change --taker-fee-bps / --slippage-bps for 0 and 5 bps variants

# 7. Backtester sanity tests + leakage audit tests
python3 tests/test_backtest_sanity.py
python3 tests/test_leakage_audit.py

# 8. Feature ablation
python3 scripts/run_feature_ablation.py \
  --dataset-dir artifacts/kalman_long/dataset \
  --out-path docs/feature_ablation_fixed.csv

# 9. HAC-corrected Sharpe (Newey-West lag 24)
python3 scripts/run_hac_sharpe.py \
  --predictions-path artifacts/walk_forward_kalman_long/walk_forward_predictions.parquet \
  --out-path docs/hac_sharpe_fixed.csv --lag 24

# 10. Figures
python3 scripts/plot_kalman_oos.py --csv artifacts/kalman_oos/kalman_oos_comparison.csv --out-path figures/kalman_oos_comparison.png
python3 scripts/plot_kalman_screen.py --csv artifacts/kalman_screen/kalman_pair_screen.csv --out-path figures/kalman_screen_cdf.png
python3 scripts/plot_walk_forward_summary.py --metrics-csv artifacts/walk_forward_kalman_long/walk_forward_metrics_by_split.csv --out-path figures/walk_forward_pnl_mean_to_std.png --metric pnl_mean_to_std --block-size 3
python3 scripts/plot_feature_importance.py --dataset-dir artifacts/kalman_long/dataset --out-path figures/feature_importance.png
python3 scripts/plot_cumulative_pnl.py --backtest-dir artifacts/backtest_sm_15bps --out-path figures/cumulative_pnl_15bps.png --title "Cumulative PnL, state machine, 15bps round-trip"
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
  NOTES.md                  *** Open questions, concerns, unresolved threads ***
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
- `docs/NOTES.md` — open questions, concerns, what's not done, phase-2 plan.
- `docs/METHODS.md` — every methodological choice and why.
- `proposal.md` — original proposal; section "Remaining Work" is now mostly
  done and superseded by RESULTS.md.
