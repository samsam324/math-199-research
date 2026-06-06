# Phase-2 L2 / volume-as-information analysis

First analysis run on **real Level-2 order-book + tick-trade data** (Tardis,
Binance spot), the data we were blocked on in phase 1. Raw metric tables are in
[`docs/l2_results/`](l2_results/).

## Data & setup

- **Source:** Tardis `book_snapshot_25` (top-25 depth) + `trades` (tick volume
  with aggressor side), Binance spot, downloaded and ingested to 1s bars via
  `scripts/download_tardis_l2.py` (`src/l2_store.py`, `src/trades_store.py`).
- **Coverage at run time:** 111 contiguous days from 2024-01-01, all 50
  top-liquidity USDT symbols (`data/l2_universe_top50.txt`).
- **Window:** dataset built at t0 = 2024-03-28 (87d train / 21d test); ML
  walk-forward = **2 splits, 60d train / 21d test / 15d step, 20,000 test
  samples**. Portfolio backtest = single split, 253 hourly bars.
- **Pairs:** 20 pairs. NB: none passed the strict cointegration (ADF) filter
  over this window, so selection fell back to **correlation ranking** — the main
  caveat (see below).
- **Volume features** (`src/microstructure_features.py`): per-leg signed
  order-flow imbalance (OFI), VPIN, Kyle's-λ, quoted spread, trade intensity —
  merged onto the pair feature store at **100% row coverage** for this window.

## Finding 1 — Volume-as-information does **not** clearly help

XGBoost, identical pipeline, base features vs. base + 10 volume features:

| metric | base | + volume |
|---|---|---|
| accuracy | 0.457 | 0.444 |
| macro-F1 | 0.347 | 0.359 |
| win rate | 48.1% | 49.5% |
| trade PnL/σ | +0.0142 | +0.0134 |
| total PnL (proxy) | +12.1 | +12.8 |
| # trades | 5,887 | 7,123 |

Order-flow features give a small macro-F1 / win-rate bump but **lower accuracy
and flat risk-adjusted PnL**; the model simply trades ~20% more without
improving net quality. Effectively a **null result** for order flow on
spread-reversion prediction in this window.

> Methodological note: an earlier run with only 49.8% volume-feature coverage
> (microstructure panels built mid-ingest) *overstated* the macro-F1 gain
> (+0.021 vs. +0.012 at full coverage) and showed accuracy up rather than down.
> The full-coverage numbers above supersede it.

## Finding 2 — "Classical beats ML" is **not robust** — it flips by window

In this Q1-2024 L2 window XGBoost is **positive** (PnL/σ +0.014) while the
z-score rule **loses** (−0.059) — the reverse of the 2025–26 full-history run
and the shorter preliminary window, where the z-score rule won. The model
ranking is regime-dependent and should not be reported as a fixed conclusion.

## Finding 3 — Realistic L2 execution costs are cheaper than the flat assumption

Portfolio backtest, flat 5 bps slippage vs. L2 book-walk slippage
(`src/l2_costs.py`, `--with-l2-costs`):

| model | flat Sharpe (total ret) | L2 Sharpe (total ret) |
|---|---|---|
| z-score | −8.45 (−12,827) | −6.50 (−9,868) |
| XGBoost | −28.0 (−22,143) | −23.4 (−18,286) |

Walking the real book makes execution **~17–23% cheaper** than the flat 5 bps
assumption for these liquid majors — i.e. prior flat-cost backtests were
conservative. (All strategies remain unprofitable after costs in the portfolio
backtest; turnover dominates. The `max_drawdown` column is percent-of-peak with
a degenerate denominator and is **not** interpreted here.)

## Caveats

1. **Correlation-fallback pairs, not cointegration-screened** — biggest weakness;
   the mean-reversion premise the strategy relies on is not yet validated on this
   window. Re-running the Kalman/ADF screen on the L2 window is the next step.
2. Single quarter (Jan–Apr 2024), 2 walk-forward splits, one portfolio split.
3. Results are directional, not conclusive.

## Reproduce

```
python scripts/build_microstructure_panel.py
python scripts/run_first_branch.py --with-micro --per-pair-label --skip-deep \
    --train-days 87 --test-days 21 --t0 2024-03-28T00:00:00Z --out-dir artifacts/fb
python scripts/run_walk_forward.py --dataset-dir artifacts/fb/dataset --out-dir artifacts/wf_base  --train-days 60 --test-days 21 --step-days 15
python scripts/run_walk_forward.py --dataset-dir artifacts/fb/dataset --out-dir artifacts/wf_vol   --train-days 60 --test-days 21 --step-days 15 --with-micro
python scripts/run_portfolio_backtest.py --predictions-path artifacts/fb/predictions.parquet \
    --pairs-path artifacts/fb/selected_pairs.parquet --dataset-dir artifacts/fb/dataset --out-dir artifacts/pb_flat
python scripts/run_portfolio_backtest.py --predictions-path artifacts/fb/predictions.parquet \
    --pairs-path artifacts/fb/selected_pairs.parquet --dataset-dir artifacts/fb/dataset --out-dir artifacts/pb_l2 --with-l2-costs
```
