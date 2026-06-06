# Phase 2 — L2 statistical arbitrage on Binance.com USDT perps

L2-native rebuild of the pair-spread project. Phase 1 (hourly OHLCV) is preserved
in the parent repo for reference; this subfolder is the clean second pass.

**Status:** infrastructure scaffolding. Real Tardis ingestion is gated on the
`data.py` reference script + a confirmed API key.

## Pre-registered hypothesis

See `docs/NOTES.md`. One hypothesis, one held-out window, evaluated once.

> Net signed institutional-bucket buy volume on the over-leg of a
> Kalman-cointegrated USDT-perpetual pair predicts mean reversion of the
> microprice spread within K seconds.

## Universe

Binance.com USDT-margined perpetuals. Universe-at-time, liquidity top-N, same
construction pattern as phase 1.

## Cadence

L2 + trades are ingested at native resolution and aggregated to 1-second bars
for the feature pipeline.

## Layout

```
src/
  tardis_ingest.py       # Tardis daily-CSV downloader -> canonical parquet (real download behind a seam)
  l2_store.py            # canonical L2 schema (per symbol per UTC day)
  trade_store.py         # canonical trades schema (per symbol per UTC day)
  universe.py            # as-of-time universe over Tardis's listing catalog
  microstructure.py      # microprice, OBI, quoted spread, signed flow, size buckets, VPIN, sweeps
  bars.py                # 1s/event aggregation of L2 + trade features
  pair_selection.py      # corr -> OLS -> ADF -> half-life -> beta stability (ported)
  kalman_hedge.py        # state-space dynamic hedge, MLE fit, forward residuals (ported)
  features.py            # per-pair feature builder (microstructure-native)
  ml_dataset.py          # sliding windows over short bars
  hmm_filter.py          # 2-state Gaussian HMM regime filter (ported)
  modeling.py            # baselines, booster, LSTM, small transformer (ported)
  modeling_big_transformer.py  # 4-layer transformer (ported)
  backtest.py            # L2-aware: walks book for impact, no flat bps
  statistics.py          # HAC, deflated Sharpe, block bootstrap

scripts/                 # CLI drivers
tests/                   # sanity + leakage audit
docs/                    # NOTES, METHODS, RESULTS, PROVENANCE
data/                    # gitignored
artifacts/               # gitignored
```

## What was ported from phase 1, what is new

See `docs/PROVENANCE.md`.

## Reproducibility

Pending real data. Reproduction stanza will be filled in after the first
end-to-end run on Tardis.
