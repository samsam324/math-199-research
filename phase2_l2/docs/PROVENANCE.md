# Provenance: what carried over from phase 1, what is new

Phase 1 = `crypto_stat_arb/` (parent of this directory).
Phase 1 results live at `crypto_stat_arb/docs/RESULTS.md` and are NOT being
re-litigated here.

## Ported as-is (concept + code)

| Phase 2 file | Phase 1 origin | Notes |
| --- | --- | --- |
| `src/kalman_hedge.py` | `src/kalman_hedge.py` | Identical state-space model. Inputs are now log-microprice instead of log-close. |
| `src/pair_selection.py` | `src/pair_selection.py` | Identical scoring (corr -> OLS -> ADF -> half-life -> beta stability). Half-life parameter renamed `_hours` -> `_bars` since cadence is configurable. |
| `src/hmm_filter.py` | `src/hmm_filter.py` | 2-state Gaussian HMM. Feature input columns are now bar-cadence analogues; caller passes them. |
| `src/modeling.py` | `src/modeling.py` | Baselines, booster, LSTM, small transformer. Architectures unchanged. TABULAR_COLUMNS now lives in `src/ml_dataset.py` and is microstructure-native. |
| `src/modeling_big_transformer.py` | `src/modeling_big_transformer.py` | 4-layer transformer. Unchanged. |
| `src/statistics.py` (NEW MODULE) | `scripts/run_hac_sharpe.py`, `scripts/run_deflated_sharpe.py`, `scripts/plot_walk_forward_summary.py` | Consolidated HAC, DSR, block bootstrap into one library module. |

## New for phase 2

| File | Purpose |
| --- | --- |
| `src/tardis_ingest.py` | Daily-CSV downloader + canonical-schema parsers. THE SEAM: only two functions hold Tardis-specific column knowledge. |
| `src/l2_store.py` | Canonical L2 schema, validation, write/read by symbol-day. Drafted in phase 1 (`crypto_stat_arb/src/l2_store.py`), finished here. |
| `src/trade_store.py` | Canonical trades schema. Symmetric to l2_store. |
| `src/universe.py` | As-of-time universe over L2 + trades file presence. Replaces phase 1 universe over OHLCV first/last bars. |
| `src/microstructure.py` | Microprice, OBI, quoted spread, trade signing (aggressor / quote rule / tick rule), size buckets, VPIN, sweep detection. |
| `src/bars.py` | 1-second bar aggregation joining L2 state and trade flow. |
| `src/features.py` | Per-pair feature builder. Microstructure-native. Pre-reg headline feature `inst_buy_imbalance_over_leg`. |
| `src/ml_dataset.py` | Sliding-window dataset over short bars. New TABULAR_COLUMNS (microstructure summary stats). |
| `src/backtest.py` | L2-aware backtester. Walks the book for impact. No flat bps when book columns are present. |

## What was deliberately NOT ported

| Phase 1 file | Reason for omission in phase 2 |
| --- | --- |
| `src/data_store.py` | Replaced by Tardis ingestion. |
| `main.py`, `store_data.py` | Specific to Binance.US REST. |
| Old `src/universe.py` | Replaced by daily-file-presence universe. |
| Old `src/features.py` | Replaced by microstructure-native feature builder. |
| `src/l2_store.py` ingestion stubs | Implemented in `src/tardis_ingest.py`. |

## Audit invariants carried forward

1. Universe construction must exclude post-t0 listings.
2. Features at time t use only data with timestamp <= t.
3. Targets use only data in [t, t + horizon].
4. Per-pair label thresholds respect `label_train_end`.
5. Kalman parameters identical across train/test boundary.
6. Deep model standardization uses training stats only.
7. Backtester held_position_t = signal_{t-1} (one-bar execution lag).
8. Walk-forward windows strictly time-ordered, no train/test overlap.

All eight have corresponding tests in `tests/test_leakage_audit.py` (8 tests)
and `tests/test_backtest_sanity.py` (6 tests).
