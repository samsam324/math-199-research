# Survivorship Bias

The Binance.US spot store used in this project was downloaded as a single
snapshot. Every symbol in the local data ends at the same wall-clock timestamp
(2026-01-22 04:00 UTC), so within the panel there is no observable delisting:
a coin is either present with data through the snapshot, or it is absent
entirely. This shapes the bias in a specific way.

## What is in the panel

- 195 USDT spot pairs, hourly bars.
- All symbols share the same final bar: 2026-01-22 04:00 UTC.
- First-observation dates range from 2019-09-23 (BTC/ETH/BNB/XRP/BCH, the
  oldest cohort) to 2025-12-18 (TWT, the newest).
- 14% of symbols (28 of 195) have less than 180 days of history before the
  primary as-of cutoff 2026-01-01. These are filtered out of any pair-selection
  pass that requires the full training window.

## Walk-forward delisting accounting

For the walk-forward windows actually used (90-day train / 30-day test):

| Window train_start | train_end | test_end | Symbols available at train_end | Dropped during test |
| --- | --- | --- | ---: | ---: |
| 2025-08-01 | 2025-10-30 | 2025-11-29 | 195 | 0 |
| 2025-08-31 | 2025-11-29 | 2025-12-29 | 195 | 0 |

No symbol drops out mid-window within our data, because the snapshot ends
after all walk windows close. Counts are reproducible via the analysis cell at
the bottom of `docs/survivorship_walk_counts.csv`.

## Why this still biases results upward

The bias is structural, not observable in the panel:

1. **Pre-snapshot delistings are invisible.** Any USDT pair that was tradable
   during the historical training window but was delisted from Binance.US
   before the 2026-01-22 download is missing from the panel entirely. Pair
   selection therefore searches only over coins that survived to the
   download date. Survivors tend to be the more liquid, longer-lived, lower-
   blow-up assets. A walk-forward window evaluated in 2024 should logically
   include those failed coins, but cannot.

2. **Cointegration that broke and stayed broken is excluded.** Two coins that
   were cointegrated in 2022 but then diverged because one was delisted in 2024
   would never enter the candidate set. The strict pair-selection screen will
   only find relationships that held together through to the snapshot. This
   inflates apparent half-life and apparent ADF significance.

3. **New listings are right-truncated.** Symbols listed in late 2025 enter the
   panel mid-window with limited history; the liquidity and coverage filters
   typically drop them, which is fine, but it means the universe in early
   walk-forward windows skews toward the older cohort.

## Direction of the bias

All three points push reported strategy performance up:

- Selected pairs are over-represented by survivors.
- Estimated mean-reversion strength is over-estimated because pairs whose
  spread blew apart and stayed apart are filtered out.
- Backtest Sharpe is over-estimated because the loss tail from delisted coin
  positions is missing.

## What we cannot fix from this snapshot

Reproducing the true historical universe would require a delisting log from
Binance.US giving the trading-status timeline for each symbol. The store does
not contain this. A correct fix is to repull data with a delisting registry
(or use an exchange that publishes one) and rebuild `compute_universe_at_time`
to include "tradable then, gone now" symbols up to the timestamp at which they
were actually removed. Until that is in place, every backtest result in this
paper carries an upward survivorship bias of unknown magnitude.

## Reproduce

```bash
python3 - <<'PY'
import sys; sys.path.insert(0, '.')
import pandas as pd
from src.data_store import StoreConfig, load_symbol, list_local_symbols

cfg = StoreConfig(interval="1h", data_dir="data")
last_obs = {s: load_symbol(cfg, s, columns=["close"]).index.max() for s in list_local_symbols(cfg)}
ser = pd.Series(last_obs).sort_values()
print(ser.describe())
PY
```
