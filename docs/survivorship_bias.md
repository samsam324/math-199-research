# Survivorship

Store was downloaded as a single snapshot. Every symbol ends at the same
wall-clock timestamp (2026-01-22 04:00 UTC). Within the panel, no
observable delisting. Bias is structural.

## Panel composition

- 195 USDT spot pairs, hourly bars
- All share final bar 2026-01-22 04:00 UTC
- First-obs range: 2019-09-23 (BTC/ETH/BNB/XRP/BCH) to 2025-12-18 (TWT)
- 28 of 195 (14%) have <180d history before 2026-01-01; filtered out by
  any pair selection requiring full training window

## Within-window delisting (literally zero)

| Window | available at train_end | dropped during test |
| --- | ---: | ---: |
| 2025-08-01..2025-10-30..2025-11-29 | 195 | 0 |
| 2025-08-31..2025-11-29..2025-12-29 | 195 | 0 |

Snapshot ends after every walk window. See
`docs/survivorship_walk_counts.csv`.

## Direction of bias (all upward)

- Pre-snapshot delistings are invisible: tokens tradable historically but
  delisted before 2026-01-22 are gone from the panel
- Pair selection only finds relationships that survived to snapshot date
- Half-life and ADF significance over-estimated
- Backtest Sharpe over-estimated (missing tail from delisted positions)

## External bound from Binance.US delisting log

14 tokens delisted from Binance.US in window 2024-01-01 to 2026-01-22:
BOND, ANT, WAVES, TUSD, CUDOS, MXC, REN, VITE, BAL, CLV, STMX, LOOM, KDA,
OXT, JAM. See `docs/binance_us_delistings.md`.

- Adding these to 195: hypothetical t0 universe ~209
- 14/209 = 6.7% upper bound
- After liquidity top-50 filter: 3-6 candidates missing (ANT, WAVES, BAL,
  maybe CLV/TUSD)
- Realistic bias: ~5% of pre-filter candidate set
- 6.7% is loose upper, ~5% is realistic upper
- Both apply only to the headline metrics; absolute values may shift but
  rankings unlikely to flip
