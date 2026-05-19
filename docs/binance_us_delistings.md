# Binance.US delisting log (2023-2026)

Source: <https://support.binance.us/en/collections/10384537-announcements>.
Each delisting removes all trading pairs for the token.

## Delistings in window t0 = 2024-01-01 to snapshot = 2026-01-22

| Date | Token(s) |
| --- | --- |
| 2024-01-17 | BOND |
| 2024-03-07 | ANT, WAVES |
| 2024-07-30 | TUSD |
| 2024-09-29 | CUDOS |
| 2025-03-10 | MXC, REN, VITE |
| 2025-06-25 | BAL, CLV |
| 2025-10-22 | STMX, LOOM |
| 2025-10-28 | KDA |
| 2025-12-11 | OXT, JAM |

14 tokens delisted in window.

## Bound

- 195 surviving USDT pairs + 14 delisted = hypothetical 209 at t0
- 14/209 = 6.7% upper bound on survivorship rate
- After liquidity top-50 filter, 3-6 candidates would have been in our
  universe (ANT, WAVES, BAL, possibly CLV/TUSD)
- The rest were small-cap and would have been filtered out anyway
- Realistic bias: 5-12% of pre-filter candidate set, much smaller after
  the liquidity + cointegration screens

## Caveats

- Announcements page may miss quiet delistings; 6.7% is a lower bound on
  the true survivorship rate
- Tokens are sometimes delisted and relisted; log reflects state at snapshot
- Stablecoin delistings (TUSD, BUSD) redeem at par, no return tail
- A correct fix is repulling data with a delisting registry, treating
  delisted symbols as tradable up to their removal date

## Sources

- Help Center: <https://support.binance.us/en/collections/10384537-announcements>
- BOND: /hc/en-us/articles/20355527212311
- ANT, WAVES: /hc/en-us/articles/21682136266135
- TUSD: /en/articles/9843543
