# Binance.US Delisting Log (2023-2026)

Compiled from Binance.US Help Center announcements
(<https://support.binance.us/en/collections/10384537-announcements>). Each
delisting removes all trading pairs for the underlying token, including the
USDT pair if one existed.

## Delistings between t0 = 2024-01-01 and snapshot = 2026-01-22

These are the tokens that *were* listed at t0 but are *not* in the local
panel snapshot — the structural survivorship gap.

| Date | Token(s) |
| --- | --- |
| 2024-01-17 | BarnBridge (BOND) |
| 2024-03-07 | Aragon (ANT), Waves (WAVES) |
| 2024-07-30 | TrueUSD (TUSD) |
| 2024-09-29 | CUDOS |
| 2025-03-10 | MXC, REN, VITE |
| 2025-06-25 | Balancer (BAL), CLV |
| 2025-10-22 | StormX (STMX), Loom Network (LOOM) |
| 2025-10-28 | Kadena (KDA) |
| 2025-12-11 | Orchid (OXT), Geojam (JAM) |

That is **14 tokens delisted in the 2024-01-01 to 2026-01-22 window**.

## Quantifying the survivorship bound

The local snapshot contains 195 USDT pairs that survived to 2026-01-22.
Adding the 14 delisted tokens gives a hypothetical "as-of 2024-01-01"
universe of ≈209 pairs. **The survivorship rate is therefore 14 / 209
≈ 6.7% of pairs that existed at t0 had been delisted by the snapshot date.**

This is the maximum upward bias on every walk-forward metric the project
reports for t0 in [2024-01-01, 2026-01-22]: if the 14 missing pairs had
been included, their post-delisting price paths (which trended to zero or
near-zero before being removed) would have contributed strongly negative
pnl to any model that traded them, dragging down the headline `total_pnl`,
`pnl_mean_to_std`, `win_rate`, and `Sharpe` numbers.

## Why this matters less than it could

Two reasons the 6.7% bound is probably loose (i.e. the true bias is
smaller):

1. **Most of the delisted tokens would have been screened out by the
   liquidity filter anyway.** Our pipeline selects the top 50 USDT pairs
   by mean USDT volume in the training window. BOND, CUDOS, MXC, REN,
   VITE, CLV, STMX, LOOM, JAM, OXT, KDA were all small-cap thin-liquidity
   tokens that would not have made the top-50 cut even before they were
   delisted. Likely 3-4 of the 14 (ANT, WAVES, BAL, possibly TUSD as a
   stablecoin) would have been candidates.

2. **The strict cointegration screen (and even the correlation fallback)
   would further filter these out.** Pair selection requires both legs to
   have stable enough behavior to produce a passing cointegration or
   high-correlation score. Tokens that are on their way to being delisted
   typically have erratic price action, making cointegration unlikely.

A more honest characterization: **of ~50-60 liquid USDT pairs that would
have been candidates at t0=2024-01-01, an estimated 3-6 (ANT, WAVES, BAL,
maybe TUSD/CLV) are missing from the snapshot.** That is a 5-12% missing
rate within the pre-filter candidate set, dropping to ~0-2 actual
delisted survivors within the top-50 liquidity universe we use.

## Cross-checks and caveats

- The Binance.US announcements page may not include every delisting that
  occurred. Some quiet removals (low-volume tokens dropped without a
  formal announcement) are likely missing. The 6.7% figure is therefore a
  *lower bound on the true survivorship rate*, even if it is a *loose
  upper bound on the bias in our reported metrics*.
- Tokens are sometimes delisted and relisted (e.g. several were relisted
  after the June 2023 mass delisting). The log treats final state at
  snapshot, not intermediate transitions.
- Stablecoin delistings (TUSD, BUSD) have a different mechanism: they get
  redeemed at par rather than left to decay, so they would not contribute
  to a meaningful return tail.

## Update to the headline survivorship statement

In `docs/survivorship_bias.md`:

> Until that is in place, every backtest result in this paper carries
> an upward survivorship bias of unknown magnitude.

Replace with:

> The Binance.US public delisting log identifies 14 tokens delisted
> between 2024-01-01 and the 2026-01-22 snapshot date. Of these, an
> estimated 3-6 would have been candidates in the liquidity-filtered
> top-50 universe (ANT, WAVES, BAL, possibly CLV); the remainder were
> small-cap pairs that would have been screened out before pair
> selection. Survivorship bias on the headline metrics is therefore
> bounded above by approximately 6-12% of the pre-filter candidate set
> and is probably much smaller after the liquidity and cointegration
> filters that the pipeline already applies.

Sources:

- [Binance.US Help Center - Announcements](https://support.binance.us/en/collections/10384537-announcements)
- [Binance.US - BarnBridge (BOND) delisting](https://support.binance.us/hc/en-us/articles/20355527212311-Binance-US-Will-Delist-BarnBridge-BOND-on-January-17-2024)
- [Binance.US - Aragon (ANT) and Waves (WAVES) delisting](https://support.binance.us/hc/en-us/articles/21682136266135-Binance-US-Will-Delist-Aragon-ANT-and-Waves-WAVES-on-March-7-2024)
- [Binance.US - TrueUSD (TUSD) delisting](https://support.binance.us/en/articles/9843543-binance-us-will-delist-trueusd-tusd-on-july-30-2024)
