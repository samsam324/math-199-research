# Remote tasks (from Mac, 2026-06-08)

Two analyses still needed before the paper can be written. Both require the L2 / Tardis data that lives on the Windows machine; neither can be done from the Mac. Listed in priority order.

## Task 1 — Run the pre-registered H1 once

The hypothesis is locked in `phase2_l2/docs/NOTES.md` (commit on 2026-06-03). It was never formally evaluated. The loop iterations did informal exploratory falsification of variants but the single one-shot test that the DSR-immunity argument depends on has not been run.

Spec, exactly as locked:

- Universe: top 10 Kalman-cointegrated pairs by OOS ADF p-value at p < 0.05.
  - **Caveat from loop 6**: that screen is now known to be a whitening artifact. Replace the Kalman-ADF selection with the cleanly-selected pairs from `scratch/clean_coint_pairs.csv` (top 10 by OOS reversion speed, the metric loop 7 validated with Spearman rho = 0.46). Note this substitution explicitly in the result writeup.
- Cadence: 1-second bars
- Window W (rolling mean): 60 bars
- Horizon K (target): 60 bars
- Institutional bucket: trades with notional >= $100,000 USDT
- Entry filter: `|spread_z| >= 2` using `z_window_bars = 3600`
- Held-out tail: LAST 20% of the available time range, defined before looking at it, no cherry-picking
- Test: one-sided Spearman correlation between `inst_buy_imbalance_over_leg(t)` and `target_spread_change(t, t+K)` on the held-out tail, computed once, with Newey-West HAC SEs at lag K=60
- Pass: p < 0.05 in the predicted (negative) direction

Output: a single dated note at `phase2_l2/docs/H1_RESULT.md` with:
- The pair list used
- The exact held-out tail boundary timestamp
- The test statistic, HAC-corrected p-value
- Whether it passes
- A single sentence on interpretation

Do not re-run, re-tune, or pick a different test if it fails. This is the locked one-shot.

## Task 2 — Delisted-coin survivorship re-test

The no-stop reversion alpha (Sharpe ~1.7-2.5, market-neutral, t=3.65 across 19 windows) has passed every survivorship check that uses the on-disk 204-symbol universe. The one outstanding hole is fully-delisted coins. The loop has been working on the universe of "coins that still exist on disk," which by construction excludes the survivorship-worst tail.

Concretely:
- LUNAUSDT and LUNCUSDT (pre- and post-rebrand) — the May 2022 collapse
- FTTUSDT — the November 2022 FTX collapse
- USTUSDT — the May 2022 depeg
- Any other USDT pair with a known delisting event in the 2023-07 → 2026-05 window

Check whether Tardis has these symbols' L2 + trades for the relevant pre-collapse windows. Document what's available and what isn't.

If at least the three majors above are available:
- Pull them
- Add to the universe with point-in-time entry (symbol enters when it would have been in the top-N liquidity universe, exits on its actual delisting date)
- Re-run the reversion alpha test from loops 10-14
- Report: does the alpha survive, weaken, or break?

If Tardis doesn't have them:
- Document the search, where they live (if anywhere), and what would be needed
- The paper will then carry an explicit "tested on point-in-time universe of coins surviving to 2026-05; fully-delisted coins unavailable" caveat instead of being silent on it

Output: `phase2_l2/docs/SURVIVORSHIP_DELISTED.md` with what was checked, what was found, and how it changes (or doesn't change) the deployable Sharpe range.

## Why these two, why now

The Mac side is rewriting `paper/paper.tex` around the corrected story (the Kalman 99.7% retraction, the no-stop alpha, the microstructure null). That rewrite needs the H1 result to land the pre-registration argument and the delisted-coin check to close the last survivorship caveat. Without these, the paper has to soften both claims.

After both land, the paper writes itself.
