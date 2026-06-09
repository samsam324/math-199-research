# H1 pre-registered one-shot — result

*Dated 2026-06-08. The single locked evaluation of the pre-registered hypothesis
(`phase2_l2/docs/NOTES.md`, locked 2026-06-03). Run once; not re-tuned.*

## Verdict: **FAIL** (and the test is severely underpowered on the available universe)

The hypothesis predicted a **negative** Spearman correlation between the over-leg institutional
buy-imbalance and the forward spread change on |spread_z|≥2 bars. The observed correlation is
**+0.007 (slightly positive — the opposite sign)**, nowhere near significant in the predicted
direction. **It does not pass.**

## What was run

- **Held-out tail boundary (fixed before looking):** **2024-10-19 19:12:00 UTC** — exactly the
  80th-percentile timestamp of the 2024 1-second L2 range (verified). Only bars at/after it were
  used for evaluation; selection and hedge fitting used only data strictly before it.
- **Pairs (the substitution, per `docs/REMOTE_TASKS.md`):** the locked spec's "top-10 Kalman-
  cointegrated pairs by OOS ADF p-value" is a known whitening artifact (loop 6), so it was
  replaced by the **cleanly Engle-Granger-cointegrated pairs (p<0.05, stablecoins dropped),
  ranked by OU reversion speed** (kappa — the metric loop 7 validated, Spearman ρ=0.46),
  computed on hourly closes strictly before the boundary. The 10 pairs:
  `VTHO/POLYX, DOT/EGLD, VTHO/IMX, VET/HBAR, EGLD/SAND, FIL/NMR, ALGO/EGLD, ADA/POLYX,
  AAVE/NMR, GRT/ZIL`. (Only 11 pairs clear EG p<0.05 — consistent with loop 6's "genuine
  cointegration is rare.")
- **Locked parameters (unchanged):** 1-second bars; W=60-bar trailing mean for X; K=60-bar
  forward target; institutional bucket ≥ $100k; entry |spread_z|≥2 with z_window=3600.
- **Features** were built with the collaborator's frozen `phase2_l2` code (microprice spread,
  `spread_z`, `inst_buy_imbalance`, `target_spread_change`). Pooled held-out sample:
  **1,026,923 bars with |spread_z|≥2** across the 10 pairs.

## Test statistic and HAC-corrected p-value

X = trailing W=60 mean of the over-leg institutional buy-imbalance; Y = `target_spread_change(t,t+60)`.

| variant | Spearman rho | one-sided p (negative dir.) | pass? |
|---|---:|---:|---|
| **Primary** (within-pair ranks, HAC lag 240, per-pair combined) | **+0.0070** | **0.937** | no |
| Primary, block-bootstrap (block 300) | +0.0070 | 0.940 | no |
| Pre-registered literal (pooled ranks, HAC lag 60) | +0.0092 | 0.967 | no |
| Robustness (raw-spread over-leg, proper alpha) | +0.0033 | 0.78 | no |

p is **>0.93 in the predicted direction at every Newey-West lag (60/120/240) and by block
bootstrap** — an unambiguous fail. (Per-pair ρ range −0.016…+0.018, mixed sign, all economically nil.)

## Two methodology notes (both documented as the spec requires)

1. **Over-leg definition.** `features.py` sets `over_is_a = (spread > 0)`, which assumes the
   fitted `alpha` centers the spread. The hourly-close-fit alpha applied to 1s microprice leaves
   the raw spread persistently off-zero (−1 to −6 for these pairs), which would freeze the
   over-leg on one leg. The over-leg was therefore taken from **`spread_z > 0`** (the
   alpha-invariant, and the coherent notion of "currently expensive" given the |spread_z|≥2
   entry). The collaborator's raw-spread version (with proper alpha) is reported as robustness;
   both fail with the same sign.
2. **HAC correction (adversarial-audit driven).** The naive plan (pooled ranks, Newey-West lag
   60) **understates** the SE: Y is an overlapping 60-ahead target and X a 60-bar mean, so the
   rank-product series is autocorrelated past 60 lags, and concatenating 10 pairs fabricates
   cross-pair autocovariances. The reported p's use **per-pair-combined HAC at lags 60/120/240**
   plus a **block bootstrap**, with **within-pair** standardization primary (pooled mixes
   per-pair Y-scales). All agree; since the result fails decisively, the corrections only
   reinforce it.

## The decisive caveat — the test is underpowered on this universe

**X is zero for 99.3% of the |spread_z|≥2 bars** (220 distinct values across 1.03M obs; three
pairs have X≡0). The $100k+ institutional bucket is essentially **empty at 1-second scale on
these cleanly-cointegrated thin-alt pairs**. So the FAIL is "**no signal *and* no power**," not
a strong falsification. This exposes a fundamental tension created by the cointegration
retraction: **the genuinely-cointegrated pairs are illiquid alts with ~no institutional flow,
while the pairs that carry meaningful $100k flow (BTC/ETH/SOL majors) are not genuinely
cointegrated** (loop 6). The pre-registered H1 cannot be meaningfully evaluated where both
conditions hold, because in this universe they are close to mutually exclusive.

**Verified the sparsity is real, not a pipeline bug:** across the 15 legs over ~6.56M held-out
1-second bars each, the number of seconds containing *any* $100k+ trade is **0 (VTHO) to 743
(ADA)** — at most ~0.01% of seconds; institutional share of dollar volume is 0–0.9%. VTHO has
**zero** $100k+ trades in the entire held-out period. The $100k threshold is simply far above
these coins' trade sizes.

## Exploratory (NOT the locked test): is there a signal at a *tradeable* threshold?

The collaborator's `NOTES.md` flags that $100k may need a liquidity-appropriate threshold. As a
**hypothesis-generating** check (it does **not** re-run or change the locked H1), the over-leg
imbalance was rebuilt with a **≥$10k** bucket (`scratch/h1_explore_threshold.py`), where X has
real variation (80% zero, vs 99.3% at $100k). Result: within-pair combined Spearman **rho =
−0.0001**, one-sided p ≈ **0.49** at lags 60/120/240 and by bootstrap — a **clean null** (per-pair
rho's scatter ±0.01–0.03 around zero, mixed sign). So even with a tradeable institutional
threshold there is **no institutional-flow → reversion signal** on these pairs: the hypothesis
looks genuinely **unsupported here, not merely underpowered**. A re-pre-registration with a lower
threshold would, on this evidence, also fail. (This stays exploratory — it would need its own
locked pre-registration to count as a test.)

## Interpretation (one sentence)

On the cleanly-cointegrated universe the pre-registered H1 fails — institutional buy-imbalance
on the over-leg shows no negative (reversion-predictive) relationship with the forward spread
change (ρ≈+0.007, one-sided p≈0.94); the locked $100k test is underpowered (institutional flow
near-absent on these illiquid pairs), but the exploratory ≥$10k variant — where X does vary —
returns a clean null (ρ≈0.000, p≈0.49), so the signal appears genuinely **absent**, not merely
unmeasured; a definitive test would still require a new pre-registration on a universe that is
both cointegrated and liquid, which (given the loop-6 cointegration retraction) may not exist
in this data.

*Scripts: `scratch/h1_select.py` (selection), `h1_build_features.py` (held-out features via
`phase2_l2` code), `h1_test.py` (this evaluation); raw log `scratch/h1_test_result.log`.*
