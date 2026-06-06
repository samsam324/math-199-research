# ⚠️ Correction: the "99.7% OOS cointegration" finding is a Kalman artifact

**Status: audit finding — recommend independent review by advisor/collaborator before acting.**
**Affects:** `docs/RESULTS.md` (the Kalman cointegration table, ~line 40–44: "Kalman finds [cointegration] on 99.7%"),
and any downstream step that selected pairs from `kalman_selected_pairs.parquet` on the Kalman ADF p-value.

## The claim
RESULTS.md reports that an OOS Kalman dynamic-hedge screen finds cointegration on **99.7%** of ~1,219 pairs
(vs 5.4% for a static OLS+ADF test), and frames this as "the strongest single finding."

## Why it is an artifact
The screen (`scripts/run_kalman_pair_screen.py` → `src/kalman_hedge.py`) runs the ADF test on the Kalman
filter's **one-step prediction innovations**. With a non-zero state-evolution variance (`q_beta > 0`), the
time-varying hedge ratio adapts to track *whatever* relationship exists between the two series, so the
innovations are **white by construction** — an ADF on them rejects the unit root for almost *any* pair,
cointegrated or not. Testing the stationarity of a filter's own innovations is circular. The train/test
split noted in NOTES.md does not fix this: it removed a parameter discontinuity at the boundary, but the
innovation-whitening is intact out-of-sample.

## The decisive evidence: a placebo that cannot be cointegrated passes identically
Identical Kalman screen (MLE on 90d train, fixed params forward-rolled on 30d test, ADF on test innovations):

| Series tested | ADF p<0.05 | p<0.001 |
|---|---:|---:|
| **Real top-50 pairs** | **100.0%** | 96.4% |
| Placebo (a): independent simulated random walks | **100.0%** | 98.3% |
| Placebo (b): coin vs **phase-randomized** surrogate of another coin | **100.0%** | 95.0% |
| Placebo (c): real pair, one leg **block-shuffled** | **100.0%** | 95.0% |

The placebos are, by construction, **not** cointegrated, yet they pass at the same 100% rate as real pairs.
The Kalman ADF therefore carries **zero information** about cointegration. (This reproduces the original
99.7% / 96.3% almost exactly — i.e. the headline number is the artifact.)

## What the genuine OOS cointegration rate is (clean, non-circular tests)
| Test (same train/test split, no Kalman whitening) | Genuine OOS cointegration rate |
|---|---:|
| Engle–Granger on test-window log-prices (all 1,225 pairs) | **2.1%** (p<0.05), 0.2% (p<0.01) |
| Static OLS train-hedge → ADF on test residual | **3.2%** (p<0.05), 0.8% (p<0.01) |

Both sit **at or below the 5% null false-positive floor**. Essentially **none** of the top-50 pairs are
genuinely cointegrated out-of-sample. (The static "5.4%" in RESULTS.md is itself ≈ the null floor — i.e. it
was already evidence of *no* cointegration, not a weak positive.)

## The reversion premise still partly survives — but modestly, and for a different reason
Re-tested on the full multi-year hourly history (24,942 bars, 2023-07→2026-05; ~4,000 |z|>2 events;
pair-clustered **and** block bootstrap). |z|>2 reversion is strongly significant (p<0.001). **But** a rolling
z-score mechanically reverts even for a pure random walk (the rolling mean chases the level). Subtracting that
mechanical floor:

| Pairs | excess reversion over random-walk floor (24h/48h/72h) |
|---|---|
| Cleanly-selected pairs | **+0.50 / +0.44 / +0.43 z** (matches a genuinely stationary half-life≈48h spread) |
| Correlation-fallback pairs | +0.14 / +0.15 / +0.18 z (barely above the mechanical floor) |

So there **is** a real, modest mean-reversion effect (~+0.45 z excess) for well-chosen pairs — but the project
does **not** have a genuinely cointegrated universe to select from, and pair-selection quality (not cointegration
p-values) is what matters.

## Recommendations
1. **Retract / revise RESULTS.md's cointegration finding.** The "99.7% OOS cointegration" should not appear in
   the paper; report instead the placebo-validated result that genuine OOS cointegration is ≈ chance (2–3%).
2. **Do not select pairs on the Kalman ADF p-value** — it is uninformative. Select on out-of-sample reversion
   speed / Ornstein–Uhlenbeck half-life or excess reversion over the rolling-z mechanical floor.
3. **Frame the real result honestly:** a modest, genuine mean-reversion effect exists for cleanly-selected
   pairs; the universe is not cointegrated; magnitudes must be reported net of the rolling-z artifact.
4. Independently reproduce the placebo before any public retraction. Scripts: `scratch/audit_part1.py`,
   `audit_part1b.py`, `audit_part2.py`; clean pairs in `scratch/clean_coint_pairs.csv`.
