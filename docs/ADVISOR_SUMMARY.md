# Phase-2 (L2 microstructure) — advisor summary

*One-page distillation of the Level-2 / volume-as-information research. Full detail and every
number's script are in [`docs/L2_FINDINGS.md`](L2_FINDINGS.md); the cointegration correction is in
[`docs/CORRECTION_kalman_cointegration.md`](CORRECTION_kalman_cointegration.md); the chronological
work log is [`docs/l2_research_log.md`](l2_research_log.md).*

## Bottom line

We tested your L2 / "volume-as-information / infer-L3-from-L2 / retail-vs-institutional" ideas on
real Tardis Level-2 order-book + tick data (Binance spot, 50 USDT majors, event-level). The honest
result is a **rigorous, mostly-null paper with two methodological contributions and one real (but
patient-only) effect.** Order-flow information is genuine but lives at a **seconds** half-life and is
priced before any tradeable horizon — there is no microstructure alpha and not even a capturable
**execution** edge. Separately, the project's headline cointegration result turns out to be a
filtering **artifact**, but the underlying mean-reversion is a **real, market-neutral, diversified
alpha** — just slow, deep-drawdown, and only profitable if you *don't* use a stop.

## The five results

1. **The "99.7% out-of-sample cointegration" headline is an artifact — please retract it.** It comes
   from running an ADF test on a Kalman filter's own innovations, which are white by construction. A
   placebo that *cannot* be cointegrated (independent random walks, phase-randomized / block-shuffled
   series) passes the identical screen at the **same ~100% rate**. Genuine OOS cointegration is ≈ chance
   (**2–3%**). *(Methodological contribution #1; independent reproduction recommended before any external use.)*

2. **Mean-reversion is nonetheless genuinely selectable out-of-sample.** In-sample reversion speed
   predicts OOS reversion: Spearman **ρ = +0.46** across 9 disjoint walk-forward splits, top-minus-bottom
   quintile **+0.69z**, clean placebo. So there *is* a real, graded property to build on.

3. **…and there is a genuine market-neutral, diversified mean-reversion effect — but a *real-but-modest*
   one, only harvestable as a patient never-stop book.** Removing the stop-loss flips a losing backtest
   (−2.25 Sharpe) to a winner; the effect is market-neutral (β ≈ −0.06) and diversified (loads <2% on the
   top-10 statistical factors). On a **single honest combined test** — no-stop 40-pair on the top-50 **plus
   the delisted coins** (LUNA/UST/FTT, point-in-time), at monthly frequency — it is **ann ≈ 2.3 Sharpe**
   (≈2.5 on a literal point-in-time liquidity-ranked universe) at a ~30% drawdown, frequency-honest. Caveats
   to report honestly: (a) **selection-sensitive** — a deflated-Sharpe correction survives if the trial set
   is the no-stop *family* but fails if the whole stop-vs-no-stop search counts; (b) **survivorship robustness
   rests on the selector *avoiding* dead coins, not withstanding them** (in the point-in-time test, LUNA is
   unselected in its crash window; dead coins contribute ~0 PnL) — the real risk is the per-pair tail (a held
   LUNA pair = −100%/leg), needing a structural-break stop; (c) its "independent" supports (t=3.65, ρ=0.46,
   market-neutrality…) are **correlated** (same universe/selection/windows), not orthogonal. So: real but
   modest, not a clean alpha. **The catch:** a stop-loss of *any*
   width destroys it (it realizes losses on spreads
   that then revert), holds average **~35 days**, and ~⅓ of the return is a generic survivor-co-movement
   floor. So it is a real strategy for a patient, well-capitalized, market-neutral investor — **not** the
   tight-risk-managed hourly stat-arb the project originally specified, which loses.

4. **Order flow / microstructure is near-efficient at every tradeable horizon — your L3 ideas, measured.**
   Across OFI, VPIN, Kyle's-λ, book-OFI, cancellations, deep book, cross-asset lead-lag, institutional
   flow, **and a from-scratch execution simulation**, order-flow information is real but contemporaneous
   with a seconds half-life. Specifics on your ideas:
   - *Retail vs institutional:* trade size proxies information **per order** (large trades ≈ 2× permanent
     impact) but the edge is gone within seconds; at 15-min–1h, institutional flow predicts mild
     **reversal**, not continuation.
   - *Infer L3 from L2:* book-side OFI + the **cancellation channel** (81–85% of best-level liquidity
     pulls are cancels, invisible to trade data) genuinely describe price formation — but predict nothing
     at a tradeable horizon. **Hidden/iceberg liquidity** (executed volume exceeding displayed depth) is
     measurable (~2–3.5% of volume; BTC ≫ thin alts) but carries no edge.
   - *Execution value:* we *measured* whether the signal helps order placement (18,432 simulated orders).
     It does **not** — aggressive crossing is cheapest for majors, and the signal's correlation with the
     per-order post-vs-cross advantage is **+0.06** (noise). The value exists in principle (a perfect-
     foresight oracle saves ~1.6 bps) but no contemporaneous L2 feature forecasts it.

5. **Realistic L2 execution costs are ~17–23% below the flat 5 bps assumption** for liquid majors (we
   walk the real book). Prior backtests were conservative — but cheaper costs don't rescue the strategy.

## Methodological contribution #2 — the rolling-z artifact

A rolling z-score mechanically mean-reverts **even on a pure random walk** (the rolling mean chases the
level), manufacturing a spectacular fake edge (+4.75 Sharpe — but random pairs and phase-randomized
noise reproduce +2.2/+2.4 under the same rule). Every reversion number we report is net of this
mechanical floor and validated against placebos.

## What's solid vs what's caveated

- **Solid:** the two artifacts (both placebo-proven), microstructure near-efficiency (≈12 independent
  tests, now re-validated across **all 12 months of 2024** incl. the Aug-5 crash — the findings replicate
  in every regime), the execution null, and that the reversion effect is real, market-neutral, and
  stop-sensitive (the backtest engine was audited clean — no look-ahead/PnL bug).
- **Caveated:** the reversion *effect* is genuine but **modest**, so report it conservatively. The honest
  single number is the **combined test: ann ≈ 2.3 Sharpe** (no-stop 40-pair, top-50 + delisted, point-in-time,
  monthly). Around it: (i) **selection-sensitive** — a deflated-Sharpe correction *survives* if the trial set
  is the no-stop family but *fails* the whole stop-vs-no-stop search, so significance depends on framing;
  (ii) **survivorship** is now *in* the combined number (~10% dent vs the no-delisted top-50), and a live book
  still needs an explicit delisting/structural-break control (forcing a LUNA pair through May-2022 = a total
  leg loss); (iii) the supports (t=3.65, ρ=0.46, market-neutrality) are **correlated** (same data/selection),
  not independent; drawdowns are leverage-equivalent.

## Suggested next steps

1. **Retract the 99.7% cointegration claim** and independently reproduce the Kalman placebo before any
   external use of the phase-1 results.
2. The reversion effect is real but **modest and selection-/survivorship-fragile** — before pursuing it,
   it needs a **pre-registered, deflated-Sharpe-aware** test of a single fixed config (wide 40+ pair,
   no-stop, market-neutral) with an explicit delisting/structural-break control, on a true point-in-time
   universe. The delisted-coin survivorship check is now **done** (Task 2: weakens but survives).
3. Treat the microstructure chapter as closed for alpha; its only real-world value is the ~20% cheaper
   cost estimate from walking the book. The pre-registered H1 (institutional flow → reversion) was run
   once and **fails / is near-powerless** on the cleanly-cointegrated universe — see `phase2_l2/docs/H1_RESULT.md`.
