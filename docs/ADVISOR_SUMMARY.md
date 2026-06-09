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

3. **…and there is a genuine market-neutral, diversified mean-reversion effect — but a *modest* one
   (~1 Sharpe, not the ~2.5 first reported), only harvestable as a patient never-stop book.** Removing the
   stop-loss flips a losing backtest (−2.25 Sharpe) to a winner; the effect is market-neutral (β ≈ −0.06) and
   diversified. It is **real and not a code artifact** — a from-zero, clean-room re-derivation by **four
   independent implementations** (one of mine, three by separate subagents, none touching our pipeline)
   reproduces it, finds no look-ahead, and confirms the stop loses on both venues. **But its magnitude was
   over-stated and the honest number is ~1.0:** the four independents put naive Binance at ~2.0–2.2 (not 2.5);
   even the *monthly* Sharpe is serial-correlation-inflated on Binance (AC(1)=+0.48 — 35-day holds span
   months), so a **HAC correction drops Binance 2.18→~1.4**; and on the **independent venue (Coinbase) it is
   ~0.85–0.90** (HAC-stable). So the **HAC-and-venue-honest figure is ~0.85–1.4, centered ~1.0**, and the edge
   is recency/pump-concentrated. Caveats: (a) **selection-sensitive** — a deflated-Sharpe correction survives
   the no-stop *family* but fails the whole stop-vs-no-stop search (and the HAC-corrected Sharpe flips it
   toward fail); (b) **survivorship robustness rests on the selector *avoiding* dead coins, not withstanding
   them** (in the point-in-time test, LUNA is
   unselected in its crash window; dead coins contribute ~0 PnL) — the real risk is the per-pair tail (a held
   LUNA pair = −100%/leg) — but **controllable** with a structural-break circuit breaker (caps the DD at ~no
   Sharpe cost, §deployability); (c) its "independent" supports (t=3.65, ρ=0.46,
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

- **Solid:** the two artifacts (both placebo-proven, and *independently re-confirmed clean-room* — Kalman
  screen passes random walks at 100% vs 5%; rolling-z self-reverts on pure noise), microstructure
  near-efficiency (≈12 tests, re-validated across all 12 months of 2024), the execution null, and that the
  reversion effect is real, market-neutral, stop-sensitive, and **replicates on an independent exchange
  (Coinbase)** AND under **four independent from-scratch re-implementations** — so it is *not* Binance
  overfitting or a code artifact (no look-ahead found by any).
- **Caveated:** the reversion *effect* is genuine but **modest — the honest HAC-and-venue-corrected figure is
  ~1.0 monthly Sharpe** (~0.85 Coinbase to ~1.4 Binance), NOT the ~2.5 first reported: the four independents
  put naive Binance at ~2.0–2.2, and even the *monthly* Sharpe is serial-correlation-inflated (Binance
  AC(1)=+0.48 → HAC ~1.4; Coinbase ~0.9, HAC-stable); the edge is recency/pump-concentrated. Around it: (i)
  **selection-sensitive** — a deflated-Sharpe correction *survives* the no-stop-family trial set but *fails*
  the whole stop-vs-no-stop search (and the HAC Sharpe flips it toward fail); (ii) **survivorship robustness
  rests on the selector *avoiding* dead coins, not
  withstanding them** (a held LUNA pair = −100%/leg) — but this is **controllable**: a structural-break
  circuit breaker (halt a pair after a >50% adverse leg move) caps the tail (worst −282%→−131%, maxDD
  −40%→−30%) at ~no Sharpe cost, whereas a *spread* stop kills the effect; (iii) the within-Binance supports
  (t=3.65, ρ=0.46, market-neutrality) are **correlated** (same data/selection); DDs leverage-equivalent.

## Suggested next steps

1. **Retract the 99.7% cointegration claim** and independently reproduce the Kalman placebo before any
   external use of the phase-1 results.
2. The reversion effect is real (it **replicates cross-exchange on Coinbase**, now done) but **modest
   (~1.0 HAC-and-venue-robust Sharpe) and selection-sensitive** — before pursuing it, it needs a **pre-registered,
   deflated-Sharpe-aware** test of a single fixed config (wide 40+ pair, no-stop, market-neutral) with an
   explicit delisting/structural-break control. The delisted-coin survivorship check is also done (Task 2).
3. Treat the microstructure chapter as closed for alpha; its only real-world value is the ~20% cheaper
   cost estimate from walking the book. The pre-registered H1 (institutional flow → reversion) was run
   once and **fails / is near-powerless** on the cleanly-cointegrated universe — see `phase2_l2/docs/H1_RESULT.md`.
