# L2 microstructure research log

Running log for the self-paced research loop. Newest iteration on top. Each
entry: what was done, what was found, what I think I'm missing, and the plan for
the next iteration.

---

## Iteration 7 — the bottom line: reversion is genuine & selectable OOS, but economically untradeable

Two complementary tests answer the project's central question. They look opposed but reconcile cleanly: the effect is **statistically real**
and **economically dead**.

### A. Does mean-reversion persist out-of-sample? **YES — it's selectable (positive, defensible finding).**
Strict OOS persistence test, all 1,225 pairs, **9 disjoint walk-forward splits** (6mo train → next 3mo test), static OLS hedge, every number net
of the rolling-z mechanical floor:
- In-sample reversion metrics **predict OOS reversion**: Spearman ρ(train OU-kappa, OOS excess reversion) = **+0.46** [+0.37,+0.54], p<0.001,
  **positive in every one of the 9 splits**. Variance-ratio and in-sample-excess agree (ρ≈±0.46).
- **Top-quintile vs bottom-quintile OOS excess reversion = +0.69z** [+0.58,+0.81], p<0.001 (top ≈+0.88z ≈ 2× universe avg, ~5× bottom).
- Placebo: random quintile spread = +0.0001z; shuffling the train→OOS link collapses ρ to −0.002. Floor-subtraction isn't manufacturing it
  (top/bottom floors identical 0.986/0.984); train kappa even predicts OOS kappa (+0.47). Scripts: `scratch/persistence_test.py`, `persistence_robust.py`.
- **So, unlike cointegration (artifact, ~chance), reversion is a graded, selectable, persistent property** — a real foundation. *But selectability ≠ profitability.*

### B. Honest net-of-cost walk-forward backtest — **NO edge; negative even gross.**
19 walk-forward splits, OU-half-life selection, static hedge, z-rule (enter|z|≥2 / exit|z|≤0.5 / stop|z|≥4), realistic costs, placebos:
- **Net Sharpe −2.25** [−3.21,−1.34] at ~30 bps round-trip; **−1.30** even at ~2 bps. **Gross Sharpe −1.15** — loses *before* costs, and is **worse than random-pair selection**.
- Why: selected pairs' train half-life (40–140h) **explodes to 200–4,000h OOS** — fast reversion doesn't persist *at tradeable speed*; spreads drift to
  new levels and hit the |z|=4 stop at a loss. The genuine OOS reversion is only **+0.155z/48h** (t≈7, real) — tiny, swamped by stop-outs/churn, and
  orders of magnitude below ~30 bps cost.
- The rolling-in-test-z variant looks spectacular (+4.75 Sharpe) **but is the mechanical artifact**: random pairs (+2.18) and **phase-randomized noise
  (+2.42)** reproduce it. Zero selection information — must not be reported as an edge. Scripts: `scratch/wf_backtest.py`, `wf_sanity.py`, `wf_diag.py`.

### Reconciliation & the project's honest bottom line
A (selectability, in z-units over 48h) and B (traded P&L) are **both right**: reversion is a real, selectable *statistical* property, but its magnitude
(+0.155–0.45z) is too small and too slow/unreliable to survive a practical entry/exit/stop rule and two-leg costs. The asymmetric stop turns the thin
mean-reversion into a net loss because the minority of |z|>2 spreads that keep diverging dominate. **The strategy, as specified, loses money.**

### The paper is now fully formed (and it's a good, honest one)
1. **The "99.7% cointegration" headline is a Kalman whitening artifact** (iter 6; `docs/CORRECTION_kalman_cointegration.md`) — methodological contribution.
2. **Mean-reversion is genuinely selectable OOS** (ρ=0.46 across 9 splits, clean placebo) — a real positive result.
3. **…but economically untradeable** after realistic costs / a practical rule (net Sharpe −2.25, gross −1.15) — and the apparent rolling-z "edge" is a
   second artifact (noise earns +2.4 Sharpe under it).
4. **Microstructure / order flow is near-efficient at tradeable horizons** (seconds half-life; value is execution/price-formation, not signal) — iters 1–6.
5. **Realistic L2 costs are ~17–23% below the flat assumption** for liquid majors — but don't rescue the strategy.
Net: a rigorous **cautionary-tale / null paper** with two genuine methodological contributions (the Kalman-innovation and rolling-z artifacts, both
demonstrated with placebos) and one real-but-unprofitable statistical effect. This is more publishable, and more honest, than the original optimistic framing.

### What am I missing? / the loop has reached its scientific conclusion → consolidate
The core questions are now answered; further signal-hunting is likely to keep hitting the near-efficiency wall. Highest-value remaining work is
**consolidation**, not new tests:
- **Distill a single paper-ready findings document** from this log (the 5 results above), with the placebo-validated artifacts front-and-center for the advisor/collaborator. This is next iteration's main task.
- Two optional robustness items if time: (i) re-estimate with a *rolling* hedge (B froze beta; real trading re-estimates) to confirm it doesn't rescue P&L; (ii) confirm the +0.155z reversion is/ isn't capturable by a slower, no-stop, vol-targeted variant (likely still sub-cost, but worth one clean check).
- Stop adding microstructure signals — that line is exhausted and consistently near-efficient.

### Plan for next iteration (prioritized)
1. **Write the paper-ready findings doc** (`docs/L2_FINDINGS.md`): the corrected cointegration result, selectable-but-untradeable reversion, microstructure efficiency, L2 costs — with the key tables and the two artifact demonstrations.
2. Optional clean robustness: rolling-hedge backtest + slow/no-stop vol-targeted reversion variant (does *any* implementation clear cost? expected: no).
3. Make sure the advisor sees `CORRECTION_kalman_cointegration.md`.

---

## Iteration 6 — ⚠️ MAJOR: the "99.7% cointegration" pillar is a Kalman artifact; reversion is real but modest

This is the most consequential iteration: chasing the iter-5 power problem to its root exposed a **false headline result** in the
existing analysis, exactly the kind of flaw worth finding. Full write-up + recommendations: **`docs/CORRECTION_kalman_cointegration.md`**.

### A. Cointegration foundation audit — **RESULTS.md Finding 1 is an artifact (retract)**
- The Kalman screen ADF-tests the filter's **one-step innovations**, which are **white by construction** (`q_beta>0` ⇒ the time-varying
  beta tracks any relationship). Testing stationarity of a filter's own innovations is circular.
- **Placebo proof:** identical screen on independent random walks / phase-randomized surrogates / block-shuffled legs **all pass at 100%**
  (p<0.05), the *same* rate as real pairs (100% / 96.4% at p<0.001). The Kalman ADF carries **zero** cointegration information.
- **Clean tests** (Engle–Granger, static OLS+ADF), OOS: genuine cointegration rate = **2.1–3.2%** — at/below the 5% null floor.
  Essentially **none** of the top-50 pairs are genuinely cointegrated OOS. (RESULTS.md's own "static 5.4%" was already ≈ the null floor.)
- **The "1219-pair, 99.7% OOS cointegration" — the project's self-described strongest finding — is false.** Any pipeline step that selected on
  `kalman_selected_pairs.parquet`'s ADF p-value selected on noise. Scripts: `scratch/audit_part1.py`, `audit_part1b.py`.

### B. Reversion premise WITH power — real but modest, and inflated by a rolling-z artifact
- Full multi-year history (24,942 bars, ~4,000 |z|>2 events, pair-clustered + block bootstrap): |z|>2 reversion is strongly significant
  (p<0.001) — this **resolves the iter-5 power complaint** (the L2 log's null was an underpowered single quarter, not a true null).
- **BUT** a rolling z-score mechanically reverts even for a random walk. Net of that mechanical floor: cleanly-selected pairs retain **+0.45z**
  genuine excess reversion (≈ a real stationary half-life≈48h spread); correlation-fallback pairs retain only **+0.15z** (barely above the floor).
  So a genuine, modest effect exists for good pairs — but there's no cointegrated universe to draw from. Script: `scratch/audit_part2.py`.

### C. Deep-book (levels 1–10) probe — another near-efficient null
- Deep imbalance and book-slope carry **strictly less** R² than top-of-book level-1 imbalance at every horizon; "deep-only" (levels 2–10) is
  weakest; depth-weighted fair-value tilt is predictively ~noise. Incremental R² of all deep features over top-of-book: +0.31% (1s) → +0.011%
  (300s), decaying even faster than the base signal. No slower signal hides in the deeper book. Not tradeable. Script: `scratch/deep_book_probe.py`.

### Where this leaves the project (honest, and actually a stronger paper)
The microstructure null-results were partly a *detection-power* story; with the full history the base reversion effect is genuine but modest, and
**the cointegration pillar was an artifact.** The honest narrative is now sharper and more defensible than the original optimistic one:
1. Crypto majors are microstructure-efficient at tradeable horizons; order-flow information has a seconds half-life (iters 1–6).
2. The one genuine microstructure positive is contemporaneous **price formation / execution** (book-OFI + the cancellation channel), not signal.
3. **The cointegration claim does not hold** (Kalman whitening artifact); there is a **modest real mean-reversion** effect (~+0.45z excess) for
   carefully-selected pairs, reported net of the rolling-z mechanical baseline.

### What am I missing? / next iteration
- If clean cointegration is at chance, **how should pairs actually be selected?** Test OOS selection on **OU half-life / reversion-speed** (not
  ADF p-value) and on **excess-reversion-over-mechanical-floor**, and check whether *that* selection yields pairs whose reversion survives OOS + costs.
- **Re-run the real backtest** with cleanly-selected pairs and the artifact-corrected reversion, using the realistic L2 costs (`src/l2_costs.py`),
  to get the honest net-of-cost edge (if any). This is the bottom-line number the paper needs.
- The Kalman-artifact correction **must reach the advisor/collaborator** (it changes the paper's headline). Independent reproduction recommended.
- Begin the paper-ready distillation around the corrected narrative.

### Plan for next iteration (prioritized)
1. **Pair selection done right:** OU half-life / excess-reversion selection, OOS, vs ADF; does it find pairs with cost-surviving reversion?
2. **Honest net-of-cost backtest** on cleanly-selected pairs with L2 costs — the bottom line.
3. Distill the corrected narrative into a paper section; make sure the cointegration correction is front-and-center for the collaborator.

---

## Iteration 5 — constructive angles close out; the binding constraint is now statistical POWER

Pivoted from alpha-hunting to two constructive uses (vol forecasting, entry-filter). Both came back honest-negative, but the
*reason* is the headline: the universe is too small/correlated to confirm even the strategy's own premise.

### A. Microstructure features for next-hour volatility vs HAR-RV (6 symbols, 90 days, OOS walk-forward) — **HAR-RV already wins**
- HAR-RV (Corsi 2009) is a strong benchmark: pooled **OOS R²≈0.51**, and as a sizing engine cuts realized-risk coefficient-of-variation **−41%** vs flat.
- 11-feature "kitchen-sink" micro set: IS incremental +1.45% but **OOS −0.95%** (overfits, *degrades* vol-targeting by 4%).
- A *parsimonious* 4-feature set (trade count, quote-update intensity, mean spread, jump proxy) flips OOS to **+1.38%** — small, positive, but only ~2–3% of HAR's OOS R², concentrated in BTC/ETH, and not worth the extra position churn.
- Verdict: microstructure does **not** meaningfully beat HAR-RV for next-hour vol OOS; the autoregressive persistence of RV already encodes the intensity signals. Scripts: `scratch/har_vol_build.py`, `scratch/har_vol_regress.py`.

### B. Does flow condition reversion at spread extremes? (8 pairs, ~240 non-overlapping |z|>2 events, cluster-bootstrap) — **sign-consistent but underpowered**
- The hypothesis ("don't fade when flow still pushes the divergence") points the **right way** at every horizon/threshold and *grows with z* (informed-vs-noise intuition) — oppose-flow extremes revert more than confirm-flow extremes.
- But **0 of 9 specifications are statistically significant** once SEs cluster by pair; the pooled continuous correlation is ≈0; per-pair signs flip (3 of 8 opposite).
- **Critical:** even the *unconditional* reversion edge fails pair-clustered significance (mean +0.17z, 95% CI **[−0.026, +0.387]**). The filter lifts gross win-rate 57.7%→61.2% and ~doubles mean PnL, but discards 65% of trades, isn't distinguishable from luck, and the gross gain (~few bps) is < costs. Scripts: `scratch/extreme_flow_*.py`.

### The real lesson of iter 5: POWER is the binding constraint (and it implicates the foundation)
Only ~240 independent extreme events across **8 correlated pairs**, one quarter → the effective sample is far smaller than it looks. At that
power, *no* small edge (micro vol gain, flow filter) can be confirmed — and, tellingly, **even the strategy's core mean-reversion premise isn't
significant once SEs are clustered properly.** This reframes the whole project: the repeated "near-efficient / no edge" results are partly a
*detection-power* statement, and the long-standing **correlation-fallback (non-cointegrated) pair selection** is the root cause — it gives few,
correlated, statistically-fragile events. We've been testing fancy signals on a foundation that can't support significance tests.

### What am I missing? / the pivot this forces
- Stop adding signals; **fix the foundation and the power.** We now have **170 book / 111 trade days** ingested (158 GB) — far more than the
  one quarter most tests used. The right next move is to (1) run the proper **Kalman/ADF cointegration screen on the full L2 window** (RESULTS.md
  claims 99.7% OOS cointegration — never used in the L2 analyses), select genuinely cointegrated pairs, and (2) **re-test the base reversion premise
  with many more events and pair-clustered/block-bootstrap SEs across the full sample.** If the premise survives with power, *then* re-test the flow
  filter on it; if it doesn't, that is itself the paper's central, honest result.
- Higher-frequency extremes (5–15 min |z| crossings) would also multiply the event count — a cheaper power boost to validate the filter.
- The microstructure chapter is essentially complete and **paper-ready**; begin distilling `l2_research_log.md` into a clean findings section.

### Plan for next iteration (prioritized)
1. **Cointegration + power:** Kalman/ADF screen on the full L2 window → cointegrated pairs; re-test base reversion (and the flow filter) with proper clustered SEs and the full event set. Does the premise hold with power?
2. Higher-frequency (sub-hourly) extreme events to multiply N for the filter test.
3. Begin the paper-ready "what L2 taught us" distillation (honest near-efficiency on alpha + positive on price-formation/execution + the power caveat).

---

## Iteration 4 — the alpha hunt fails twice (cleanly); the near-efficiency thesis is now robust

Took the two remaining shots at tradeable-horizon directional alpha, each designed around its classic trap.

### A. Cross-asset lead-lag BTC→alts (6 symbols; staleness control) — **null (artifact + arbitraged)**
- Huge **contemporaneous** co-movement (r(0) 0.44–0.61 at 1s, rising to 0.64–0.81 at 60s as noise averages out), but **no exploitable directional lead**.
- The decisive control worked: the 1–5s "BTC leads alt" cross-correlation shoulder **collapses at 30s/60s sampling** (where alts aren't stale — they're stale 50–70% of seconds at 1s). That is the textbook **spurious-lead-lag-from-stale-prices** signature, not causation.
- Predictive R² 0.3–1.4%, <1 bp per +1 SD vs ~10 bps taker cost. The only strong signal (BTC return → alt at 5s, t=17) is a few-seconds, already-arbitraged effect. Spread lead (AVAX−ETH BTC-flow t≈2.2) is <2 bps on a two-leg trade. Not tradeable. Script: `scratch/leadlag_xasset.py`.

### B. Institutional net flow → 15-min / 1-h returns (BTC/ETH/SOL, 31 days; HAC) — **null, and the sign is REVERSAL**
The advisor's "volume-as-institutional-information," at a tradeable horizon, building on iter-2's ~2× permanent impact for large orders.
- Institutional (>$10k) = 35–46% of volume but 2–3% of orders — a meaningful split.
- **Incremental R² of institutional flow over (total flow + momentum): +0.036% (15m), +0.032% (1h) — essentially zero.**
- Where it loads "significantly" (pooled 15m t=−2.63; SOL 1h t=−2.38) the sign is **negative**: institutional buying predicts *lower* future returns — short-horizon **price-pressure mean-reversion**, the opposite of informed continuation. The informed-vs-noise test fails (institutional doesn't load + while retail loads −; dominant effect is *negative* momentum = reversal).
- Per-symbol 1h "hits" are overfitting (BTC +1.85t at 12d → +0.89t at 31d; SOL persists − ; incoherent across same-class assets).
- Reconciles iter-2: large orders *do* carry ~2× permanent impact, but it's impounded **within seconds**; by 15m–1h it's fully priced, leaving mean-reverting noise. Script: `scratch/inst_flow_horizon2.py`.

### The thesis is now robust (4 iterations, ~6 independent tests, all pointing the same way)
**Crypto majors are microstructure-efficient at every tradeable horizon.** Order flow (trade + book) and large-order footprint explain
*contemporaneous* price formation extremely well and carry genuine **permanent** information — but that information has a **seconds-scale
half-life**. By any horizon you could actually trade against ~10–20 bps costs, the directional predictability is gone or has reverted to
mean-reverting price-pressure noise. Every natural "volume-as-alpha" hypothesis we could construct has now been falsified with the right control:
seconds-OFI (dead after cost), spread-OFI (dead), VPIN regime-filter (dead + inverted), cross-asset lead-lag (staleness + arbitraged), institutional
flow at 15m–1h (reversal, not information). The genuine, robust positive findings are about **price formation / execution, not signal**: book-OFI
dominates contemporaneous moves (+0.12–0.16 incr R²), and the cancellation channel (>80% of best-level liquidity withdrawal, invisible to trade data)
carries directional info contemporaneously.

### What am I missing? / where to take it next
- I've been hunting **return** predictability and consistently finding near-efficiency. Two constructive directions remain that don't require beating
  efficiency:
  1. **Execution value (the one place the signal has real $):** quantify how much an OFI/cancellation-aware passive placement saves vs the L2 book-walk
     cost we already model in `src/l2_costs.py`. This converts the robust contemporaneous signal into a number on the strategy's bottom line.
  2. **Volatility (risk), not direction:** realized vol is highly predictable (iter-3, R²≈0.30). Do microstructure features (book-OFI volatility,
     cancellation intensity, trade intensity, quoted-spread dynamics) improve *volatility* forecasting *incrementally* over realized vol? Better vol
     forecasts → better position sizing → better Sharpe **without** directional alpha. This is the most-overlooked, still-plausible win.
- **Consolidation:** the microstructure story has reached a natural conclusion on the alpha question and is now paper-ready — next loop should also
  begin distilling `l2_research_log.md` into a clean "what L2 taught us" findings section (honest null on alpha + positive on price-formation/execution).
- Robustness watchpoint: still mostly Jan–Feb 2024, BTC/ETH/SOL. The 31-day institutional run helps; a volatile later month would strengthen generality.

### Plan for next iteration (prioritized)
1. **Execution-value experiment:** simulate OFI/cancellation-aware passive vs aggressive placement on real book data; measure bps saved vs `l2_costs` book-walk; feed into a with/without backtest.
2. **Microstructure → volatility forecasting:** incremental-over-realized-vol test for vol (not return); if it works, wire a vol feature into sizing.
3. Begin the paper-ready findings distillation.

---

## Iteration 3 — VPIN regime-filter is dead (and inverted); book-side flow ("L3 from L2") genuinely adds info

Tested the two tradeable-horizon ideas from iter 2, each designed around its failure mode (redundancy / incrementality).

### A. VPIN as a regime / trade-filter (8 pairs, 40 days, volume-clock VPIN, HAC) — **dead, instructively**
- VPIN is **not** redundant with lagged realized vol (incremental R²≈**+0.012**, significant in 7/8 pairs) — but that increment is
  ~10× smaller than what the spread's own lagged realized vol already provides (R²≈0.30).
- **The sign is backwards from the toxicity thesis:** high VPIN predicts *lower* next-hour spread vol and *lower* decoupling
  (high-VPIN hrs: 1.12 bps vs 1.29; decoupling 0.080 vs 0.122). Mechanically, volume-clock |buy−sell|/vol is highest when flow is
  one-directional/orderly (trending) and lowest in balanced two-way churn — and it's the *churn/disagreement* regime that decouples a pair.
  So VPIN is a **directional-consensus meter, not a danger meter.** "Skip high-VPIN hours" would *increase* drawdown.
- Verdict: VPIN adds nothing exploitable beyond realized vol, and points the opposite way from the hypothesis. Filter idea killed.
  Script: `scratch/vpin_spread_vol.py`.

### B. Book-OFI / cancellation flow from raw L2 snapshots (Cont–Kukanov–Stoikov; BTC, ETH; HAC) — **the best result yet**
The advisor's "infer L3 from L2," done concretely: reconstruct book-side order flow + cancellations from per-update `book_snapshot_25`.
- **Contemporaneous price formation (the CKS finding, replicated on crypto):** book-OFI explains within-second mid moves much better
  than trade flow. Joint R² 0.29 (BTC)/0.32 (ETH); **book-OFI incremental over trade-OFI = +0.125/+0.164** (larger than trade's increment
  over book). Book/cancel/add flow describes price formation better than trades alone.
- **Predictive:** book-OFI adds a small but strongly significant increment at 1s (+0.004 R²), decaying to ~0 by 30s — a fast, fast-decaying
  complement; trade-OFI persists longer.
- **Cancellation channel (invisible to trade data):** **81–85% of best-level size reductions are CANCELS, not trades.** Trailing cancel-imbalance
  adds a further +0.003–0.007 R² with the correct sign (more bid cancels → price down; t up to −38). This is genuinely new information that pure
  trade-flow analysis (and the whole prior pipeline) never saw. Scripts: `scratch/book_ofi_incremental.py`, `scratch/book_ofi_cancel_stretch.py`.
- Verdict: book-side flow is **not** redundant with trades — it dominates contemporaneous price formation and the cancellation channel carries
  directional info. Still sub-10s and low R² (≤2–3%, near-efficient mids), so incremental microstructure signal, not standalone alpha.

### Cumulative picture after 3 iterations (this is becoming the paper's microstructure story)
1. The "volume-as-information null" was a **timescale artifact** (iter 1).
2. Order flow is informative at **seconds** (single-asset & spread) but **economically dead standalone** vs costs (iter 2).
3. Trade **size** proxies information **per order, not per dollar**; institutions minimize footprint (iter 2).
4. **VPIN toxicity** doesn't work as a regime filter and its sign is inverted (iter 3).
5. **Book-side flow + cancellations ("L3 from L2") genuinely add information** over trades, esp. contemporaneously (iter 3).
→ **Overarching thesis forming:** crypto 1s mids are near-efficient; trade+book order flow explains *contemporaneous* price formation well
  but offers only fleeting (<10s), tiny (R²~0.1–0.4%) *forecast* power. The microstructure payoff is in **execution** (cancellation-aware queue
  placement, which leg to post vs lift), **not** a directional signal for an hourly pairs strategy.

### What am I missing? / next iteration
- I keep finding "informative contemporaneously, dead for forecasting." The one **untested shot at tradeable-horizon alpha** is
  **cross-asset lead-lag**: does BTC order flow / return *lead* alt returns (and thus the spread) at a **1–5 minute** horizon (slower, possibly
  cost-surviving)? Crypto is known for BTC leading alts; the phase-1 proposal flagged lead-lag. This is the most promising remaining direction.
- **Quantify the execution value** of book-OFI/cancellation signal: how much could cancellation-aware posting save vs the L2 book-walk cost we
  already model (`src/l2_costs.py`)? Ties the microstructure work to a real $ number in the backtester.
- **Robustness:** everything is Jan–Feb 2024, BTC/ETH-heavy. Re-test the headline (book-OFI increment, seconds-decay) on later months / more
  symbols now that ~152 GB is cached, to make sure it's not a sample artifact.

### Plan for next iteration (prioritized)
1. **Cross-asset lead-lag order flow:** test whether BTC (and ETH) signed flow / return at t predicts alt and pair-spread returns at 1s–5min, HAC,
   impact-decay curve; is any horizon both significant AND large enough to beat costs? This is the alpha hunt.
2. Execution-value estimate of cancellation-aware placement vs `l2_costs` book-walk.
3. Out-of-sample robustness of the book-OFI result on a later month.

---

## Iteration 2 — order flow is real but economically dead on the spread; size *does* proxy information (per order)

This iteration asked the question I was missing in iter 1: **permanent (information) vs transient
(mechanical/liquidity) impact**, and tested the signal on the actual *pair spread* at the right horizon.

### A. Permanent-vs-transient impact by trade size (BTC, ETH; event-time response functions, bootstrap SEs)
Resolves the iter-1 confound where a naive 1s-bar correlation suggested "small flow is more informative."
- Class shares: institutional (>$10k) = **6% of orders but ~75% of volume**; retail (<$1k) = 77% of orders, ~5% of volume.
- **Per-order permanent impact (R at 300s asymptote): institutional ≈1.58 bps (BTC) / 2.07 (ETH) ≈ 2× retail (~0.76 bps).** Monotone in size, separation many bootstrap-SEs wide.
- Institutional orders show a **transient hump** (peak ~60s, then 13–20% decay back) — classic temporary
  liquidity impact; retail/mid are ≈100% permanent (too small to create a dislocation that relaxes).
- **The iter-1 "small flow more informative" was an artifact:** small trades' moves are fully permanent only
  because they're too tiny to decay, not because they carry more news. Per-dollar the ranking inverts (retail
  ~24,000 vs institutional ~74 bps/$1M) but that excess is **mechanical** (tick/bounce on a tiny denominator), not information.
- **Answer to the advisor's retail-vs-institutional question:** trade size **is** a proxy for information *per order*
  (large = more permanent impact); per *dollar*, institutions are far cheaper (they split/time to minimize footprint).
  Script: `scratch/impact_decomp.py`.

### B. Horizon-matched two-leg OFI → pair-spread returns (5 pairs, microprice spread, HAC SEs)
The strategy-relevant redo of "volume-as-information."
- **Reverses the prior null:** two-leg signed OFI predicts short-horizon spread returns with the right sign and
  overwhelming significance at **1–10s** (BTC/ETH 1s HAC-t≈31, alive to 60s; thinner pairs die by 10–30s). Mirrors the single-asset decay.
- **But economically dead standalone:** peak R²≈**0.001** (BTC/ETH 5s), ~0.0001 elsewhere; a 1-unit OFI shock ⇒ ~0.02 bps spread move at 10s,
  vs ~20 bps round-trip two-leg taker cost — the signal is **2–3 orders of magnitude smaller than the cost of crossing**.
  Scattered "significant" 300s coefficients are slow-drift artifacts (vanish in the mid-spread robustness check).
  Script: `scratch/pair_ofi_spread.py`.

### Synthesized conclusion (advisor-facing)
1. Order flow **is** genuine microstructure information — at **seconds**, both single-asset and on the spread. The prior
   "volume-as-information is null" was purely a timescale artifact.
2. **It is not standalone alpha for the pairs strategy** — the directional spread edge (R²~0.1%) is dwarfed by costs;
   at best it's an **execution/queue-placement tilt** inside a strategy whose alpha comes from elsewhere.
3. **Size proxies information per order, not per dollar.** Useful for a retail/institutional split, but "big trade = informed
   alpha" doesn't hold once you normalize and separate transient from permanent impact.

### What am I missing? (next-iteration ideas)
- **The directional seconds-edge is dead, but is there a *tradeable-horizon* use of volume-as-information?** VPIN/order-flow
  *toxicity* is documented to predict short-horizon **volatility/adverse selection**, not return direction. Hypothesis worth
  testing: does hourly/daily order-flow toxicity predict **spread blow-ups / pair decoupling** — i.e. a **regime/trade-filter**
  (when *not* to hold the spread) that could actually improve the hourly strategy? This reconnects microstructure to the phase-1 goal.
- **L3-from-L2 (the advisor's specific idea) still not done.** Next: reconstruct order **add/cancel/trade events** from
  `incremental_book_L2` deltas, build **book-side** OFI (Cont–Kukanov–Stoikov uses book events, not just trades),
  cancellation rates, and queue-depletion/fill-probability features — a richer signal than trade flow alone. True queue
  position & icebergs are unrecoverable (proxy: trade volume > displayed size = hidden execution).
- Permanent impact via R(300s) has wide SEs; a Hasbrouck (1991) VAR long-run impulse response would be a more robust permanent-impact estimate.

### Plan for next iteration (prioritized)
1. **VPIN-as-regime-filter:** compute hourly VPIN/toxicity per pair, test if it predicts next-hour spread volatility /
   |spread change| / decoupling; if yes, add as a *trade filter* to the backtest and measure Sharpe with/without. (Tradeable horizon!)
2. **Start L3 reconstruction:** download a few days of `incremental_book_L2`, build the event classifier (add/cancel/trade),
   compute book-OFI + cancellation features, test informativeness at seconds horizon vs trade-OFI.
3. Robustify permanent-impact with a small Hasbrouck VAR.

---

## Iteration 1 — the prior "null" was an artifact; order flow is a *seconds*-scale signal

### The major flaw I found in the last analysis (`docs/l2_analysis.md`)
**Timescale mismatch — fatal.** Order-flow signals (OFI, VPIN, Kyle's λ) live on a
seconds-to-minutes horizon, but `src/microstructure_features.py` aggregates them to
**hourly** bars and `src/features.py` (`DatasetConfig.horizon=24`) tests them against a
**24-hour-ahead** spread-change label. That is a ~1,400× horizon mismatch: the intra-hour
imbalance that carries the information is summed/averaged away on the feature side, and the
label is swamped by a day of unrelated price action. The test has ~zero statistical power
for the hypothesis it claimed to refute. **"Volume-as-information doesn't help" was a
methodology artifact, not a finding.** (Independent audit reached the same conclusion.)

Secondary flaws (see audit): the 1s trade schema (`src/trades_store.py`) discards individual
trade sizes, so retail-vs-institutional flow was *never testable* in the main pipeline;
pairs were correlation-fallback (no cointegration passed), so the mean-reversion target is
itself shaky; the 49.8%→100% coverage flip means the reported null sits inside the noise band;
`max_drawdown` is a degenerate metric. The collaborator's `phase2_l2/` did this correctly
(1s bars, 60-bar window/horizon, **notional-bucketed** signed flow, pre-registered single eval)
— we should adopt its design, not re-report the hourly/24h null.

### What I verified empirically (BTCUSDT, ETHUSDT; 2024-01-02..01-11; ~864k 1s obs each)
Trailing-10s signed order flow → future mid log-return, **Newey-West (HAC) SEs** (the subagent's
naive OLS inflated t-stats on overlapping windows — corrected here):

| horizon | BTC β | BTC HAC-t | BTC R² | ETH HAC-t |
|---|---|---|---|---|
| 1s | 9.7e-7 | **8.0** | 0.0045 | **11.9** |
| 5s | 1.3e-6 | **2.7** | 0.0014 | **4.4** |
| 10s | 1.1e-6 | 1.3 | 0.0005 | **3.0** |
| 30s | 7.4e-7 | 0.5 | 0.0001 | 1.5 |
| 60s | 7.9e-7 | 0.3 | ~0 | 0.7 |
| 300s | −6.8e-6 | −1.4 | 0.0006 | −0.0 |

**Findings (honest, HAC-robust):**
1. Order flow **is** informative — but only at **1–10 seconds**; significance is gone by 30–60s.
   Contemporaneous price impact is huge (t>140, not overlapping → robust). Forecast R² is small
   (~0.5% at 1s) and ultra-short-lived.
2. The "5-minute mean-reversion/overshoot" the subagent flagged (t=−22.8) is **NOT robust** —
   it falls to t=−1.4 (BTC) / ~0 (ETH) under HAC. Do not claim it without block-bootstrap.
3. **Trade size is a poor proxy for "informed" flow.** Raw BTC trades (2024-01-02, 2.25M trades,
   $2.94B): >$10k trades = **48% of volume from 2.77% of trades**; <$1k "retail" = 8.3% of volume
   but 82% of count. Yet **small (≤$10k) signed flow correlates *more* with price** (contemp. 0.59
   vs 0.36; next-10s 0.235 vs 0.127) with ~3.5× higher per-dollar impact. This *contradicts* the
   naive "large trade = institutional = informed" hypothesis. ⚠️ Caveat: small-trade flow is denser
   per 1s bar, so the raw-correlation comparison may have a frequency confound — needs a
   frequency-matched / per-trade-impact control before it's a real result.

### What am I missing? (open questions for the next loop)
- **Strategy relevance:** I tested single-asset mid, not the **pair spread**. Does combined
  two-leg OFI predict short-horizon *spread* moves, and can a *seconds-scale* spread signal be
  turned into anything tradable given costs? A 1–10s edge of 0.5% R² likely dies after fees — so
  the honest question may be "does order flow help *execution/timing* of an hourly strategy,"
  not "does it predict the spread."
- **Right target:** use **microprice** (size-weighted mid, Stoikov 2018), not mid, to remove
  bid-ask bounce; the contemporaneous-vs-forecast split and impact-decay curve are the deliverable.
- **Size data is missing at the bar level:** `trades_store` must be extended to carry per-bucket
  signed volume (retail/mid/institutional) *before* 1s aggregation, like `phase2_l2/src/bars.py`.
- **L3-from-L2:** we have raw `incremental_book_L2` available to download; differencing level sizes
  + matching trade prints reconstructs add/cancel/trade events → queue depletion / fill-probability
  features. True queue position & icebergs are unrecoverable (proxy: anomalous fills where
  trade volume > displayed size).
- **Inference rigor:** every short-horizon regression needs block-bootstrap or HAC; overlapping
  24h labels in the main pipeline also need the HAC/DSR correction phase-1 used elsewhere.
- **Cointegration:** rerun the Kalman/ADF screen *on the L2 window* before any spread claim
  (RESULTS.md says Kalman finds cointegration OOS 99.7% — the L2 run inconsistently didn't use it).

### Plan for next iteration (prioritized)
1. **Extend `trades_store`** to emit per-notional-bucket signed volume (retail <$1k / mid / inst >$10k)
   from the raw per-trade csv.gz, re-ingest a few symbols, rebuild size-aware panels.
2. **Horizon-matched spread test:** build 1s pair-spread + two-leg OFI; regress future spread return
   at h∈{1s..5m} with microprice and HAC SEs; produce the impact-decay curve. This is the proper
   redo of the volume-as-information question.
3. **Frequency-controlled small-vs-large** impact test (per-trade, not per-1s-bar) to confirm/kill
   the "size isn't informativeness" result.
4. Later: L3 event reconstruction from `incremental_book_L2`; reconcile with `phase2_l2/`.

### Artifacts
- `scratch/ofi_horizons.py`, `scratch/raw_sizes.py` (exploratory), `scratch/verify_ofi.py` (HAC verification).
- Prior (flawed-horizon) results remain in `docs/l2_analysis.md` / `docs/l2_results/` — kept for the record;
  this log supersedes their interpretation.
