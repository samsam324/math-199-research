# L2 / microstructure research — consolidated findings

**Status: paper-ready synthesis of the phase-2 L2 research loop (iterations 1–8).**
This document distills `docs/l2_research_log.md` (8 iterations, ~20 independent tests on
real Tardis Level-2 order-book + tick-trade data) into the results that should go in the
paper. It supersedes the optimistic framing in `docs/RESULTS.md` and `docs/l2_analysis.md`
where they conflict; both are kept for the record. Every headline number below is backed
by a committed script in `scratch/` and was checked against a placebo or a robustness
control before being reported.

---

## Executive summary

The phase-2 question was: *does real L2 data — order-book depth, signed tick volume, the
advisor's "volume as information / retail-vs-institutional" and "infer L3 from L2" ideas —
turn this crypto pairs-trading strategy into something that works?* After eight iterations
the honest answer is a **rigorous, largely-null result with two genuine methodological
contributions and one real statistical effect that is monetizable only outside the
strategy's intended (hourly, stop-managed) design.** This is a more defensible — and more
publishable — paper than the original optimistic one.

The five results:

1. **The "99.7% out-of-sample cointegration" headline (RESULTS.md §1) is a statistical
   artifact, not a finding.** It comes from ADF-testing a Kalman filter's own one-step
   innovations, which are white by construction. A placebo that *cannot* be cointegrated
   passes the identical screen at the same ~100% rate. **Retract it.** Genuine OOS
   cointegration is ≈ chance (2–3%). *(Methodological contribution #1.)*

2. **Mean-reversion is nonetheless genuinely *selectable* out-of-sample.** Unlike
   cointegration, in-sample reversion speed predicts OOS reversion: Spearman
   ρ = +0.46 across 9 disjoint walk-forward splits, top-minus-bottom quintile +0.69z,
   with a clean placebo. So there *is* a real, graded, persistent statistical property to
   build on. *(A real positive result.)*

3. **…and its profitability hinges entirely on the stop/exit rule — the real finding.**
   With a conventional |z|=4 stop the strategy loses (net Sharpe −2.25), and the loss is
   *caused by the stop* (it realizes losses on spreads that later revert), not by costs.
   Remove the stop and hold to convergence and it earns +2.51 net Sharpe (robust to
   plausible delisting rates) — but only as a slow, multi-month exposure (median ~35-day
   holds; 78% of positions never converge within the quarter) with a −41% drawdown, ~⅓ of
   it generic survivor co-movement (random-pair placebo +0.89). So the premise is
   monetizable, but **not** as the hourly, stop-managed stat-arb the project specified —
   that version loses. *(Iteration 7's "loses even gross" headline was stop-specific and is
   corrected here — see §3.)*

4. **Microstructure / order flow is near-efficient at every tradeable horizon.** Across
   OFI, VPIN, Kyle's-λ, book-OFI, cancellations, deep book, cross-asset lead-lag and
   institutional flow, order-flow information has a **seconds-scale half-life** and is
   2–3 orders of magnitude below trading costs by any horizon you could act on. The one
   genuine positive is **contemporaneous price formation / execution**, not a signal:
   book-side OFI and the cancellation channel (invisible to trade data) describe how price
   forms, and that has execution value, not alpha. *(The advisor's L3-from-L2 and
   retail/institutional ideas, done concretely — see §4.)*

5. **Realistic L2 execution costs are ~17–23% cheaper than the flat 5 bps assumption** for
   liquid majors, computed by walking the real book (`src/l2_costs.py`). Prior flat-cost
   backtests were conservative — but cheaper costs do not rescue the strategy.

A sixth, structural finding underlies all of this: the **rolling z-score mechanically
reverts even on a pure random walk**, which silently inflates reversion and backtest
numbers. Subtracting that mechanical floor is required for every reversion claim here.
*(Methodological contribution #2.)*

---

## Relationship to the original `RESULTS.md` (what stands, what is retracted)

| RESULTS.md claim | Verdict here |
|---|---|
| §1 "Kalman finds cointegration on 99.7% of pairs OOS" (the self-described strongest finding) | **Retract** — Kalman-innovation whitening artifact; placebos pass at 100% (§1 below). Genuine rate 2–3%. |
| §1 "Static OLS finds cointegration on 5.4%" | This *is* the honest number, and it ≈ the 5% null false-positive floor — i.e. it was already evidence of *no* cointegration, not a weak positive. |
| §2 "LSTM/booster beat z-score; ML pnl_mean_to_std ~0.38" | Not re-litigated here, but §3 "ML = fancy z-score" (their own ablation: `latest_spread_z` carries 79% of PnL) is the operative truth; the model ranking also **flips by window** (`l2_analysis.md` Finding 2). Don't report a fixed ranking. |
| §4 "5 bps break-even is retracted; 15 bps loses deterministically" | **Largely consistent.** With a conventional stop the strategy loses at every cost level (§3). Removing the stop *can* profit — but only as a multi-month, −41%-drawdown hold, not an hourly cost-managed strategy. The "no clean hourly edge after costs" conclusion stands. |
| §4 Deflated-Sharpe caveat (pre-cost signal indistinguishable from selection-best at N≥50) | **Consistent** — the whole project's edge does not survive selection adjustment. |

Net: the original paper's *cautious* statements (cost ceiling, DSR, "ML = fancy z-score")
all stand. Its one *optimistic* pillar — 99.7% cointegration — falls. The L2 work replaces
it with a cleaner, placebo-validated narrative.

---

## Data & methods (brief)

- **Source.** Tardis.dev historical: `book_snapshot_25` (top-25 depth, per-update) +
  `trades` (tick volume with aggressor side), Binance spot, top-50 USDT symbols
  (`data/l2_universe_top50.txt`). ~175 GB raw cache; 170 book / 111+ trade days ingested
  into 1s bars (`src/l2_store.py`, `src/trades_store.py`); per-trade sizes read from the
  raw csv.gz where needed (for the size/institutional work).
- **Inference discipline.** Every short-horizon regression uses Newey–West HAC SEs (naive
  OLS inflates t-stats on overlapping returns — this caught a spurious "5-min reversal"
  that collapsed from t=−22.8 to t=−1.4). Reversion/backtest claims use pair-clustered and
  block bootstrap, strict out-of-sample walk-forward splits, and an explicit placebo
  (random-walk / phase-randomized / block-shuffled / random-pair) for every positive claim.

---

## Result 1 — The cointegration headline is a Kalman whitening artifact (retract) *(contribution #1)*

Full write-up: **`docs/CORRECTION_kalman_cointegration.md`.**

The screen (`scripts/run_kalman_pair_screen.py` → `src/kalman_hedge.py`) runs the ADF test
on the Kalman filter's **one-step prediction innovations**. With non-zero state-evolution
variance (`q_beta>0`), the time-varying hedge ratio adapts to track *whatever* relationship
exists, so the innovations are **white by construction** — an ADF on them rejects the unit
root for almost any pair. Testing the stationarity of a filter's own innovations is circular.

**Decisive placebo** — an identical screen on series that *cannot* be cointegrated:

| Series tested | ADF p<0.05 | p<0.001 |
|---|---:|---:|
| Real top-50 pairs | 100.0% | 96.4% |
| Placebo (a): independent simulated random walks | 100.0% | 98.3% |
| Placebo (b): coin vs phase-randomized surrogate of another coin | 100.0% | 95.0% |
| Placebo (c): real pair, one leg block-shuffled | 100.0% | 95.0% |

The placebos pass at the *same* 100% rate as real pairs — reproducing the original
99.7%/96.3% almost exactly. The Kalman ADF carries **zero** cointegration information.
Clean, non-circular tests (Engle–Granger and static OLS+ADF on the test window) put genuine
OOS cointegration at **2.1–3.2%** — at or below the 5% null floor. Essentially none of the
universe is genuinely cointegrated OOS. Scripts: `scratch/audit_part1.py`, `audit_part1b.py`.

**Implication:** do not select pairs on the Kalman ADF p-value (it is noise). Select on OOS
reversion speed / OU half-life instead — which §2 shows actually works.

---

## Result 2 — Mean-reversion is genuinely selectable out-of-sample *(a real positive)*

Strict OOS persistence test, all 1,225 pairs, **9 disjoint walk-forward splits** (6mo train
→ next 3mo test), static OLS hedge, every number **net of the rolling-z mechanical floor**
(§6). Scripts: `scratch/persistence_test.py`, `persistence_robust.py`.

- In-sample reversion metrics **predict OOS reversion:** Spearman ρ(train OU-κ, OOS excess
  reversion) = **+0.46** [+0.37, +0.54], p<0.001, **positive in every one of the 9 splits.**
  Variance-ratio and in-sample-excess agree (ρ ≈ ±0.46).
- **Top-quintile vs bottom-quintile OOS excess reversion = +0.69z** [+0.58, +0.81], p<0.001
  (top ≈ +0.88z ≈ 2× universe average, ~5× the bottom quintile).
- **Placebos are clean:** random quintile spread = +0.0001z; shuffling the train→OOS link
  collapses ρ to −0.002; floor-subtraction isn't manufacturing it (top/bottom mechanical
  floors identical, 0.986/0.984); train κ even predicts OOS κ (+0.47).

So — unlike cointegration — reversion is a **graded, selectable, persistent** statistical
property. This is the project's one genuine positive foundation. *But selectability of a
z-units-per-48h property is not the same as profitability* (§3).

---

## Result 3 — Profitability hinges entirely on the stop/exit rule *(the real bottom line)*

Honest net-of-cost walk-forward backtest (`scratch/wf_backtest.py`, `wf_robustness.py`,
`wf_nostop_stress.py`): 19 walk-forward splits, OU-half-life pair selection, static train
hedge, z-entry |z|≥2, realistic costs, with random-pair and phase-randomized placebos. The
headline is **not** a single Sharpe — it is that the P&L is **entirely determined by the
exit/stop rule**, which iteration 8 isolated.

### The stop rule is the dominant P&L lever

| Variant (static hedge, OU-selected) | Gross S | Net @30bps | Net @2bps | Random-pair placebo (net@30) |
|---|---:|---:|---:|---:|
| **\|z\|=4 stop** (iter-7 baseline) | −1.15 | **−2.25** [−3.21,−1.34] | −1.30 | −0.79 |
| **No stop** (hold to z-exit) | +2.65 | **+2.51** [+1.83,+3.22] | +2.63 | +0.89 |
| Wide stop \|z\|=6 | +1.34 | +0.85 | +1.27 | +0.23 |
| Time-exit (2× train half-life) | +3.03 | +1.87 | +2.88 | −0.39 |
| Hold-to-convergence, no stop | +2.80 | +2.68 | +2.79 | +0.88 |

**Correction to iteration 7.** The earlier "loses even gross (−1.15), worse than random"
headline was **specific to the |z|=4 stop**, not a property of the strategy. A tight
asymmetric stop is *actively harmful* to a mean-reverting position — it realizes losses
exactly when the spread is most stretched and most likely to revert. Removing it flips gross
to +2.65 and net to +2.51, beating the random-pair placebo (+0.89) by ~2× the across-seed
SD; the time-exit and hold-to-convergence variants (also static-z, artifact-free) confirm it.
*(Rolling-hedge variants score even higher, +3–4, but they recompute z from a trailing window
and their placebos run +2.1–2.6 — mostly the rolling-z mechanical artifact of §6; do not lead
with them. The clean evidence is the static-z no-stop / convergence cells.)*

### But the no-stop "win" is not the strategy the project specified

Diagnostics on the no-stop run (`scratch/wf_nostop_stress.py`) reveal its character:
- **Median holding time ≈ 848 bars (~35 days); 78% of positions never converge within the
  3-month window.** This is a *multi-month hold-the-spread* exposure, not the hourly
  stop-managed stat-arb the project set out to build.
- **Portfolio max drawdown −41%** (worst pair-window −77.5%). The +2.51 annualized Sharpe
  hides long, deep underwater stretches and is flattered by the autocorrelation of
  multi-week holds (the 168h-block bootstrap understates the SE because holds exceed the
  block — the honest unit of observation is the window, not the hour).
- **~⅓ of the return is generic survivor co-movement:** the random-pair placebo also earns
  +0.89 with no stop — bounded spreads of survivor majors partially revert over a quarter
  regardless of selection. OU selection adds the rest (+1.62 edge over placebo).

### Is the no-stop result a survivorship artifact? Stress-tested — mostly no.

The universe is the *current* top-50 liquid coins (survivorship-filtered: every coin survived
to 2026), so "hold to convergence" never meets a permanently-decoupling pair — exactly the
tail a stop exists to protect against, and which Result 1 (no cointegration) says we cannot
certify away ex ante. I stress-tested this by injecting, in each test window, permanent
structural breaks (a fraction p of selected pairs diverge by up to ~86% in one leg and never
revert; selection stays on the *unbroken* train, as in live trading). Net Sharpe vs break rate:

| break prob p (per pair / quarter) | 0% | 2% | 5% | 10% | 20% |
|---|---:|---:|---:|---:|---:|
| **No-stop**, 50% blowups | +2.51 | +2.45 | +2.37 | +2.17 | +1.92 |
| **No-stop**, ~86% blowups (delisting-scale) | +2.51 | +2.20 | +1.92 | +1.28 | +0.12 |
| **\|z\|=4 stop** (any blowup size) | −2.25 | ≈−2.3 | ≈−2.3 | ≈−2.4 | ≈−2.4 |

The no-stop edge is **robust**: it only collapses to ≈0 under an *extreme* 20%-of-pairs-per-
quarter near-delisting rate (~80%/yr attrition); at plausible rates (≤5%/quarter) it stays
+1.9–2.4. Diversification across 10 pairs absorbs breaks. So the no-stop result is **not**
primarily a delisting-tail artifact — the honest caveats are the holding horizon, the −41%
drawdown, and the survivor-co-movement third, not blow-up risk.

### Honest bottom line on tradeability

Mean reversion is real and selectable (Result 2), and it *is* monetizable — but only in a
form far from the project's stated design: a slow, multi-month, deep-drawdown hold-the-spread
exposure on a survivorship-filtered universe, partly generic beta. As the **hourly,
stop-managed stat-arb** originally specified it **loses** (−2.25), because the stop required
to bound risk at hourly cadence is precisely what destroys the thin, slow reversion. The
defensible claim is the *sensitivity itself*: report the stop/exit-rule dependence (and the
window-dependence — the model ranking also flips by window, `l2_analysis.md` Finding 2), not
a single Sharpe.

---

## Result 4 — Microstructure / order flow is near-efficient at tradeable horizons *(the advisor's ideas, done)*

Across six iterations and ~ten tests, every natural "volume-as-alpha" hypothesis was
falsified with the appropriate control, and the genuine positives are about price formation,
not signal.

**The advisor's "retail vs institutional / volume as information" (iters 1–2, 4).**
- Trade size **is** a proxy for information *per order*: per-order permanent (information)
  impact is monotone in size, institutional (>$10k) ≈ 1.6–2.1 bps ≈ **2× retail**, with a
  transient liquidity hump that decays 13–20% (classic temporary impact). *Per dollar* the
  ranking inverts but that excess is mechanical (tick/bounce on a tiny denominator), not
  information — institutions split/time to minimize footprint. Script: `scratch/impact_decomp.py`.
- At a **tradeable** horizon (15m–1h) institutional net flow adds **+0.03% incremental R²**
  (≈ zero), and where it loads the sign is **reversal** (price-pressure mean-reversion),
  *not* informed continuation: the seconds-scale information is fully priced by 15 min.
  Script: `scratch/inst_flow_horizon2.py`.

**The advisor's "infer L3 from L2" (iter 3) — the best microstructure result.**
- Reconstructing book-side order flow + cancellations from per-update `book_snapshot_25`:
  book-OFI explains within-second mid moves much better than trade flow (book-OFI
  incremental over trade-OFI = +0.125/+0.164 R² for BTC/ETH — larger than the reverse).
- **The cancellation channel is genuinely new information:** **81–85% of best-level size
  reductions are cancels, not trades** — invisible to trade-flow analysis — and trailing
  cancel-imbalance carries directional info with the right sign (t up to −38).
- But all of it is **contemporaneous** (predictive increment decays to ~0 by 30s); R² ≤ 2–3%.
  This is **price-formation / execution** value, not a forecast. Scripts:
  `scratch/book_ofi_incremental.py`, `book_ofi_cancel_stretch.py`, `deep_book_probe.py`.

**Everything else, all null with the right control:** seconds-OFI on the spread (R²≈0.001,
~0.02 bps vs ~20 bps cost; `pair_ofi_spread.py`); VPIN regime-filter (dead and sign-inverted
— it's a directional-consensus meter, not a toxicity meter; `vpin_spread_vol.py`);
cross-asset BTC→alt lead-lag (the 1–5s "lead" is **stale-price** spurious — it collapses at
30/60s sampling — and otherwise arbitraged; `leadlag_xasset.py`); deep book levels 2–10
carry *less* than top-of-book (`deep_book_probe.py`); microstructure vs HAR-RV for vol
forecasting (HAR already wins, OOS R²≈0.51; micro adds ~2–3% of that, not worth the churn;
`har_vol_regress.py`).

**Thesis:** crypto majors are microstructure-efficient at every horizon you could trade
against ~10–20 bps costs. Order flow (trade + book) explains *contemporaneous* price
formation extremely well and carries genuine **permanent** information, but that information
has a **seconds-scale half-life**. The microstructure payoff is in **execution** (which leg
to post vs lift, cancellation-aware queue placement), not a directional signal.

---

## Result 5 — Realistic L2 execution costs are ~17–23% below the flat assumption

Portfolio backtest, flat 5 bps slippage vs L2 book-walk slippage (`src/l2_costs.py`,
`--with-l2-costs`): walking the real top-of-book makes execution **~17–23% cheaper** than the
flat assumption for these liquid majors (e.g. z-score total return −12,827 → −9,868; XGBoost
−22,143 → −18,286). Prior flat-cost backtests were therefore *conservative*. (All strategies
remain unprofitable after costs; turnover dominates. The `max_drawdown` column in those
tables is percent-of-peak with a degenerate denominator and is **not** interpreted.)

---

## Result 6 — The rolling-z mechanical-reversion artifact *(contribution #2)*

A rolling z-score `(spread − rolling_mean) / rolling_std` mechanically reverts even for a
**pure random walk**, because the rolling mean chases the level — so |z|>2 is almost always
followed by "reversion" toward a mean that is itself moving. This silently inflates every
reversion statistic and, in a backtest, manufactures a spectacular fake edge: a rolling-z
variant of the §3 strategy shows **+4.75 Sharpe**, but **random pairs reproduce +2.18 and
phase-randomized noise +2.42** under the identical rule. It carries zero selection
information and must never be reported as an edge. Two consequences enforced throughout this
work: (i) every reversion magnitude is reported **net of the random-walk mechanical floor**;
(ii) any rolling-window statistic is validated against a random-pair / phase-randomized
placebo. Scripts: `scratch/wf_sanity.py`, `wf_diag.py`, `audit_part2.py`.

---

## Honest limitations

- **No genuinely cointegrated universe to draw from** (Result 1) — pair selection falls back
  to reversion-speed ranking, which is selectable (Result 2) and monetizable only in the
  slow, deep-drawdown form of Result 3, not as an hourly stop-managed strategy.
- **Survivorship-filtered universe** (current top-50). It is stress-robust for the no-stop
  result (§3) but still flatters any hold-to-convergence rule; a true point-in-time universe
  (with delisted coins) would be the clean fix and is the most valuable remaining data work.
- Microstructure tests are concentrated in 2024, BTC/ETH/SOL-heavy; broader months would
  strengthen generality (the 31-day institutional run and full-history reversion test help).
- Effective sample size is the binding constraint for small effects: few, correlated pairs
  give statistically fragile extreme-event counts — a real limit on detecting any thin edge.
- Single exchange / quote currency / interval, as in phase 1.

---

## Reproduce — script index

| Result | Scripts |
|---|---|
| 1. Cointegration artifact | `scratch/audit_part1.py`, `audit_part1b.py`; `docs/CORRECTION_kalman_cointegration.md` |
| 2. Selectable reversion | `scratch/persistence_test.py`, `persistence_robust.py` |
| 3. Stop/exit-rule dependence | `scratch/wf_backtest.py`, `wf_robustness.py`, `wf_nostop_stress.py`, `wf_sanity.py`, `wf_diag.py` |
| 4. Microstructure | `scratch/impact_decomp.py`, `inst_flow_horizon2.py`, `book_ofi_incremental.py`, `book_ofi_cancel_stretch.py`, `pair_ofi_spread.py`, `vpin_spread_vol.py`, `leadlag_xasset.py`, `deep_book_probe.py`, `har_vol_regress.py` |
| 5. L2 costs | `src/l2_costs.py`, `scripts/run_portfolio_backtest.py --with-l2-costs` |
| 6. Rolling-z artifact | `scratch/wf_sanity.py`, `wf_diag.py`, `audit_part2.py` |

Full chronological detail with "what am I missing" at each step: `docs/l2_research_log.md`.

---

## Bottom line for the advisor

The L2 data did its job — it let us *test* the volume-as-information and L3-from-L2 ideas
properly, and the verdict is honest near-efficiency: the information is real but lives at
seconds and is priced before any tradeable horizon; its value is in execution, not alpha.
Along the way the L2 window exposed that the project's headline cointegration result was a
filtering artifact (genuine cointegration ≈ chance), while showing that mean-reversion *is*
a real, selectable property. That property is even monetizable — but only as a slow,
multi-month, −41%-drawdown hold-the-spread exposure, *not* the hourly stop-managed stat-arb
the project specified, which loses because the stop that bounds hourly risk is exactly what
destroys the thin, slow reversion. The deliverable is a rigorous, largely-null paper with
**two methodological contributions** (the Kalman-innovation and rolling-z artifacts, both
demonstrated with placebos) and **one real effect whose tradeability is entirely a function
of the stop/exit rule** — more honest, and more publishable, than the original optimistic
framing. **Action items:** retract the 99.7% cointegration claim from RESULTS.md;
independently reproduce the Kalman placebo (`CORRECTION_kalman_cointegration.md`) before any
external use; report the stop/exit-rule sensitivity (§3), not a single Sharpe.
