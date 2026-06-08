# L2 microstructure research log

Running log for the self-paced research loop. Newest iteration on top. Each
entry: what was done, what was found, what I think I'm missing, and the plan for
the next iteration.

---

## Iteration 16 — redo on full-2024 L2 data: every microstructure finding generalizes (incl. the Aug-5 crash)

User asked to "redo analysis with the new data." The L2 download (paused-research, kept running) now covers **all of 2024**
(~362 days × 50 symbols raw), vs the ~5–10 days of Jan 2024 the microstructure chapter originally used — directly attacking
its biggest standing caveat ("Jan 2024 only, BTC/ETH-heavy"). Note: the ingested 1s stores are stale (data/l2 → Jun, data/
trades → Apr), so the redo reads RAW csv.gz directly and samples **13 days across all 12 months + 4 symbols**, deliberately
including volatile regimes (Aug-5 yen-carry crash, Mar-13 BTC-near-ATH, Nov–Dec rally). Three parallel subagents; I verified
each script+log against the reported numbers. New files: `scratch/{book_ofi,exec_value,hidden_liquidity,inst_flow}_2024.{py,log}`.

### Results — all four findings replicate; near-efficiency holds in every regime
- **Book-OFI + cancellations** (the "best micro result"): contemporaneous book-OFI incr-over-trade R² = +0.11–0.48 (stronger
  in illiquid AVAX), cancel share 77–90%, predictive decay to ~0 by 30s — all 4 symbols, all year. **On Aug-5 order flow gets
  LESS predictive, not more** (predictive R² @1s drops, cancel signal collapses to t≈0).
- **Execution value**: aggressive crossing still cheapest (pooled 1.10 bps); L3 signal still no edge (corr +0.03, loses to
  placebo, sign-flip fails, oracle saves +1.40). **On the crash the oracle opportunity GROWS (+2.34 bps) but the signal's
  corr collapses to +0.003** — it helps least exactly where it could help most. No regime-dependence.
- **Hidden / iceberg liquidity**: prevalence replicates (1.3–4% of volume; iceberg BTC/ETH/SOL ~10–12% ≫ AVAX ~1.9%);
  predictive null holds (HAC). *New descriptive observation:* hidden volume **spikes 2–3× on the crash** (BTC 5.7% vs 2.7%) —
  institutions hide more in stress — but still non-predictive.
- **Institutional flow**: reversal-not-information all year; ~0 incremental R²; **zero** continuation regimes across 26
  day×horizon cells incl. the crash.
- **Per-order impact by trade size** (`impact_decomp_2024.py`, added after the disk fix): the advisor's per-trade
  retail-vs-institutional question. Replicates across all 2024 + 4 symbols at the robust seconds horizon — 1s impact
  monotone in size (institutional > mid > retail on every symbol), institutional ≈ **2.0–2.6× retail**, ~11 bootstrap-SEs
  wide; institutional transient hump replicates on calm days (~85–92% permanent). Caveat: the 300s "permanent" estimate is
  noisy on the broad sample (drift-contaminated for tiny retail orders / the crash) — the finding rests on seconds-horizon
  impact. **Process note:** the subagent for this one *stalled on a self-armed Monitor* (the known failure mode) and returned
  without finishing; it had run BTC only, whose lone-300s-outlier looked "broken." I took over, confirmed the script was sound
  (correct raw-book adaptation, bootstrap SEs, regime split), ran it to completion myself, and read all 4 symbols — clean
  replication. Lesson reinforced: verify stalled-subagent output by re-running, don't trust a partial log.

### What this settles / what am I missing
- The microstructure null is **robust across all 2024 market conditions, not a Jan-2024 artifact** — if anything near-
  efficiency holds *more* firmly in stress. The single genuinely new finding is descriptive (hidden-liquidity surges in the
  crash). Updated `L2_FINDINGS.md` Result 4 (new "Full-2024 robustness" subsection) + limitations.
- The reversion/backtest results are unaffected (they use multi-year hourly `spot_1h`, not the L2 raw).
- Caveat that remains: still 2024-only and ≤4 symbols for event-level work; the ingested 1s stores would need a re-ingest to
  run the *ingested-reading* scripts (ofi_horizons, pair_ofi_spread, impact_decomp) on full-year — those still cap at H1 2024.
- The download is still running and eating disk (~160 GB free, ~190 GB/day); it has no remaining analytical purpose — should
  be stopped soon.

### Plan / state
The "Jan 2024 only" caveat is now closed. The paper is unchanged in conclusions but materially more robust. Remaining optional
work: re-ingest 2024 to refresh the 1s stores and re-run the ingested-reading OFI scripts (confirmatory); else the project is
complete. Recommend stopping the background download.

---

## Iteration 15 — the survivorship check (no data pull needed!) — the alpha SURVIVES and strengthens

I'd flagged the survivorship-free universe as needing a delisted-coin data pull. "What am I missing?" — I checked what's
already on disk first, and **`data/spot_1h` has 204 symbols, not the 50 the backtests use.** The extra ~154 are exactly the
lower-quality / pumped-and-crashed / meme / late-listed names a "current top-50" filter removes (1000REKTUSDT, USELESSUSDT,
NOBODYUSDT, TROLLUSDT, FARTCOINUSDT, …). So a much-less-survivorship-biased test was possible with zero new data.

### Test (`scratch/wf_survivorship.py`) — identical pipeline, top-50 vs full 204-symbol universe
Per-window ≥90%-coverage gating so coins enter only once listed (genuine point-in-time *entry*: 11 names start 2021, 73 in
2022, 24 in 2023, 53 in 2025; 98 vs 39 usable symbols/window). Monthly (frequency-honest) Sharpe / maxDD:

| config | top-50 (survivor) | FULL 204 (incl. meme/crashed) |
|---|---|---|
| no-stop z, 10 pairs | 2.41 / −41% | **3.17 / −29%** |
| no-stop z, 40 pairs | 2.54 / −29% | **3.76 / −23%** |
| \|z\|=4 stop, 10 pairs | −1.21 | +0.32 |

**The no-stop alpha not only survives a 4× broader, lower-quality, point-in-time-on-entry universe — it strengthens**
(monthly Sharpe 2.54→3.76, maxDD −29%→−23%). More symbols ⇒ better selection pool + more diversification, dominating any
harm from junk names. Strong evidence the alpha is **not** a survivor-majors artifact.

### Honest caveats (so this is "much less" not "zero" survivorship)
- The on-disk set under-represents **exits**: only 3 coins (ICX/LTO/STG) have in-sample truncated history, and
  fully-delisted coins (LUNA→0, FTT→0) are absent from disk. So the *entry* side of point-in-time is well-tested, the
  *death* side barely.
- The backtest's NaN-as-flat handling would understate a sudden-delisting loss (position just goes flat at last mark).
- The broad-universe Sharpe itself is **not deployable** — illiquid memes can't be traded at $1/leg / 30 bps. The robust
  finding is the *direction* (survives/strengthens when you drop the filter), not the inflated magnitude.
- Net: combined with the iter-8 structural-break stress robustness, the survivorship concern is now **substantially
  addressed**; a truly point-in-time universe with dead coins is the only gold-standard check still open (needs a pull).

### What am I missing? / state
- This was the highest-value remaining check and it came back reassuring with zero new data. The central positive result
  (market-neutral, diversified, frequency-honest Sharpe ~2, ~30% DD, never-stop, survivorship-robust) is now about as
  thoroughly vetted as this dataset allows. Updated `L2_FINDINGS.md` (Result 3 + limitations) and `ADVISOR_SUMMARY.md`.
- I am now genuinely at the terminus: the only remaining task (delisted-coin pull) is a real data-acquisition project, not a
  loop iteration. Recommend the user act on the deliverables.

### Plan for next iteration (only if continued)
1. Delisted-coin data pull (LUNA/FTT/etc.) for the gold-standard point-in-time test — a project decision, not a quick script.
2. Otherwise the loop is at its terminus; further iterations have negative expected value.

---

## Iteration 14 — consolidation: advisor one-pager + the honest Sharpe is a range (~1.7–2.5)

The science has converged; this iteration is communication, the highest-value remaining work.

### A. Advisor one-pager (`docs/ADVISOR_SUMMARY.md`)
Distilled the 13 iterations into a 1-page advisor-facing memo: the bottom line, the 5 results + 2 methodological
artifacts, each with its key number and the honest caveats, and 3 suggested next steps (retract the cointegration
headline; build reversion as a wide no-stop book validated on a point-in-time universe; treat microstructure as closed
for alpha). It leads with the corrected Sharpe ~2 headline and explicitly flags the survivorship caveat. Points to
`L2_FINDINGS.md` / `CORRECTION_kalman_cointegration.md` for detail.

### B. Sharpe-range refinement (flaw-check on iter 13)
"What am I missing on iter-13?" — even the monthly Sharpe (~2.5) is mildly optimistic, because some positions are held
>1 month (p95 hold ≈ 88 days), so monthly returns retain residual autocorrelation. The most conservative unit is the
quarter/window: iter-9's window-level gave ~1.7 annualized (10 pairs, t=3.65). So the defensible deployable figure is a
**range ~1.7–2.5, central ≈ 2**, not a point estimate. Noted in both `L2_FINDINGS.md` and the one-pager. (No re-run
needed — both endpoints already computed.)

### State of the project
Fourteen iterations. The paper is complete, internally consistent, frequency-honest, and now packaged for the advisor.
Every one of the advisor's L3-from-L2 / volume-as-information / retail-vs-institutional ideas has been measured (real
contemporaneous info, no edge — including a from-scratch execution sim and hidden-liquidity detection). The one positive
result (market-neutral diversified reversion alpha, Sharpe ~2, ~30% DD, never-stop) is fully characterized, bounded, and
correctly scaled, with two placebo-proven methodological artifacts.

### What am I missing? / honest assessment
- I have reached the end of productive *new* analysis on this dataset. Continuing to spin the loop would manufacture
  low-value work or risk over-fitting narratives onto a converged result. The remaining genuine task — a point-in-time
  (survivorship-free) universe — requires a delisted-coin data pull (LUNA/FTT/etc.), which is a real data-acquisition
  effort, not a quick script, and the stress test already shows the result is break-robust.
- The most honest recommendation to the user is to **pause the research loop here** and act on the deliverables
  (send `ADVISOR_SUMMARY.md`; decide whether to fund the point-in-time data pull).

### Plan for next iteration (only if continued)
1. Point-in-time / survivorship-free universe — the one material open check; needs delisted-coin historical data.
2. Otherwise, the loop is at a natural terminus; further iterations have diminishing/negative value.

---

## Iteration 13 — self-correction: iter-12's "+3.4 Sharpe" was hourly-inflated; honest deployable Sharpe is ~2–2.5

Found a real flaw in my own iter-12 frontier: it reported annualized *hourly* Sharpes (+3.2/+3.4) for a strategy whose
median hold is ~35 days. Hourly returns of a month-long-hold strategy are heavily autocorrelated, so the annualized hourly
Sharpe overstates the deployable number (HAC only partly corrects it). Recomputed Sharpe across sampling frequencies
(`scratch/wf_sharpe_freq.py`):

| config | hourly | daily | weekly | monthly (honest) | monthly 95% CI |
|---|---|---|---|---|---|
| 40p no-stop z | 3.18 | 4.23 | 3.65 | **2.54** | [1.97, 3.23] |
| 40p no-stop conv | 3.45 | 4.01 | 3.24 | **2.06** | [1.59, 2.61] |
| 10p no-stop z | 2.53 | 3.29 | 2.98 | **2.41** | [1.75, 3.18] |

### Findings (and corrections to iter 12)
- **Honest deployable Sharpe ~2.0–2.5 (monthly), not ~3.2–3.4.** Sharpe rises hourly→daily (microstructure noise averages
  out) then falls weekly→monthly (hold autocorrelation reduces effective N) toward the true low-freq value. Iter-12's headline
  was inflated; **corrected in `L2_FINDINGS.md`** (Result 3 + exec summary).
- **It still survives:** monthly CIs are strictly positive (lower bounds 1.6–2.0). Real alpha, just ~2, not 3+.
- **The "best config" flips at the honest frequency:** 40p no-stop **z-exit** (monthly 2.54) beats convergence-exit (2.06) —
  the reverse of the hourly ranking. So iter-12's "conv-exit is best" was also an hourly artifact. Lesson reinforced: for a
  multi-week-hold strategy, *always* judge Sharpe at a frequency ≥ the holding period.

### What am I missing? / state
- This is the second time a result needed deflating for the slow-hold autocorrelation (first: iter-9 window-level check).
  The honest, stable bottom line is now locked: **a market-neutral, diversified mean-reversion alpha at a realistic Sharpe
  ~2–2.5 and ~30% drawdown, capturable only by a patient never-stop book on a survivorship-filtered universe.**
- 13 iterations in, the science is complete and now also correctly *scaled*. Genuinely the only remaining work is (1) the
  survivorship-free universe (data pull; low marginal value given break-robustness) and (2) advisor packaging.

### Plan for next iteration (prioritized)
1. **Advisor one-pager** from `L2_FINDINGS.md`, now with the corrected Sharpe ~2–2.5 headline.
2. Point-in-time universe only if a cheap delisted-coin pull is feasible.
3. `L2_FINDINGS.md` = single source of truth (now frequency-honest throughout).

---

## Iteration 12 — the risk-rule efficient frontier: the alpha is capturable at ~30% DD by a patient, diversified, never-stop book

The advisor-relevant culmination of iters 8–11: is the no-stop −41% drawdown fatal, or does a practical risk rule capture
the alpha at a bearable drawdown? Mapped (Sharpe, max-DD, Calmar) across pair-count / stop / exit / sizing
(`scratch/wf_frontier.py`, realistic 30 bps). DD is in per-unit-notional / leverage-equivalent units (read for ranking).

### Findings
- **Diversification cuts drawdown AND raises Sharpe:** 10→20→40 pairs (no-stop z): maxDD −41% → −31% → **−29%**, Sharpe
  +2.51 → +2.93 → **+3.18**. The −41% baseline DD was partly idiosyncratic and shrinks with breadth — the single most useful
  practical lever.
- **Best risk-adjusted = 40-pair, no stop, convergence exit:** Sharpe **+3.44**, maxDD −31%, **Calmar 5.1**. Vol-targeting
  barely moves the frontier (Sharpe 3.0 vs 3.2; DD similar).
- **Stops hurt at every width** — the iter-8 lesson, now quantified on the frontier: tight \|z\|=4 is catastrophic (−265%
  leverage-equiv DD, churn, 23.5k trades); even a *wide* \|z\|=6 stop is the worst of both worlds (−78% DD, Sharpe collapses
  to 1.5, 13.7k trades) — it still cuts reverting winners *and* realizes the rare blow-up the no-stop rule holds through.
- So the honest, most-positive-yet framing: **the alpha is capturable at Sharpe ~3 / Calmar ~5 / ~30% DD by a patient,
  well-capitalized, market-neutral book that diversifies wide and never stops.** A real strategy profile — just not the
  tight-risk-managed hourly one the project specified. Updated `L2_FINDINGS.md` Result 3 + exec summary.

### What am I missing? / state of the project
- The reversion result is now characterized AND bounded (selectable, market-neutral, diversified, significant, robust, and
  capturable at ~30% DD with breadth + no stop). The microstructure chapter is closed. The remaining caveats are intrinsic:
  survivorship universe, multi-month holds, ~⅓ survivor-co-movement floor, leverage-equiv DD units.
- **I am at genuine diminishing returns on new analysis.** 12 iterations have produced a complete, honest, internally
  consistent paper (`L2_FINDINGS.md`): 2 methodological artifacts (Kalman, rolling-z), microstructure near-efficiency across
  ~12 tests incl. execution + hidden liquidity, and a vindicated-but-patient-only reversion alpha with a mapped risk frontier.
- The two genuinely remaining items are (1) the **survivorship-free / point-in-time universe** (needs a delisted-coin data
  pull; the stress test already shows break-robustness, so lower marginal value), and (2) **advisor packaging** (one-pager).

### Plan for next iteration (prioritized)
1. **Advisor-facing one-pager** distilled from `L2_FINDINGS.md` — the research has converged; communication is now the
   highest-value work.
2. Point-in-time universe (delisted coins) only if a cheap pull is feasible.
3. `L2_FINDINGS.md` remains the single source of truth.

---

## Iteration 11 — hidden/iceberg liquidity (novel L3-from-L2) + the no-stop edge is diversified, not a factor bet

Two parallel deliverables: a genuinely novel microstructure *measurement* (the advisor's "infer L3 from L2", taken as
literally as the data allow) and the final characterization of the no-stop reversion result.

### A. Hidden / iceberg liquidity from L2 + tape (`scratch/hidden_liquidity.py`) — clean measurement, null edge
Detected hidden liquidity as `max(0, executed-vol-at-price − max-displayed-size-at-price)` per best-price episode (~2.3M
traded episodes, BTC/ETH/SOL/AVAX, 8 days; conservative lower bound since post-update snapshots under-detect; side
convention validated — 76% of buys print at the tracked ask).
- **Descriptive (novel, publishable):** hidden liquidity is **2–3.5% of at-best executed volume** in crypto majors, with a
  clean depth/institutionalization gradient — **BTC highest** (9.80% of traded-through price levels show a hidden component),
  **thin AVAX lowest** (2.34%). Median hidden chunk ≈13% of the level's volume. So ~1-in-10 (BTC) to 1-in-40 (AVAX) traded
  price levels reveal hidden supply.
- **Predictive (HAC):** **null.** Next-hour realized vol: incremental R² +0.0025, hidden terms insignificant
  (hidden_fraction t=−1.55, iceberg_rate t=+1.02). Next-hour direction: signed-hidden subsumed by OFI (contemporaneous
  t=−1.85, marginal & sensibly negative — hidden ask supply caps buying — but not significant; next-hour incremental R²
  +0.0005). Real, measurable, contemporaneous-only → **reinforces near-efficiency** once more. This was the last untested
  literal "L3 from L2" angle; it closes the microstructure chapter for good.

### B. PCA: the no-stop edge is DIVERSIFIED idiosyncratic reversion, not a factor bet (`scratch/wf_nostop_pca.py`)
Iter-10 showed market-neutrality (beta≈0 to PC1). This asks: is the +2.5 a concentrated bet on one slow eigenportfolio?
- Universe PCA: PC1 (market) = 49.3% of variance, then a flat tail (PC2–10 ≈3–5% each; top-10 = 84%).
- **The no-stop return is explained only 1.7% by the top-10 statistical factors** (PC1 alone 0.35%, max single-PC 0.49% at
  PC2). So it is **diversified idiosyncratic** mean-reversion across many ~independent pair bets — not one crowded basket.
  (Mild caveat: long-short pair strategies are inherently low-factor-R²; but 1.7% rules out a concentrated-factor explanation.)
- **Net characterization (iters 8–11), the no-stop +2.51:** market-neutral (β≈−0.06), not a factor bet (factor R²=1.7%),
  window-level significant (t=3.65, 18/19 windows), survivorship/break-robust — a *genuine, diversified, market-neutral
  mean-reversion alpha*. Caveats unchanged: slow (median ~35d holds, 78% non-converging), −41% drawdown, ~⅓ generic
  survivor-co-movement floor; an hourly |z|=4 stop destroys it.

### What am I missing? / next iteration
- The no-stop result is now characterized as far as the current (survivor) universe allows, and the microstructure chapter
  is definitively closed (alpha, vol, regime, lead-lag, institutional flow, deep book, execution, hidden liquidity — all
  measured, all near-efficient). I'm reaching genuine diminishing returns on *new* analysis.
- The two remaining items are both about *robustness of the one positive result*, not new signal hunting: (1) the
  **point-in-time / survivorship-free universe** (needs a delisted-coin data pull) — the cleanest unaddressed check; (2) an
  **OU-half-life / reversion-speed selection vs cost** sweep to find the *best-case* tradeable variant (slow no-stop, vol-
  targeted, wider universe) — does *any* honest implementation clear cost at a *bearable* drawdown? Expected no, but it
  would bound the positive result cleanly.
- Increasingly the highest-value work is **consolidation / advisor packaging**, not more tests.

### Plan for next iteration (prioritized)
1. Point-in-time universe (delisted coins) if a cheap pull is feasible — else the best-case tradeable-variant sweep.
2. Draft the **advisor-facing one-page summary** from `L2_FINDINGS.md` (the 5 results + 2 artifacts + the vindicated-but-
   slow reversion alpha), since the research has converged.
3. Keep `L2_FINDINGS.md` as the single source of truth.

---

## Iteration 10 — the no-stop edge is genuine MARKET-NEUTRAL alpha, not disguised crypto beta

Characterized the central iter-8 result by chasing its biggest untested confound: over a 2021–2026 net-up crypto regime,
is the no-stop +2.51 Sharpe really reversion alpha, or just net long-market beta wearing a "market-neutral" label? The PnL
is `pos*(r_A−r_B)` — dollar-neutral but not automatically beta-neutral.

### Test (`scratch/wf_nostop_factor.py`) — beta explanation REJECTED
Regressed the no-stop portfolio's hourly returns on (i) an equal-weight crypto index and (ii) BTC, with block-bootstrap CIs:
- **beta ≈ −0.06** (equal-wt) / −0.07 (BTC) — tiny and *slightly negative* (the correct contrarian sign for buying
  underperformers / shorting outperformers). corr ≈ −0.06.
- **market-neutralized Sharpe = +2.62 (eq) / +2.67 (BTC) — essentially unchanged from the raw +2.65.** Removing the market
  factor does nothing.
- **Daily frequency confirms** (beta −0.04, n=1727 days): no slow-beta accumulation hiding in the multi-week holds.
- % bars same-sign as market = 45% (≈ independent / mildly contrarian).
- The stopped strategy is also ~market-neutral (beta −0.02) and still loses (−1.15) — so beta isn't the difference either.

### What this settles
The no-stop edge is **genuine market-neutral mean-reversion alpha**, not a bull-market artifact. Combined with iters 8–9 it
is now fully vetted: market-neutral (iter 10), window-level significant (iter 9: t=3.65, 18/19 windows), and
survivorship/structural-break robust (iter 8) — but realized only as a slow, multi-month hold (median ~35d, 78% never
converge in-window), with a −41% drawdown and ~⅓ generic survivor-co-movement. **The project's mean-reversion premise is
vindicated as a real, market-neutral statistical effect; the failure was the risk rule + horizon (an hourly |z|=4 stop
realizes losses on spreads that then revert), not the absence of reversion.** Updated `L2_FINDINGS.md` Result 3 accordingly.

### What am I missing? / next iteration
- The no-stop result is now characterized about as far as the survivor universe allows. The one remaining clean check is
  still the **point-in-time / survivorship-free universe** (delisted coins) — but it needs a data pull and the stress test
  already shows break-robustness, so it's lower marginal value than it was.
- A tighter **factor decomposition** (PCA/Johansen: is the +2.5 a single slow mean-reverting basket factor, or many
  independent pair bets?) would precisely name the exposure — cleaner than the market regression. Reasonable next step.
- Microstructure chapter remains closed. `L2_FINDINGS.md` is the canonical paper draft and is now internally consistent
  across all 10 iterations.

### Plan for next iteration (prioritized)
1. **PCA/eigenportfolio characterization** of the no-stop returns: how many independent factors, and is PC1 a slow
   mean-reverting basket? Names the +2.5 precisely with existing data.
2. Point-in-time universe (delisted coins) — only if a cheap data pull is feasible.
3. Keep `L2_FINDINGS.md` as the single source of truth; consider drafting the advisor-facing summary.

---

## Iteration 9 — the advisor's "execution value" finally MEASURED — and it's a clean null (another self-correction)

This iteration chased the advisor's L3-from-L2 idea to the one place I'd repeatedly *claimed* it pays off but never tested:
**execution.** From iters 3–4 onward I asserted "the microstructure payoff is in execution, not signal" and listed
"quantify the execution value vs the `l2_costs` book-walk" as a next step — for six iterations, unmeasured. So I measured it.

### A. Execution-value experiment (`scratch/exec_value.py`, `exec_value_verify.py`) — the L3 signal does NOT help execution
18,432 parent orders ($10k/$50k, randomized side, every 5 min) across BTC/ETH/SOL/AVAX, 8 days of **event-level** raw book
+ trade tape, implementation shortfall vs arrival mid. Three executions: aggressive cross, naive passive (real queue/fill
model, cross on non-fill), and **L3-aware** (post vs cross on the book-OFI + cancellation signal). Pooled (H=30s, bps):
- **Aggressive crossing is cheapest: +1.34 bps.** Naive passive +2.79 (median fill ~0 but the unfilled-tail chase kills the
  mean). L3-aware +1.98. The majors' spread is ≈1 tick (BTC ≈0.02 bps) so there's almost no spread to capture passively.
- **The L3-from-L2 signal carries no execution-timing info.** It LOSES to a random post/cross at the same post-rate
  (z=+2–3 worse); **sign-flipping doesn't rescue it** (I verified — flipped is also worse, confounded only by post-rate);
  and `corr(signal, per-order passive−aggressive advantage) = +0.06` — noise. No L2 feature beats it (cancel +0.03,
  spread −0.02, |OFI| +0.02). **Not a sign error, not a weak signal — the absence of signal.**
- **The opportunity is real but unforecastable from contemporaneous L2.** A perfect-foresight ORACLE (post when passive
  will be cheaper, else cross) hits **−0.23 bps** — a +1.57 bps swing — so execution value EXISTS. But capturing it needs
  to predict whether price drifts before your passive fills, which a *contemporaneous* signal by construction can't. Same
  wall as the alpha results: the signal describes the present, not the near future.
- **Self-correction:** the six-iteration "payoff is in execution" claim is now **retracted** in `L2_FINDINGS.md` Result 4.
  The microstructure signal is genuine *price-formation description* but yields neither a directional nor an execution edge.

### B. Iter-8 self-check (`scratch/wf_nostop_winlevel.py`) — the no-stop correction holds at the conservative unit
The "find a flaw in the last analysis" pass on iter-8: the no-stop +2.51 *hourly* Sharpe has multi-week holds, so the
168h-block bootstrap CI is too tight. Re-tested at the honest unit — the **19 disjoint test windows**: mean window return
+41.6%, **t=+3.65, 16/19 windows positive** (sign-p=0.004); paired no-stop−stop positive in **18/19 windows** (sign-p<0.001);
the stop's worst window is **−150%** vs no-stop's −23%. So iter-8's correction is *not* an autocorrelation artifact — it
strengthens. (The flaw I looked for wasn't there; the result is conservative-robust.)

### What am I missing? / next iteration
- **The microstructure chapter is now definitively closed.** Every angle — directional alpha, vol forecasting, regime
  filtering, lead-lag, institutional flow, deep book, and now execution — is measured and near-efficient/null. The advisor's
  L3-from-L2 and volume-as-information ideas are answered concretely and honestly: real contemporaneous information, no
  actionable edge of any kind. Further microstructure signal-hunting is not warranted.
- The one genuinely open data task remains the **point-in-time / survivorship-free universe** for the reversion backtest
  (iter-8 plan item 1) — delisted coins (LUNA/FTT/...) are absent; the no-stop result is stress-robust but a clean
  point-in-time re-run would close the last gap.
- Also worth one clean iteration: **characterize the no-stop exposure as basket cointegration** (PCA/Johansen on survivor
  majors) — is it pairwise alpha or a slow market-neutral reversion factor? That would precisely name what the +2.51 is.

### Plan for next iteration (prioritized)
1. **Point-in-time universe** (include delisted symbols) → re-run the §3 stop/no-stop matrix; does no-stop survive a
   genuinely survivorship-free universe? Cleanest remaining check; needs a small delisted-coin data pull.
2. **Characterize the no-stop exposure** (basket cointegration / PCA factor) — name the +2.51 honestly.
3. `L2_FINDINGS.md` is the canonical paper draft; keep it as the single source of truth.

---

## Iteration 8 — consolidation + a MAJOR self-correction: iter-7's "loses even gross" was stop-specific

Two deliverables: (1) the paper-ready synthesis **`docs/L2_FINDINGS.md`** (all 8 iterations, with the two artifact
demonstrations front-and-center); (2) the robustness work I flagged as the last loophole — and it overturned my own
iter-7 headline. This is exactly the "find a major flaw in the last analysis" the loop is for; the flaw was mine.

### A. The robustness matrix (`scratch/wf_robustness.py`) — the stop rule is the entire story
Re-ran the §3 backtest varying stop / exit / hedge / sizing, each vs a random-pair placebo:
- **Removing the |z|=4 stop flips net Sharpe −2.25 → +2.51 (gross −1.15 → +2.65)**, beating the random-pair placebo
  (+0.89) by ~2× across-seed SD. Time-exit (+1.87) and hold-to-convergence (+2.68) confirm it; all artifact-free static-z.
- **So iter-7's "loses even gross, worse than random" was specific to the asymmetric |z|=4 stop, not a property of the
  strategy.** A tight stop is *actively harmful* to mean reversion — it realizes losses exactly when the spread is most
  stretched and most likely to revert. I overstated the negativity in iter-7; **corrected in the log and L2_FINDINGS.md.**
- Rolling-hedge cells score +3–4 but their placebos run +2.1–2.6 (the §6 rolling-z artifact) — do not lead with them.

### B. Is the no-stop "win" real or a survivorship artifact? Stress-tested it myself (`scratch/wf_nostop_stress.py`)
My first instinct was "no-stop only works because the universe is survivorship-filtered (current top-50, every coin
survived to 2026) so hold-to-convergence never meets a permanent decoupling." **I tested that instead of asserting it,
and I was mostly wrong about the mechanism:**
- Injected delisting-scale permanent breaks (a fraction p of selected pairs diverge ~86% in one leg, never revert;
  selection on the unbroken train). The no-stop Sharpe is **robust** — only collapses to ≈0 (+0.12) at an *extreme*
  20%-of-pairs-per-quarter near-delisting rate; at plausible ≤5%/qtr it stays +1.9–2.4. Diversification across 10 pairs
  absorbs breaks. The stopped strategy stays ≈−2.3 throughout. **The no-stop edge is NOT primarily a delisting-tail artifact.**
- **The real caveats, from the diagnostics:** the no-stop run is a *multi-month hold* — **median holding ≈848 bars (~35
  days); 78% of positions never converge within the 3-month window;** **portfolio max drawdown −41%** (worst pair-window
  −77.5%); and **~⅓ of the return is generic survivor co-movement** (random-pair placebo +0.89; OU adds the +1.62 edge).
  Its hourly Sharpe is also flattered by long-hold autocorrelation (168h-block bootstrap understates SE since holds ≫ block).

### Reconciliation & the corrected bottom line
Reversion is real and selectable (iter-7 A, ρ=0.46) **and monetizable — but only in a form far from the project's stated
design:** a slow, multi-month, −41%-drawdown hold-the-spread exposure on a survivorship-filtered universe, partly generic
beta. As the **hourly, stop-managed stat-arb** originally specified it **loses** (−2.25), because the stop that bounds
hourly risk is exactly what destroys the thin, slow reversion. So the honest headline is the **sensitivity itself**: P&L
is entirely a function of the stop/exit rule (and window regime), not a single Sharpe. This is more nuanced, more correct,
and more interesting than iter-7's flat "untradeable." The four other results (cointegration artifact, microstructure
near-efficiency, L2 costs, rolling-z artifact) are unchanged.

### What am I missing? / next iteration
- **The single most valuable remaining data work is a point-in-time universe** (include delisted coins: LUNA, FTT, etc.).
  Every reversion/backtest number here is on the current top-50 = survivors. The no-stop result is stress-robust, but a
  true survivorship-free re-run would close the last honest gap and is worth doing now that ~175GB is cached.
- Could test whether the no-stop edge is just long-horizon basket co-integration (PCA/Johansen on the survivor majors) —
  i.e. is "hold bounded survivor spreads" the same as a slow market-neutral mean-reversion factor? If so, frame it as such.
- The microstructure chapter is closed (near-efficient, exhausted). L2_FINDINGS.md is now the canonical paper draft.
- Make sure the advisor sees both `CORRECTION_kalman_cointegration.md` and the §3 stop-sensitivity (it changes the verdict
  from "untradeable" to "depends entirely on the risk rule; the specified hourly version loses").

### Plan for next iteration (prioritized)
1. **Point-in-time / survivorship-free universe** (delisted coins included) → re-run the §3 stop/no-stop matrix; does the
   no-stop result survive a genuinely point-in-time universe? This is the cleanest remaining check.
2. **Characterize the no-stop exposure**: is it a slow market-neutral reversion factor (basket cointegration) rather than
   pairwise alpha? PCA/Johansen on the survivor majors; report honestly.
3. Polish `L2_FINDINGS.md` into the actual paper section; ensure the stop-sensitivity table and both artifacts are central.

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
