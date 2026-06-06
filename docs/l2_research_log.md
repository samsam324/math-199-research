# L2 microstructure research log

Running log for the self-paced research loop. Newest iteration on top. Each
entry: what was done, what was found, what I think I'm missing, and the plan for
the next iteration.

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
