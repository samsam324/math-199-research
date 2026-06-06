# L2 microstructure research log

Running log for the self-paced research loop. Newest iteration on top. Each
entry: what was done, what was found, what I think I'm missing, and the plan for
the next iteration.

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
