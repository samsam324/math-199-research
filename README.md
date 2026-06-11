# MATH 199 — White by Construction: Crypto Stat-Arbitrage Screens That Pass Their Own Placebos

UCLA Math 199 research project. Authors: Jack Lutz, Sammy Adham; advisor: Frank Kronewitter.

**The deliverable is the paper: [`paper/main.tex`](paper/main.tex)** (12 pages, builds with `pdflatex`). Everything else in this repo is the work behind it.

## What the project found

We set out to build a profitable hourly statistical-arbitrage strategy on crypto pair spreads. We did not, and the more useful result is why. The paper makes three points:

1. **Two screens in common use manufacture structure that isn't there.** A Kalman dynamic-hedge cointegration test reports out-of-sample cointegration on almost every liquid pair — but it tests a filter's own innovations, which are near-white by construction, and placebos that cannot be cointegrated pass at the same near-100% rate. A rolling z-score reports strong mean reversion, yet it reverts mechanically even on a pure random walk. Each is exposed by a single matched placebo, and the placebo is the contribution.
2. **The structure that does survive isn't tradable.** Order flow in the Level-2 book carries real information but decays within seconds; reversion speed ranks pairs (Spearman ρ = 0.46) but does not convert into profit.
3. **One market-neutral reversion book clears honest accounting at a monthly Sharpe near 1.0** — but only without a stop, only as a multi-week hold, and a literal held-out backtest inflates it to ~5 on untradeable micro-caps. The one pre-registered clean test has a confidence interval that spans zero, so we read 1.0 as an exploratory estimate, not a significant edge.

## If you knew the earlier version of this project

This began as a hunt for a profitable strategy, built on an ML pipeline (LSTM / gradient booster / transformer over Kalman spreads). That pipeline still lives in `src/` and `scripts/`, but the paper does not use it. Once we benchmarked the Kalman cointegration screen against a white-noise placebo, the old "central result" — 99.7% of pairs cointegrated out of sample — turned out to be an artifact of the filter, and the project pivoted to negative results and a market-efficiency reading. **`docs/RESULTS.md` is the pre-pivot record and is quarantined; do not cite it.** The paper supersedes it.

## Reproducing the paper's numbers

Environment: Python 3.11, `pip install -r requirements.txt`. Data lives under `data/` (gitignored): `spot_1h/` (Binance.US hourly USDT klines), `spot_1h_delisted/` (LUNA/UST/FTT/LUNC, for the collapse tests), `coinbase_1h/` (USD cross-check), and the Tardis Level-2 sample under `l2/` and `microstructure/`.

[`docs/claims_ledger.md`](docs/claims_ledger.md) is authoritative: it maps every number in the paper to its source. The scripts that generate them:

| Paper result | Script |
|---|---|
| Kalman screen passes placebos (Table 1) | `scratch/audit_part1.py` |
| Positive/negative control + whiteness (Table 3) | `scratch/kalman_positive_control.py` |
| Frequency robustness (Table 2) | `scratch/timeframe_robustness.py` |
| Rolling-z mechanical floor (Table 4) | `scratch/timeframe_robustness.py` |
| Artifact sensitivity to analysis choices | `scratch/sensitivity.py` |
| Reversion selectability (ρ = 0.46) | `scratch/persistence_test.py` |
| Microstructure / execution null (Table 5) | `phase2_l2/`, `scratch/exec_value_2024.py`, `scratch/book_ofi_2024.py` |
| Survivor Sharpe chain, stop vs no-stop | `scratch/wf_backtest.py`, `scratch/wf_robustness.py` |
| Four independent reimplementations | `scratch/subagent_independent_check.py`, `my_independent_run1.py`, `indep_run2.py`, `indep3_backtest.py` |
| Pre-registered held-out test | `scratch/prereg_run.py` (config locked in `docs/prereg_reversion.md`) |
| Forced-held-collapse tail | `scratch/forced_collapse.py` |
| Micro-cap capacity bound | `scratch/microcap_adv.py` |
| Survivorship-adjusted Sharpe + circuit breaker | `scratch/survivorship_adjusted_sharpe.py`, `scratch/nostop_breakstop.py` |

Each script prints the numbers it backs and, where relevant, writes a `.csv`/`.log` next to itself. Most run in seconds to a few minutes; the Kalman MLE fits (positive control) take ~2 minutes, and the backtest and survivorship sweeps a few minutes each.

Build the paper:

```bash
cd paper && pdflatex main.tex && pdflatex main.tex   # two passes for refs; 12 pages, 0 warnings
```

## What still needs a human

- An independent, from-scratch reproduction of the Kalman placebo — the central claim — by someone who did not write this pipeline.
- A true point-in-time universe that includes delisted coins; the on-disk store holds currently listed symbols, so it under-represents exits.
- Co-author and advisor sign-off before anything is posted publicly.

## Repository layout

```
paper/                     *** The deliverable ***
  main.tex                  The paper (12 pages, pdflatex, manual thebibliography)
  fig_freq_invariance.*     Kalman frequency-invariance figure
  fig_rollingz_noise.*      Rolling-z-on-a-random-walk figure

docs/
  claims_ledger.md          *** Every paper number mapped to its source ***
  timeframe_robustness.md   Frequency-robustness results
  prereg_reversion.md       Locked pre-registration config
  prereg_result.md          Held-out evaluation result
  L2_FINDINGS.md            Microstructure / execution analysis log
  RESULTS.md                Pre-pivot record (QUARANTINED — do not cite)

scratch/                    Analysis scripts behind the paper (see table above)
src/                        Library code (Kalman filter, pair selection, backtester, L2 store)
scripts/                    Pre-pivot ML pipeline drivers (superseded; not used by the paper)
phase2_l2/                  Level-2 microstructure pipeline
data/                       Local parquet stores (gitignored)
tests/                      Backtester sanity + leakage-audit tests
proposal.md                 Original proposal (a profit hunt; superseded by the paper)
```

## The earlier pipeline (superseded, preserved)

`src/` and `scripts/` hold the Phase-1 ML pipeline and the Phase-2 L2 scaffolding from before the pivot. They still run, but the paper does not depend on them. `docs/RESULTS.md` records what they produced and is kept for provenance only.
