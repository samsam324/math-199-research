# Remote tasks for the Windows agent (updated 2026-06-19)

The Mac side has applied the bulk of the paper revision in `paper/main_v2.tex`. The
remaining work is the parts that need either the L2 data or matplotlib runs against the
committed CSVs. Read `paper/main_v2.pdf` to see exactly where each new figure and table
goes (placeholders are visible bordered boxes with the data source path printed inside).
Full revision plan is in `docs/PAPER_REVISION_TASKS.md`; this file is just the action list.

## STATUS: done on the Windows side (2026-06-19)

All figures generated and wired, both stub tables filled, paper promoted. `main_v2.tex` has
been promoted to `paper/main.tex` (old 4-ref draft preserved in git history); compiles to a
22-page `main.pdf` with 0 undefined refs/citations, 0 overfull boxes, 0 em-dashes, and 25/25
bib entries cited. Generators: `scratch/make_figures_v2.py` (all 9 figures), wired by
`scratch/wire_v2_figures.py`, with adversarial-review corrections in `scratch/review_fixes.py`.

What changed vs. this list:
- Figs 6 (OFI decay) and 9 (cancellation) were built from the existing numbers already in
  `scratch/book_ofi_2024.log` (master tables [1]/[2]/[3]), so **no L2 rerun was needed**. The
  OFI decay figure therefore spans horizons {contemporaneous, 1, 5, 10, 30}s (the log's range),
  which matches the caption's "near zero by 30 seconds." Extending to 60/120/300s would need a
  live `book_ofi_incremental.py` run against raw L2.
- T3 (institutional impact) headline cells verified exactly against `impact_decomp_2024.log`
  (h=1 RAW bps); no change needed.
- Adversarial review fixed two real caption errors (circuit-breaker "collapses past 10%/yr" was
  wrong; the no-breaker Sharpe crosses zero near 46%/yr) and one data slip (the exec-signal
  correlation is +0.03 in the committed log, not +0.06).
- **Open for Sammy:** T1 (`tab:venue-comparison`) Binance figures 2.18/1.40 are not in a
  committed log (Coinbase 0.88 matches `cross_exchange.log`); confirm the source.
- **Open for Jack:** the title is the suggested one, not `TBD`.

## What `main_v2.tex` already has

- Suggested title (replace if you prefer something else)
- Bibliography expanded from 4 to 23 entries
- New `\section{Related work}` with four subsections + "what is new here" closer
- Honest contributions paragraph in the intro
- Methods appendix A.1 through A.10 before `\end{document}`
- 8 new figure environments with `\figplaceholder` boxes (visible in the PDF, ready to
  swap for real images)
- 3 new tables, of which T1 (`tab:venue-comparison`) is filled with real numbers and
  T2/T3 have row labels and `TBD` cells

## What you need to do

### 1. Generate the figures (priority order)

For each one, replace the `\figplaceholder{...}{...}` line with the corresponding
`\includegraphics[width=...]{filename}` once the file exists. The Python sketches are in
`docs/PAPER_REVISION_TASKS.md` Section 4.

| Tag in main_v2 | Source data | Needs L2 data? |
|---|---|---|
| `fig_kalman_innovations_white.pdf` | rerun `scratch/kalman_positive_control.py` to dump per-bar innovations for one real pair + one RW placebo | no (uses hourly) |
| `fig_poscontrol_vs_negcontrol.pdf` | `scratch/kalman_positive_control.csv` (already on disk) | no |
| `fig_reversion_persistence_scatter.pdf` | `scratch/persistence_pairs.csv` (already on disk) | no |
| `fig_circuit_breaker_retention.pdf` | `scratch/survivorship_adjusted_sharpe.csv` (already on disk) | no |
| `fig_forced_collapse_perpair.pdf` | `scratch/forced_collapse.csv` (already on disk) | no |
| `fig_hac_inflation.pdf` | `docs/hac_sharpe_per_split.csv` (already on disk) | no |
| `fig_microcap_adv.pdf` | `scratch/microcap_adv.csv` (already on disk) | no |
| **`fig_ofi_decay.pdf`** | rerun `scratch/book_ofi_incremental.py` with horizons `[1,2,5,10,30,60,120,300]` s | **YES (L2)** |
| `fig_cancellation_share.pdf` (nice-to-have) | parse `scratch/book_ofi_2024.log` cancel-share fields, or rerun `scratch/book_ofi_cancel_stretch.py` | **YES (L2)** |

### 2. Reshape `fig_freq_invariance.pdf`

Currently duplicates Table 3. Replace with a placebo-gap view (real minus random walk pass-rate,
per frequency, for both Kalman and clean Engle-Granger). Detail in `PAPER_REVISION_TASKS.md`
Section 4 "Existing figure: reshape" subsection.

### 3. Fill in the two stub tables

- **T2 (`tab:exec-symbol-breakdown`)**: data is in `scratch/exec_value_2024_summary.csv`.
  Replace the `TBD` cells with per-symbol Aggressive and Signal-timed costs at \$10k and
  \$50k notional.
- **T3 (`tab:institutional-impact`)**: the headline per-symbol numbers are already in the
  cells (BTC `0.13/0.23/0.33`, ETH `0.15/0.24/0.32`, SOL `0.15/0.26/0.32`, AVAX `0.18/0.38/0.38`)
  but please verify against `scratch/impact_decomp_2024.log` and update if any are off.

### 4. When satisfied, promote and recompile

```
git mv paper/main_v2.tex paper/main.tex      # overwrite old main.tex
pdflatex main.tex && pdflatex main.tex       # second pass for refs
```

Or keep `main_v2.tex` separate and decide later. Either is fine.

### 5. Verify before commit

- Every `\cite{}` resolves (no "?" in the PDF)
- Every `\ref{}` resolves
- All figure files referenced in `\includegraphics{}` exist
- Word count and figure count meet the three-student-project length target Mihai flagged

That's it. The bibliography, related work, contributions, and appendix don't need any more
work from the Mac side. Once the figures land and the tables fill in, the paper is ready
for Mihai's review.
