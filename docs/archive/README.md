# Archived results

These files document earlier passes that are superseded by the current
findings in `docs/RESULTS.md`. They are preserved for git provenance and
to show the methodological evolution of the project (single chronological
split → short walk-forward → long walk-forward → Kalman pipeline).

If you are reading the project for the first time, **skip this directory
and read `docs/RESULTS.md` instead.**

## Contents and what superseded each

- `first_branch_results.md` — Frank Kronewitter's initial single-split
  results on the static-OLS pipeline. Superseded by `long_pipeline_results.md`.
- `baseline_comparison_results.md` — Frank's expansion adding naive
  baselines and a single-split walk-forward. Superseded by the 27-split
  walk-forward in `long_pipeline_results.md`.
- `may18_run_results.md` — first results on the static-OLS pipeline with
  the full HMM/Kalman/backtest pipeline. Superseded by Kalman pipeline.
- `kalman_pipeline_results.md` — first results on the Kalman pipeline
  with only 2 walk-forward splits. Superseded by the long 27-split
  pipeline.

The current authoritative files are:

- `docs/RESULTS.md` for headline findings
- `docs/METHODS.md` for methodology
- `docs/long_pipeline_results.md` for the full 27-split tables
- `docs/state_machine_results.md` for cost sensitivity
- `docs/survivorship_bias.md` + `docs/binance_us_delistings.md` for survivorship
- `docs/backtest_notes.md` for backtester architecture
