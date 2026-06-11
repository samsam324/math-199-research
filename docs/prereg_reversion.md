# Pre-registration: single-config evaluation of the never-stop reversion book

**Locked 2026-06-10, before running.** The git commit of this file timestamps the lock. The
configuration below is fixed. We will run it once on the held-out window, report whatever the
metric comes out to, and not retune after seeing the result. This is the credibility capstone
for Section 5.

> **Correction before running (2026-06-10).** The initial lock specified a rolling 240h
> z-score. That conflates with the rolling-z artifact of Section 3 and is not the configuration
> that produces the effect; the validated never-stop book uses a static train-window z. Corrected
> below before any evaluation run. No result had been observed at the time of this edit.

## Why

The honest ~1.0 monthly Sharpe of the never-stop book is the product of a long search across
stop, exit, pair-count, and selection choices (`docs/L2_FINDINGS.md` Result 3). Its
deflated-Sharpe significance survives the no-stop family but fails the whole stop-vs-no-stop
search. A single locked configuration, evaluated once, removes the "best of many configs"
problem for that one config. It does not undo the original selection — that caveat stays in
Section 6 — but it pins the headline to one pre-specified number.

## Locked configuration

- **Universe:** the 40 most liquid Binance.US USDT pairs by median hourly quote volume over the
  selection window. Fixed count, no discretion.
- **Selection:** rank candidate pairs by in-sample OU half-life (faster reversion = better) on
  the selection window only; take the top 40. Static OLS hedge ratio, fit on the selection
  window and frozen.
- **Spread / signal:** static-OLS log-price spread with a **static train-window z-score** --
  the spread mean and standard deviation are estimated on the selection window and frozen,
  $z = (s - \mu_{\text{train}})/\sigma_{\text{train}}$. No rolling re-centering, which would
  invoke the mechanical artifact of Section 3.
- **Entry:** open when flat and $|z| \ge 2$.
- **Exit:** close on convergence ($|z| \le 0.5$) or sign flip. **No stop-loss.**
- **Risk control:** structural-break circuit breaker — halt a pair permanently after any single
  leg moves more than 50% against the position in one bar (the delisting/collapse guard).
- **Capital:** equal dollar per leg per pair, market-neutral (dollar-balanced legs).

## Held-out evaluation window

- **Selection window:** start of data through 2025-05-31.
- **Evaluation window:** 2025-06-01 through 2026-05-18 (the most recent ~12 months). Selection,
  hedge fit, and liquidity ranking use only data up to 2025-05-31; the book then trades the
  evaluation window with frozen choices.

## Pre-specified metric and decision rule

- **Primary metric:** realized monthly Sharpe on the evaluation window, Newey--West (HAC)
  corrected, with a block-bootstrap 95% CI.
- **Reported alongside:** number of trades, median hold, max drawdown, beta to an equal-weight
  crypto index, and the deflated-Sharpe context for this single config.
- **Decision rule, fixed in advance:** we report the realized HAC monthly Sharpe and its CI
  whatever it is. We do not retune, reselect, or re-window after seeing it. A result near the
  ~1.0 prior strengthens Section 5; a materially lower result is reported as a degradation and
  weakens the claim. Either outcome is the finding.

## Reproduce

Runner: `scratch/prereg_run.py` (to be written and committed after this lock). It must read its
configuration from this file's values and write `scratch/prereg_result.json` plus a log. No
parameter in the runner may differ from the locked values above.
