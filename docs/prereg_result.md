# Pre-registered reversion test — result

Config locked in `docs/prereg_reversion.md` (commit `bd4909b`, before running). Runner
`scratch/prereg_run.py`, diagnostic `scratch/prereg_diag.py`, outputs
`scratch/prereg_result.json`, `scratch/prereg_*.log`. Binance.US hourly, top-50 universe.

## What we ran

The locked parameters (40 OU-selected pairs, static train-window z, $|z|\ge2$ entry,
$|z|\le0.5$ exit, no stop, 50%-adverse circuit breaker, 15bps/leg) on the held-out window, two
ways: a single 4.4-year selection, and the validated rolling 6mo-train / 3mo-test selection
restricted to test windows on or after 2025-06-01.

## Results

| Evaluation | pairs/window | months | monthly Sharpe | HAC hourly-ann (95% CI) | maxDD | beta |
|---|---:|---:|---:|---|---:|---:|
| single-window, no-stop | 2 | 7 | 1.60 | 0.95 [-0.21, 1.89] | -18% | +0.15 |
| rolling held-out, no-stop | 40 | 9 | 5.17 | 5.85 [4.57, 7.20] | -11% | -0.11 |
| rolling held-out, +circuit breaker | 40 | 9 | 5.04 | 5.90 [4.53, 7.40] | -12% | -0.12 |

The circuit breaker barely moves the result (no qualifying break fired in this window).

## Reading it — the held-out Sharpe is inflated, not a clean alpha

Two honest findings, neither of which confirms a deployable ~1.0:

1. **Over a single long window, almost nothing is mean-reverting in band.** A 4.4-year OU
   selection yields only **2** qualifying pairs, and their Sharpe is not significant (CI spans
   zero). Long-horizon reversion in the tradeable band is rare, consistent with the near-null
   cointegration of Section 3.

2. **The rolling held-out Sharpe (~5) is a micro-cap / volatility artifact, not a tradeable
   edge.** It is broad-based, not one pump (top-3 pairs are 13% of PnL; dropping them leaves
   Sharpe 5.0), but it is concentrated in **high-volatility, low-liquidity micro-cap alts**
   (WAXPUSDT, ZILUSDT, VTHOUSDT, ACHUSDT, EGLDUSDT). Individual pair-windows post +300% to
   +517% in three months, and the most volatile window (2025-10 to 2026-01) shows a monthly
   Sharpe near 21. Returns of that size (a +517% leg implies $e^{5.2}\approx 150\times$) are not
   realizable: a flat 15bps cost wildly understates the true cost of trading these names at
   size, and unit-notional accounting ignores capacity. The number is an accounting artifact of
   cheap costs on illiquid coins.

This is the paper's own thesis applied to its marginal survivor. The literal backtest produces
a spectacular Sharpe that does not survive realistic scrutiny. The in-sample figure was already
corrected from ~2.5 to ~1.0 only after cross-venue validation (Coinbase ~0.9-1.2, where these
micro-caps are absent or tamer) and removing hindsight pumps; the held-out Binance run, on a
volatile recent regime with a 40-pair micro-cap-heavy universe, reproduces the **uncorrected,
inflated** version more extremely.

## Honest decision (per the pre-registered rule)

We report the realized number and its drivers. It is a loss of credibility for the raw
backtest, not a confirmation of alpha. Section 5 leads with the bound and the fragility: there
is a real, market-neutral, selectable reversion signal, but its monetizable magnitude is highly
uncertain and backtest-inflated. The venue-robust honest figure stays near ~1.0 as an upper-ish
estimate, and even that rests on unit-notional accounting that ignores micro-cap capacity. The
defensible claim is the sensitivity itself, not a point Sharpe.
