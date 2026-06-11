# Claims ledger

Every quantitative claim in `paper/main.tex`, with its source. Produced by the final
whole-paper fact-verification pass against the source docs. All claims verified; no fabrications
or unsourced numbers found. One acknowledged cross-table nuance is noted at the end.

| Claim | Value | Source |
|---|---|---|
| Kalman screen, real pairs (matched window) | 100% (p<.05), 96.4% (p<.001) | CORRECTION_kalman_cointegration.md:25 |
| Placebo (a) independent random walks | 100% / 98.3% | CORRECTION:26 |
| Placebo (b) phase-randomized surrogate | 100% / 95.0% | CORRECTION:27 |
| Placebo (c) leg block-shuffled | 100% / 95.0% | CORRECTION:28 |
| Clean static-OLS ADF (matched window) | ~3% (3.2%) | CORRECTION:38 |
| Clean Engle-Granger (matched window) | 2-3% (2.1%) | CORRECTION:37 |
| Frequency table real / RW placebo (1H/4H/1D) | 100/100/70 vs 100/100/68.3 | timeframe_robustness.md:16-18 |
| Frequency clean EG / static OLS (1H/4H/1D) | EG 25/11.7/5; OLS 41.7/20/8.3 | timeframe_robustness.md:16-18 |
| Rolling-z floor per event (1H/4H/1D) | +0.98/+1.61/+1.51 (sd 0.26/0.39/0.45), 200 surrogates | timeframe_robustness.md:35-37 |
| Rolling-z trading Sharpe / placebos | 4.75 / 2.18 / 2.42 | L2_FINDINGS.md:610 |
| Reversion selectability Spearman rho | 0.46 [0.37,0.54], p<.001, positive in all 9 splits | L2_FINDINGS.md:159 |
| Top-bottom quintile OOS excess reversion | +0.69z [0.58,0.81] | L2_FINDINGS.md:161 |
| Persistence placebos | random quintile +0.0001z; shuffle rho -> -0.002 | L2_FINDINGS.md:162 |
| rho=0.46 best-of-four metrics, not corrected | yes | L2_FINDINGS.md:456 |
| Institutional/mid/retail 1s impact (BTC) | 0.33/0.23/0.13 bps; institutional ~2.0-2.6x retail | L2_FINDINGS.md:480 |
| Book-OFI incremental R^2 (BTC/ETH) | +0.12 to +0.16 | L2_FINDINGS.md:494 |
| Cancellation share of best-quote reductions | 81-85% | L2_FINDINGS.md:495 |
| Tradeable-horizon institutional flow | +0.03% incremental R^2, reversal sign | L2_FINDINGS.md:486 |
| Hidden/iceberg liquidity | ~1-4% of at-best volume | L2_FINDINGS.md:567 |
| Execution costs (cross/passive/signal/random/oracle) | +1.34/+2.79/+1.98/+1.88/-0.23 bps | L2_FINDINGS.md:524 |
| Execution: orders, signal correlation, oracle swing | 18,432 orders; corr +0.06; +1.57 bps | L2_FINDINGS.md:516,540 |
| Microstructure null robust across 2024 (incl Aug 5) | holds | L2_FINDINGS.md:555 |
| L2 cost saving (on 5 bps slippage component) | 17-23% | L2_FINDINGS.md:592 |
| Survivor beta / top-10 PCA share | -0.06 / 1.7% | L2_FINDINGS.md:274,281 |
| Stop |z|=4 net Sharpe | -2.25 | L2_FINDINGS.md:192 |
| Hold / non-convergence / co-movement floor | ~35 days / 78% / ~1/3 | L2_FINDINGS.md:213 |
| Sharpe chain (naive / reimpl / HAC Binance / Coinbase / honest) | 2.5 / 2.0-2.2 / ~1.4 / 0.85-0.90 / ~1.0 | L2_FINDINGS.md:422,430,48 |
| Pre-registered held-out monthly Sharpe (no-stop) | 5.17, 9 months | prereg_result.md:19 |
| Held-out micro-cap pair-window legs | +300 to +517%; ~150x | prereg_result.md:38 |
| Structural-break circuit breaker | halt after >50% adverse leg; ~no Sharpe cost | L2_FINDINGS.md:407 |
| Deflated Sharpe | survives no-stop family, fails whole stop-vs-no-stop search | L2_FINDINGS.md:362 |
| Four clean-room reimplementations, no look-ahead | yes | L2_FINDINGS.md:418 |
| Costs | 15 bps/leg (10 fee + 5 slippage), ~30 bps round-trip | prereg_result.md:9 |
| HAC example (spurious 5-min reversal) | t=-22.8 -> -1.4 | L2_FINDINGS.md:115 |
| Survivorship break-injection (gross no-stop) | erased only near ~20%/qtr (~80%/yr) attrition | L2_FINDINGS.md:232-242 |
| Rolling-z trailing-demean bias (analytic) | negative autocorrelation of order 1/L (known) | textbook/analytic |

Nuance (acknowledged in the paper): the real-pair p<.001 rate is 96.4% in Table 1 (matched 2024
90d/30d window, 250-pair subsample) and 100% in Table 2 (recent multi-year window, 60 pairs/cell).
Different experiments, not a contradiction; the paper states this. No blockers; both guardrails
hold (the survivor leads with its bound; the two artifacts and the rho=0.46 result are prominent).
