# Long walk-forward (27 splits, Kalman pipeline, PRE-AUDIT)

Pre-audit run. Numbers superseded by `docs/RESULTS.md`. Kept for
provenance because it's the first run with 27 splits and the original
universe-leak fix.

t0=2024-01-01, 180d train, 750d test, 90/30/30 walk-forward, 130-symbol
universe -> 50 by liquidity -> 20 top correlation. 424k samples, 135k test.

## Per-model means (27 splits)

| Model | Acc | F1 | total_pnl | pnl_mean_to_std | win_rate | trades/split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lstm | 0.595 | 0.500 | 12.33 | 0.279 | 0.603 | 4621 |
| booster | 0.590 | 0.511 | 13.06 | 0.307 | 0.623 | 4466 |
| transformer | 0.391 | 0.295 | 3.86 | 0.047 | 0.520 | 4549 |
| majority | 0.379 | 0.183 | 5.51 | 0.148 | 0.565 | 5000 |
| random | 0.350 | 0.333 | 0.32 | 0.008 | 0.506 | 3828 |
| zscore | 0.338 | 0.277 | 7.63 | 0.282 | 0.960 | 591 |
| persist | 0.239 | 0.127 | 0.00 | 0.000 | 0.000 | 0 |

## Notes

- Booster slightly beats LSTM. Zscore matches LSTM on pnl_mean_to_std.
- Transformer CI included zero -> indistinguishable from random.
- HMM filter cuts PnL on every model.
- Cost wall: every model loses to 15 bps round-trip with the bar-by-bar
  backtester (now superseded by the state machine version).

## What changed post-audit

- Kalman parameter discontinuity bug fixed (training spread now uses
  fitted Q, not defaults). Booster pnl_mean_to_std 0.307 -> 0.356.
- Per-pair label threshold leak fixed (label_train_end = t0 now).
- 6 backtester sanity tests + 10 leakage tests added.

See `docs/RESULTS.md` for the post-audit numbers and
`docs/leakage_audit.md` for the audit writeup.
