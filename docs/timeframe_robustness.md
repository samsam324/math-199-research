# Timeframe robustness of the two screening artifacts

Run: `scratch/timeframe_robustness.py` (seed 7). Log: `scratch/timeframe_robustness.log`.
Tables: `scratch/timeframe_kalman_artifact.csv`, `scratch/timeframe_rollingz_floor.csv`.
Data: Binance.US hourly close panel, top-50 USDT (`data/spot_1h`), 58,262 hourly bars
(2019-09 to 2026-05), resampled to 4h and 1D.

Question: are the Kalman-innovation cointegration artifact and the rolling-z mechanical-
reversion artifact specific to the hourly bar, or do they hold at coarser bars too? If they
hold, the "single interval (hourly)" limitation is closed for the two artifacts.

## Kalman cointegration screen — real pairs vs random-walk placebo

| freq | real p<.05 | real p<.001 | RW placebo p<.05 | RW placebo p<.001 | clean EG p<.05 | static OLS p<.05 |
|------|-----------:|------------:|-----------------:|------------------:|---------------:|-----------------:|
| 1H   | 100.0 | 100.0 | 100.0 | 100.0 | 25.0 | 41.7 |
| 4H   | 100.0 | 100.0 | 100.0 | 100.0 | 11.7 | 20.0 |
| 1D   | 100.0 |  70.0 | 100.0 |  68.3 |  5.0 |  8.3 |

n = 60 pairs per cell. The Kalman screen passes real pairs and independent random walks at the
same ~100% rate at every frequency. At daily the p<.001 rates are 70.0 (real) vs 68.3 (placebo),
statistically identical. The screen carries no cointegration information at any cadence.

The clean Engle-Granger benchmark is far lower and falls toward the 5% null floor as the bar
coarsens: 25% hourly over a multi-year window down to 5% daily. Its elevation at fine
frequencies reflects the common BTC trend shared by crypto majors over long windows, not
genuine pairwise cointegration. (The matched short-window test in `audit_part1.py`, t0=2024,
90d/30d, put it at 2-3%.) Either way it is nothing like the screen's 100%, and the gap is the
point.

## Rolling z-score mechanical reversion floor (pure random walks)

| freq | test bars | lookback | horizon | mean event reversion (z) | sd | n surrogates |
|------|----------:|---------:|--------:|-------------------------:|---:|-------------:|
| 1H   | 2160 | 240 | 48 | +0.98 | 0.26 | 200 |
| 4H   |  540 |  60 | 24 | +1.61 | 0.39 | 200 |
| 1D   |  220 |  30 | 10 | +1.51 | 0.45 | 200 |

A rolling z-score applied to a pure random walk produces strong positive mean-reversion at
every frequency (+0.98 to +1.61 z per |z|>2 event). Subtracting a trailing mean from a random
walk manufactures it. Frequency-invariant.

## Conclusion

Both artifacts hold at hourly, 4-hour, and daily. The "single interval (hourly)" limitation is
closed for the two screening artifacts; Section 3 reports them as cadence-independent.
