# Entry/Exit State Machine Backtest Results

The bar-by-bar backtest in `docs/long_pipeline_results.md` treated every
classifier flicker as a position change, charging cost on each one. That
made the cost-wall headline ("every model loses to 15 bps") partly an
artifact of the trade definition.

`src/backtest.py` now supports a proper entry/exit state machine:

```
open when:  flat AND pred_class == 0 (revert) AND |spread_z| >= entry_z
              -> position = -sign(spread_z)
close when: |spread_z| <= exit_z, OR pred_class == 2 (diverge), OR
            spread sign flips relative to entry sign
hold when:  pred_class == 1 (persist), do nothing
```

The Sharpe denominator was also fixed: it was previously dividing per-bar
pnl by a time-varying "deployed capital" denominator, which produced
non-comparable bar returns and could yield positive Sharpe on a negative
total return. Returns are now expressed on a fixed capital base
`N_pairs * leg_notional * 2`, matching standard portfolio convention.

## Cost sensitivity (long Kalman pipeline, 27 walk-forward splits, entry_z=2.0, exit_z=0.5)

| Round-trip cost per leg | LSTM Sharpe | LSTM total $ | zscore Sharpe | zscore total $ | boost Sharpe | boost total $ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **0 bps** | **8.06** | **+$125,340** | 7.83 | +$125,116 | 7.81 | +$124,697 |
| **5 bps** | 0.69 | +$10,604 | 0.74 | +$11,654 | 0.67 | +$10,557 |
| **15 bps** | -14.25 | -$218,870 | -13.73 | -$215,270 | -13.85 | -$217,723 |

Break-even cost is approximately 5 bps round-trip per leg. Below that, the
strategy makes money; above it, the spread reversion is not strong enough
to overcome the cost drag.

At a more conservative `entry_z = 2.5`:

| Round-trip cost per leg | LSTM Sharpe | LSTM total $ | zscore Sharpe | zscore total $ |
| --- | ---: | ---: | ---: | ---: |
| 0 bps | 4.89 | +$66,010 | 4.80 | +$66,001 |
| 5 bps | 0.30 | +$4,029 | 0.32 | +$4,367 |
| 15 bps | -9.01 | -$119,934 | -8.81 | -$118,901 |

Higher entry threshold reduces trade frequency by ~50% but does not change
break-even cost, because the per-trade economics are similar. The signal is
about the same size per trade across the entry-threshold range; what
changes is how often you act on it.

## What this lets the paper say

- **Pre-cost, the spread mean-reversion signal exists.** Sharpe ~8 with
  LSTM/boost/z-score at zero cost; ~61% win rate per trade. The Kalman
  dynamic spread carries genuine information about future spread direction.
- **The break-even round-trip cost is ~5 bps per leg.** Binance.US default
  taker fee is 10 bps per side, so the strategy is well above break-even
  under realistic taker fees. Maker fees (typically 1-2 bps per side, or
  zero with rebates) put it just under.
- **No ML model dominates classical at the trading level once costs are
  applied.** Zero-cost: LSTM marginally best (Sharpe 8.06 vs 7.83 for
  z-score) but within noise. With any non-trivial cost, all three (boost,
  LSTM, z-score) collapse to similar break-even-ish performance with the
  same operating-point trade-off seen at the spread-signal level.
- **Transformer ranks below boost/LSTM/z-score at every cost level**,
  consistent with the walk-forward classification finding that it's
  statistically indistinguishable from random at this sample size.

## What this does NOT let the paper say

- **Nothing about market impact.** Slippage is modeled as a flat 2.5/5 bps
  per leg, not as a function of trade size. Real impact for $10k notional
  on liquid USDT pairs is probably small but the L2 data needed to model
  it properly is blocked on UCLA access.
- **Nothing about non-Binance.US execution.** Pulling Coinbase or Kraken
  data might give a different cost profile and a different break-even.
- **Nothing about funding costs on the synthetic short leg.** Spot USDT
  shorting on Binance.US isn't free; the model treats short legs as
  costless to hold. In practice this would mean margin or perp futures
  with their own funding rates.

## Reproduce

```bash
# Zero cost (signal sanity check)
python3 scripts/run_portfolio_backtest.py \
  --predictions-path artifacts/walk_forward_kalman_long/walk_forward_predictions.parquet \
  --pairs-path artifacts/kalman_long/selected_pairs.parquet \
  --dataset-dir artifacts/kalman_long/dataset \
  --out-dir artifacts/backtest_sm_zerocost \
  --state-machine --sm-entry-z 2.0 --sm-exit-z 0.5 \
  --taker-fee-bps 0 --slippage-bps 0

# Maker fees ~5 bps round-trip
python3 scripts/run_portfolio_backtest.py \
  --predictions-path artifacts/walk_forward_kalman_long/walk_forward_predictions.parquet \
  --pairs-path artifacts/kalman_long/selected_pairs.parquet \
  --dataset-dir artifacts/kalman_long/dataset \
  --out-dir artifacts/backtest_sm_5bps \
  --state-machine --sm-entry-z 2.0 --sm-exit-z 0.5 \
  --taker-fee-bps 2.5 --slippage-bps 2.5

# Binance.US default 15 bps round-trip
python3 scripts/run_portfolio_backtest.py \
  --predictions-path artifacts/walk_forward_kalman_long/walk_forward_predictions.parquet \
  --pairs-path artifacts/kalman_long/selected_pairs.parquet \
  --dataset-dir artifacts/kalman_long/dataset \
  --out-dir artifacts/backtest_sm_15bps \
  --state-machine --sm-entry-z 2.0 --sm-exit-z 0.5
```
