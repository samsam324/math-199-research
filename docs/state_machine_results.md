# Entry/exit state machine backtest

Bar-by-bar backtest treats every classifier flicker as a trade. State
machine collapses that into discrete open/close.

Rule:
- open when flat AND pred=0 AND |spread_z|>=entry_z; position = -sign(spread_z)
- close when |spread_z|<=exit_z OR pred=2 OR spread sign flips
- hold on pred=1

Sharpe denominator: fixed capital base `N_pairs * leg_notional * 2`. The
old time-varying deployed denominator could yield positive Sharpe on
negative total return (bug, fixed).

## Cost sensitivity, entry_z=2.0, exit_z=0.5

| Round-trip per leg | LSTM Sharpe | LSTM $ | zscore Sharpe | zscore $ | boost Sharpe | boost $ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 bps | 8.06 | +125,340 | 7.83 | +125,116 | 7.81 | +124,697 |
| 5 bps (maker) | 0.69 | +10,604 | 0.74 | +11,654 | 0.67 | +10,557 |
| 15 bps (Binance.US taker) | -14.25 | -218,870 | -13.73 | -215,270 | -13.85 | -217,723 |

- Break-even ~5 bps round-trip per leg
- Binance.US taker (10 bps/side, 20 bps round-trip per leg) destroys it
- Binance.US maker (~5 bps round-trip per leg) at the boundary

## entry_z=2.5, exit_z=0.5

| Cost | LSTM Sharpe | LSTM $ | zscore Sharpe | zscore $ |
| --- | ---: | ---: | ---: | ---: |
| 0 bps | 4.89 | +66,010 | 4.80 | +66,001 |
| 5 bps | 0.30 | +4,029 | 0.32 | +4,367 |
| 15 bps | -9.01 | -119,934 | -8.81 | -118,901 |

- Conservative entry: ~50% fewer trades, similar break-even
- Per-trade economics similar across entry thresholds

## Paper claims this supports

- Spread mean-reversion signal exists pre-cost (Sharpe ~8, 61% win rate)
- Cost ceiling ~5 bps round-trip per leg
- No ML model meaningfully beats classical at the trading level once
  costs are real
- Transformer ranks below boost/LSTM/zscore at every cost level

## What this does NOT cover

- No market impact (flat slippage; need L2)
- No funding on synthetic shorts (spot USDT shorting on Binance.US
  isn't free in practice)
- No cross-exchange comparison

## Reproduce

```
python3 scripts/run_portfolio_backtest.py \
  --predictions-path artifacts/walk_forward_kalman_long/walk_forward_predictions.parquet \
  --pairs-path artifacts/kalman_long/selected_pairs.parquet \
  --dataset-dir artifacts/kalman_long/dataset \
  --out-dir artifacts/backtest_sm_{0,5,15}bps \
  --state-machine --sm-entry-z 2.0 --sm-exit-z 0.5 \
  --taker-fee-bps {0,2.5,10} --slippage-bps {0,2.5,5}
```
