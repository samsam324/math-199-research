# First Branch Results

Run command:

```bash
python3 scripts/run_first_branch.py --out-dir artifacts/first_branch
```

The branch builds the Binance statistical-arbitrage workflow through liquidity
filtering, pair selection, feature engineering, shared ML dataset construction,
model training, and common evaluation.

## Data and Selection

- Local symbols: 195
- As-of universe at 2026-01-01 00:00:00 UTC: 195
- Liquidity-filtered universe: top 50 symbols by mean USDT volume
- Training close panel: 3517 hourly rows by 50 symbols
- Strict cointegration filters found no passing pairs on the local Binance.US
  slice, so the pipeline used the documented high-correlation fallback selector
  for downstream modeling.
- Selected fallback pairs: 20
- Feature rows: 77,278
- ML samples: 73,938
- Sequence shape: 73,938 x 168 x 16

## Metrics

The reported PnL metrics are in log-spread units and are intended for relative
model comparison, not as a transaction-cost-complete trading backtest.

| Model | Accuracy | Macro F1 | Trades | Total PnL | Sharpe-like | Max Drawdown | Win Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zscore_rule | 0.2060 | 0.1821 | 4019 | 12.1627 | 1.3637 | -4.1925 | 0.5673 |
| sklearn_hist_gradient_boosting | 0.4856 | 0.3186 | 4996 | 5.1814 | 1.2670 | -5.4414 | 0.5222 |
| lstm | 0.5152 | 0.3476 | 5000 | 10.5653 | 2.6012 | -4.6387 | 0.5474 |
| transformer | 0.5708 | 0.3930 | 5000 | 18.8906 | 4.7487 | -2.9760 | 0.6048 |

## Interpretation

On this run, the transformer produced the strongest classification and
spread-signal metrics, followed by the LSTM. The classical z-score rule had weak
classification accuracy but still produced positive spread-signal PnL in this
comparison. The gradient-boosting baseline was directionally useful but weaker
than the sequence models on this dataset.

The main caveat is that the strict cointegration screen did not find passing
pairs, so these results should be treated as a pipeline/modeling baseline over
correlated liquid pairs rather than evidence of true cointegrated statistical
arbitrage.
