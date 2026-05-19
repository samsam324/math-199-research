# Portfolio Backtest Architecture Notes

## Why this isn't running through minitron's engine

minitron's strategy DSL (SSS) and underlying C++ engine are built around long-
only equity ETF strategies on a fixed universe of US-listed instruments
(`BROAD_ASSET_CLASSES`, `SECTORS`, `FACTORS`, etc.). Every reference strategy
in `minitron/research/strategies/` imports tickers like `SPY`, `XLF`, `PPC`,
`CALM` and uses primitives (`adj_close`, `pct_change`, `roll_mean`) that
assume a single-asset long-or-flat position model.

Our strategy is structurally different:
- Two-leg dollar-neutral pair trade with a synthetic short leg
- Time-varying hedge ratio applied per pair
- USDT spot universe that is not on minitron's allowed list
- Hourly bars on a 24x7 continuous market
- Engine binary is Windows/MinGW-built; we are on macOS

Translating this into SSS would require teaching the DSL new short and hedge
primitives, extending the universe constants, and rebuilding the engine on
Mac. That is its own multi-week project, not a paper deliverable.

## What we shipped instead

`src/backtest.py` and `scripts/run_portfolio_backtest.py` implement a focused,
cost-aware portfolio backtester just for the crypto pair-spread case:

- Marked-to-market every bar, dollar-neutral by construction
- Taker fee (10 bps default) + slippage (5 bps default) charged on each $
  traded each bar
- Equal-notional sizing per pair, configurable cap on max simultaneous active
  pairs
- Outputs portfolio return stream, per-pair pnl, position history, trade
  ledger
- Metrics: annualized Sharpe, max drawdown ($ and %), total turnover, win
  rate, n trades

The return signature of `metrics_for_minitron(result)` matches the field names
minitron's `run_backtest` emits for equity strategies, so the paper's
comparison table stays apples-to-apples in shape with what minitron would
produce, and a future adapter (when/if the engine ever speaks crypto) can
plug in without changing the results section.

## What this backtester does NOT do

Some choices were deliberately deferred because they need their own design
work and are out of scope for the first results pass:

- **Non-overlapping discrete trades**: every bar is a marked-to-market
  rebalance. A real "open at z > 2, close at z < 0.5" rule needs entry/exit
  state machines per pair and changes the unit of analysis from bar to trade.
  Worth doing as a comparison, but not gating first results.
- **Capacity / market impact**: we charge a flat slippage in bps. A real
  capacity model needs L2 depth (blocked on UCLA data) and a square-root
  impact term.
- **Funding for synthetic shorts**: spot USDT shorting on Binance.US isn't
  free; in practice it would be margin or perp futures. The current model
  treats short legs as costless to hold, which understates real costs.
- **Cross-pair correlation in position sizing**: pairs are sized
  independently. A real portfolio would size by joint covariance of pair
  returns.

These limitations belong in the discussion / limitations section of the paper.
