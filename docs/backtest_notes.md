# Backtester notes

## Why not minitron

- minitron's SSS DSL + C++ engine assumes long-only equity ETF strategies
  on a fixed universe (SPY, XLF, PPC, CALM, ...)
- Our strategy: dollar-neutral pair trade, time-varying hedge, USDT spot,
  hourly continuous market
- Engine binary is Windows/MinGW; we're on macOS
- Translating would mean new DSL primitives + universe constants + Mac
  engine build. Not a paper deliverable.

## What we ship instead

`src/backtest.py` + `scripts/run_portfolio_backtest.py`:

- Marked-to-market every bar OR entry/exit state machine
- Dollar-hedged: $L on A, $L*beta on B
- Taker fee + slippage per side, bps of notional traded
- Equal-notional sizing per pair, configurable max active
- Outputs: portfolio returns, per-pair pnl, position history, trades
- Metrics: annualized Sharpe (fixed capital base), max drawdown, turnover,
  win rate, n trades. Field names mirror what minitron emits for equity
  strategies so a future adapter slots in.

## What it does NOT model

- Non-overlapping discrete trades on top of bar-by-bar state. The state
  machine collapses flicker into open/close events but does not enforce
  position-limit semantics across overlapping signals.
- Market impact: flat bps slippage, not a function of trade size. Needs L2.
- Funding for synthetic shorts: spot USDT shorting on Binance.US isn't
  free in practice (margin or perp futures). Model treats shorts as
  costless to hold.
- Cross-pair covariance in sizing: pairs sized independently.

These belong in the discussion / limitations section of the paper.

## Sharpe denominator (audit fix)

- Old: divided per-bar pnl by time-varying deployed capital
  (`active_pairs * leg_notional * 2`)
- Problem: bar returns weren't on a consistent capital base; could give
  positive Sharpe on negative total return
- Fixed: use constant capital base = `N_pairs * leg_notional * 2`
- Standard portfolio convention. Sanity tests pass with the fix.
