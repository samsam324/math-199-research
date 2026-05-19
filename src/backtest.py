"""
Portfolio backtester for cryptocurrency pair-spread signals.

Why this lives here instead of inside minitron
----------------------------------------------
minitron's SSS DSL and C++ engine are built around long-only equity ETF
strategies on a fixed predefined universe (SPY, XLF, XLK, ...). The crypto
pair-trade we run here is dollar-neutral with synthetic short legs, sized by
a time-varying hedge ratio, and lives on a free-form USDT spot universe.
Translating it into SSS would mean teaching the DSL several new primitives
and re-binding the C++ engine; meanwhile this self-contained module gives us
correct, cost-aware portfolio returns that can be lifted into a minitron
adapter once the engine ever speaks crypto.

What it does
------------
Convert per-bar class predictions into a marked-to-market portfolio:

    class 0 (revert)  -> short the spread sign (long the under-leg, short the
                         over-leg)
    class 1 (persist) -> flat
    class 2 (diverge) -> long the spread sign

Position sizing is constant gross notional per pair leg, equal-risk across
active pairs. Hourly returns are accumulated, transaction costs and slippage
are applied on every position change. Output is one return stream per pair
plus a portfolio return stream, with Sharpe / drawdown / turnover metrics
that match the shape minitron reports for equity strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    # Costs / slippage are per-side, in basis points of notional traded.
    taker_fee_bps: float = 10.0       # Binance.US default ~0.10% per leg per side
    slippage_bps: float = 5.0
    # Each pair gets this gross notional per leg. Both legs trade equal $.
    leg_notional: float = 10_000.0
    # Risk-free rate (annualized) for excess-return Sharpe. Crypto convention: zero.
    rf_annual: float = 0.0
    bars_per_year: int = 24 * 365      # hourly bars, continuous market
    # Cap on max simultaneous active pairs. None means no cap.
    max_active_pairs: Optional[int] = None
    # Min spread |z| to enter; 0 means always trade on the class signal alone.
    entry_z_threshold: float = 0.0
    # Optional regime mask (per-row bool); when set, only allow entries where True.
    apply_regime_mask: bool = False


@dataclass
class BacktestResult:
    per_pair_returns: pd.DataFrame      # index: timestamp, columns: pair, values: pnl per bar in $
    portfolio_returns: pd.Series        # index: timestamp, $ pnl
    positions: pd.DataFrame             # index: timestamp, columns: pair, values: signed signal {-1, 0, +1}
    trades: pd.DataFrame                # one row per position change
    metrics: Dict[str, float] = field(default_factory=dict)


def _class_to_signal(pred_class: np.ndarray, spread_sign: np.ndarray) -> np.ndarray:
    """class 0 = revert (against sign), 2 = diverge (with sign), 1 = flat."""
    pred_class = np.asarray(pred_class, dtype=int)
    spread_sign = np.asarray(spread_sign, dtype=float)
    signal = np.zeros_like(spread_sign, dtype=float)
    signal[pred_class == 0] = -spread_sign[pred_class == 0]
    signal[pred_class == 2] = spread_sign[pred_class == 2]
    # Coerce zero-sign positions (spread exactly at mid) to flat.
    signal[spread_sign == 0] = 0.0
    return signal


def _pair_returns(
    pair_id: str,
    pair_df: pd.DataFrame,
    close_a: pd.Series,
    close_b: pd.Series,
    beta: float,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """
    Build a per-bar pnl series for a single pair.

    Position convention:
      signal = +1  ->  long $L of A, short $L * (beta * p_a / p_b) of B
                       (dollar-neutral at entry; let it drift between rebals)
      signal = -1  ->  reverse
      signal = 0   ->  flat both legs

    For simplicity we rebalance every bar to maintain dollar-neutrality at the
    current prices, so $ pnl per bar = signal_{t-1} * L * (r_a - beta_eff * r_b),
    where beta_eff is rebalanced to (beta * p_a / p_b) at t-1 in dollar terms.

    Costs are charged on |delta position| in notional terms each bar.
    """
    df = pair_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")

    aligned = pd.DataFrame({"close_a": close_a, "close_b": close_b}).reindex(df.index).ffill()
    df["close_a"] = aligned["close_a"]
    df["close_b"] = aligned["close_b"]
    df = df.dropna(subset=["close_a", "close_b"])

    if df.empty:
        return pd.DataFrame(columns=["timestamp", "pair", "signal", "pnl", "turnover"])

    df["ret_a"] = df["close_a"].pct_change().fillna(0.0)
    df["ret_b"] = df["close_b"].pct_change().fillna(0.0)
    df["spread_sign"] = np.sign(df["current_spread"].to_numpy(dtype=float))

    signal = _class_to_signal(df["pred_class"].to_numpy(dtype=int), df["spread_sign"].to_numpy(dtype=float))

    if cfg.entry_z_threshold > 0 and "latest_spread_z" in df.columns:
        gate = np.abs(df["latest_spread_z"].to_numpy(dtype=float)) >= cfg.entry_z_threshold
        signal = np.where(gate, signal, 0.0)

    if cfg.apply_regime_mask and "mean_reverting_state" in df.columns:
        signal = np.where(df["mean_reverting_state"].to_numpy(dtype=bool), signal, 0.0)

    df["signal"] = signal

    # Position held during bar t is signal at t-1 (we trade at the close of t-1).
    held = np.concatenate([[0.0], signal[:-1]])
    df["held_position"] = held

    # Dollar-hedged spread pnl in log space:
    #   long $L of A, short $L*beta of B  ==>  bar pnl = $L * (r_a - beta * r_b).
    # The price ratio close_a/close_b is NOT needed here because both legs are
    # sized in dollars; it would only enter if we were sizing in units of B.
    df["pnl"] = df["held_position"] * cfg.leg_notional * (df["ret_a"].to_numpy() - beta * df["ret_b"].to_numpy())

    # Turnover and cost: $ traded per bar = L on A + L*|beta| on B per unit
    # position change. Charged at taker fee + slippage per side.
    delta = np.abs(signal - held)
    notional_traded = delta * cfg.leg_notional * (1.0 + abs(beta))
    cost_bps = cfg.taker_fee_bps + cfg.slippage_bps
    df["turnover"] = notional_traded
    df["pnl"] = df["pnl"] - notional_traded * (cost_bps / 1e4)

    out = df.reset_index()[["timestamp", "signal", "held_position", "pnl", "turnover"]]
    out.insert(1, "pair", pair_id)
    return out


def run_backtest(
    predictions: pd.DataFrame,
    close_panel: pd.DataFrame,
    pairs: pd.DataFrame,
    cfg: BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    """
    Parameters
    ----------
    predictions : DataFrame with columns
        sample_id, pair, timestamp, current_spread, y_class, pred_class
        Optionally: latest_spread_z, mean_reverting_state
    close_panel : DataFrame indexed by timestamp, columns are symbols (close prices).
    pairs : DataFrame with columns sym_a, sym_b, beta_a_on_b (and pair name).
    """
    if predictions.empty:
        raise ValueError("predictions is empty")
    if close_panel.empty:
        raise ValueError("close_panel is empty")

    if "pair" not in pairs.columns:
        pairs = pairs.assign(pair=pairs["sym_a"] + "_" + pairs["sym_b"])
    pair_lookup = pairs.set_index("pair")[["sym_a", "sym_b", "beta_a_on_b"]].to_dict("index")

    per_pair_frames: List[pd.DataFrame] = []
    trade_rows: List[Dict[str, object]] = []
    for pair_id, pair_df in predictions.groupby("pair", sort=False):
        meta = pair_lookup.get(pair_id)
        if meta is None:
            continue
        sym_a, sym_b = meta["sym_a"], meta["sym_b"]
        if sym_a not in close_panel.columns or sym_b not in close_panel.columns:
            continue
        beta = float(meta["beta_a_on_b"])
        rows = _pair_returns(pair_id, pair_df, close_panel[sym_a], close_panel[sym_b], beta, cfg)
        if rows.empty:
            continue
        per_pair_frames.append(rows)

        # Trade ledger: one row per position change
        rows["signal_prev"] = rows["signal"].shift(1).fillna(0.0)
        changes = rows[rows["signal"] != rows["signal_prev"]]
        for _, ch in changes.iterrows():
            trade_rows.append(
                {
                    "pair": pair_id,
                    "timestamp": ch["timestamp"],
                    "from_signal": float(ch["signal_prev"]),
                    "to_signal": float(ch["signal"]),
                    "turnover": float(ch["turnover"]),
                }
            )

    if not per_pair_frames:
        raise RuntimeError("No pairs produced any backtest rows. Check overlap between predictions, pairs, and close_panel.")

    long_form = pd.concat(per_pair_frames, ignore_index=True)
    per_pair_pnl = (
        long_form.pivot_table(index="timestamp", columns="pair", values="pnl", aggfunc="sum")
        .sort_index()
        .fillna(0.0)
    )
    per_pair_signal = (
        long_form.pivot_table(index="timestamp", columns="pair", values="signal", aggfunc="last")
        .sort_index()
        .fillna(0.0)
    )

    if cfg.max_active_pairs is not None:
        active = (per_pair_signal != 0).astype(int)
        ranked = active.cumsum(axis=1)
        keep = ranked.le(cfg.max_active_pairs).astype(int)
        per_pair_signal = per_pair_signal * keep
        per_pair_pnl = per_pair_pnl * keep

    portfolio = per_pair_pnl.sum(axis=1)
    trades = pd.DataFrame(trade_rows).sort_values("timestamp") if trade_rows else pd.DataFrame()
    metrics = _portfolio_metrics(portfolio, per_pair_pnl, per_pair_signal, trades, cfg)

    return BacktestResult(
        per_pair_returns=per_pair_pnl,
        portfolio_returns=portfolio,
        positions=per_pair_signal,
        trades=trades,
        metrics=metrics,
    )


def _portfolio_metrics(
    portfolio: pd.Series,
    per_pair_pnl: pd.DataFrame,
    positions: pd.DataFrame,
    trades: pd.DataFrame,
    cfg: BacktestConfig,
) -> Dict[str, float]:
    pnl = portfolio.to_numpy(dtype=float)
    if len(pnl) == 0:
        return {}
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak

    # Return per bar over deployed capital. Deployed = active pair count * leg_notional * 2 (both legs).
    active_pairs = (positions != 0).sum(axis=1).to_numpy(dtype=float)
    deployed = np.maximum(active_pairs, 1.0) * cfg.leg_notional * 2.0
    bar_ret = pnl / deployed

    mean_ret = float(np.mean(bar_ret))
    std_ret = float(np.std(bar_ret, ddof=1)) if len(bar_ret) > 1 else 0.0
    rf_per_bar = cfg.rf_annual / cfg.bars_per_year
    sharpe = float((mean_ret - rf_per_bar) / std_ret * np.sqrt(cfg.bars_per_year)) if std_ret > 0 else 0.0

    total_turnover = float(positions.diff().abs().sum().sum() * cfg.leg_notional * 2.0)

    return {
        "total_pnl_dollars": float(np.sum(pnl)),
        "mean_bar_return": mean_ret,
        "std_bar_return": std_ret,
        "sharpe_annualized": sharpe,
        "max_drawdown_dollars": float(drawdown.min()) if len(drawdown) else 0.0,
        "max_drawdown_pct_of_peak": float((drawdown / np.maximum(peak, 1.0)).min()) if len(drawdown) else 0.0,
        "bars": int(len(pnl)),
        "active_bars": int((active_pairs > 0).sum()),
        "trades": int(len(trades)),
        "turnover_dollars": total_turnover,
        "win_rate_bars": float((pnl > 0).sum() / max(1, (pnl != 0).sum())),
    }


def metrics_for_minitron(result: BacktestResult) -> Dict[str, float]:
    """
    Return metric names that match what minitron's `run_backtest` emits, so
    results tables stay apples-to-apples when we later wire this through a
    minitron adapter.
    """
    m = result.metrics
    return {
        "sharpe": m.get("sharpe_annualized", 0.0),
        "total_return": m.get("total_pnl_dollars", 0.0),
        "max_drawdown": m.get("max_drawdown_pct_of_peak", 0.0),
        "n_trades": m.get("trades", 0),
        "turnover": m.get("turnover_dollars", 0.0),
        "win_rate": m.get("win_rate_bars", 0.0),
        "bars": m.get("bars", 0),
    }
