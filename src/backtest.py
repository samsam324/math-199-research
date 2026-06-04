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
    # Bar-by-bar minimum |z| gate (used when state machine is off).
    entry_z_threshold: float = 0.0
    # Optional regime mask (per-row bool); when set, only allow entries where True.
    apply_regime_mask: bool = False
    # Entry / exit state machine. When enabled, a position is opened only on
    # |spread_z| >= entry_z AND a reversion-class prediction; it is held until
    # |spread_z| <= exit_z OR a divergence-class prediction OR a flat prediction
    # that crosses the spread sign. This collapses bar-by-bar classifier
    # flickers into discrete trades.
    use_state_machine: bool = False
    entry_z: float = 2.0
    exit_z: float = 0.5
    # When True, slippage comes from the L2 order book (src/l2_costs.L2CostModel)
    # instead of the flat slippage_bps. Legs/timestamps with no L2 coverage fall
    # back to slippage_bps. taker_fee_bps still applies on top either way.
    use_l2_costs: bool = False
    l2_levels: int = 10
    l2_data_dir: str = "data/l2"


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


def _state_machine_signal(
    pred_class: np.ndarray,
    spread_z: np.ndarray,
    spread_sign: np.ndarray,
    entry_z: float,
    exit_z: float,
) -> np.ndarray:
    """
    Entry/exit state machine.

    Open when:
      flat AND |spread_z| >= entry_z AND pred_class == 0 (revert)
      -> position = -sign(spread_z), i.e. short the spread
    Close (go flat) when:
      |spread_z| <= exit_z, OR
      pred_class == 2 (diverge), OR
      spread sign flips relative to entry sign (the spread crossed zero).

    Position class 1 (persist) is a hold instruction; we keep the current
    position. Returns a per-bar signal in {-1, 0, +1}.
    """
    pred_class = np.asarray(pred_class, dtype=int)
    z = np.asarray(spread_z, dtype=float)
    s = np.asarray(spread_sign, dtype=float)
    n = len(pred_class)

    position = 0.0
    entry_sign = 0.0
    out = np.zeros(n, dtype=float)
    for t in range(n):
        z_t = float(z[t]) if np.isfinite(z[t]) else 0.0
        s_t = float(s[t]) if np.isfinite(s[t]) else 0.0
        cls = int(pred_class[t])

        if position == 0.0:
            # Look for entry
            if cls == 0 and abs(z_t) >= entry_z and s_t != 0.0:
                position = -s_t  # short the spread relative to its current sign
                entry_sign = s_t
        else:
            # Look for exit
            flip = (s_t != 0.0 and np.sign(s_t) != np.sign(entry_sign))
            if abs(z_t) <= exit_z or cls == 2 or flip:
                position = 0.0
                entry_sign = 0.0

        out[t] = position
    return out


def _l2_leg_cost_bps(
    cost_model,
    symbol: str,
    timestamps: np.ndarray,
    trade_mask: np.ndarray,
    leg_notional: np.ndarray,
    held_dir: np.ndarray,
    fallback_bps: float,
) -> np.ndarray:
    """
    Per-bar slippage in bps for one leg. At each bar where the leg trades, query
    the L2 book for the cost of executing `leg_notional[t]` dollars; the trade
    side follows the direction the leg moves toward. Bars without a trade cost
    nothing. Falls back to `fallback_bps` whenever L2 has no data at that point.
    """
    out = np.zeros(len(timestamps), dtype=float)
    if cost_model is None or not cost_model.has_symbol(symbol):
        out[trade_mask] = fallback_bps
        return out
    for t in np.nonzero(trade_mask)[0]:
        side = "buy" if held_dir[t] >= 0 else "sell"
        bps = cost_model.slippage_bps(symbol, pd.Timestamp(timestamps[t]), float(leg_notional[t]), side)
        out[t] = fallback_bps if bps is None else bps
    return out


def _pair_returns(
    pair_id: str,
    pair_df: pd.DataFrame,
    close_a: pd.Series,
    close_b: pd.Series,
    beta: float,
    cfg: BacktestConfig,
    sym_a: Optional[str] = None,
    sym_b: Optional[str] = None,
    cost_model=None,
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

    pred = df["pred_class"].to_numpy(dtype=int)
    spread_sign = df["spread_sign"].to_numpy(dtype=float)

    if cfg.use_state_machine:
        if "latest_spread_z" not in df.columns:
            raise ValueError(
                "use_state_machine=True requires 'latest_spread_z' in the predictions frame; "
                "join it from samples.parquet before calling run_backtest."
            )
        spread_z = df["latest_spread_z"].to_numpy(dtype=float)
        signal = _state_machine_signal(pred, spread_z, spread_sign, cfg.entry_z, cfg.exit_z)
    else:
        signal = _class_to_signal(pred, spread_sign)
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
    # position change. Taker fee is flat per side; slippage is either the flat
    # cfg.slippage_bps or, when a cost_model is supplied, the L2 book-walk cost
    # of filling each leg's notional (with per-leg flat fallback if L2 is missing).
    signed_delta = signal - held
    delta = np.abs(signed_delta)
    trade_mask = delta > 0
    ts_arr = df.index.to_numpy()

    leg_a_notional = delta * cfg.leg_notional
    leg_b_notional = delta * cfg.leg_notional * abs(beta)
    # signal=+1 is long A / short B, so the B leg trades opposite to A.
    a_dir = np.sign(signed_delta)
    b_dir = -a_dir

    slip_a = _l2_leg_cost_bps(cost_model, sym_a, ts_arr, trade_mask, leg_a_notional, a_dir, cfg.slippage_bps)
    slip_b = _l2_leg_cost_bps(cost_model, sym_b, ts_arr, trade_mask, leg_b_notional, b_dir, cfg.slippage_bps)

    cost_a = leg_a_notional * ((cfg.taker_fee_bps + slip_a) / 1e4)
    cost_b = leg_b_notional * ((cfg.taker_fee_bps + slip_b) / 1e4)

    df["turnover"] = leg_a_notional + leg_b_notional
    df["pnl"] = df["pnl"] - (cost_a + cost_b)

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

    cost_model = None
    if cfg.use_l2_costs:
        from src.l2_costs import L2CostModel
        from src.l2_store import L2Config

        ts = pd.to_datetime(predictions["timestamp"], utc=True)
        cost_model = L2CostModel(
            start=ts.min(),
            end=ts.max() + pd.Timedelta(hours=1),
            cfg=L2Config(levels=cfg.l2_levels, data_dir=cfg.l2_data_dir),
        )

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
        rows = _pair_returns(
            pair_id, pair_df, close_panel[sym_a], close_panel[sym_b], beta, cfg,
            sym_a=sym_a, sym_b=sym_b, cost_model=cost_model,
        )
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

    # Return per bar on a FIXED capital base = N_pairs * leg_notional * 2.
    # Using a variable deployed denominator (active_pairs at time t) breaks the
    # Sharpe estimator because the resulting bar_ret series is no longer a
    # well-defined return on a single capital base.
    n_pairs = positions.shape[1]
    capital_base = max(1.0, n_pairs * cfg.leg_notional * 2.0)
    bar_ret = pnl / capital_base
    active_pairs = (positions != 0).sum(axis=1).to_numpy(dtype=float)

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
