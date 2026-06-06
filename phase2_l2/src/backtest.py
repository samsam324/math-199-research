"""
L2-aware portfolio backtester for crypto pair-spread signals.

Replaces phase 1's flat-bps cost model with a book-walking impact model.
Position changes consume liquidity off the contemporaneous L2 snapshot,
producing an effective fill price = volume-weighted price across the levels
walked. Slippage is the gap between effective fill and pre-trade midprice.

Architecture:
  - Per-bar PnL accumulates marked-to-market against midprice.
  - On a position change at bar t, fill price is computed by walking the bar-t
    book in the trade direction for the position's notional.
  - One-bar execution lag preserved (audit invariant from phase 1):
    held_position_t = signal_{t-1}.

Configurable: optional flat-bps fallback for synthetic tests that don't carry
L2 snapshots; queue-position model deferred.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    # Per-leg dollar notional.
    leg_notional: float = 10_000.0

    # Risk-free rate (annualized) for excess Sharpe. Zero for crypto by convention.
    rf_annual: float = 0.0
    bars_per_year: int = 24 * 365 * 60 * 60   # 1s bars, continuous market

    # Optional cap on simultaneous active pairs.
    max_active_pairs: Optional[int] = None

    # Bar-by-bar |z| gate (used when state machine is off).
    entry_z_threshold: float = 0.0

    # Entry/exit state machine.
    use_state_machine: bool = False
    entry_z: float = 2.0
    exit_z: float = 0.5

    # Cost model: walk the book using book_a / book_b snapshots merged into
    # predictions; if absent or empty, fall back to flat bps.
    walk_book: bool = True
    fallback_taker_fee_bps: float = 5.0
    fallback_slippage_bps: float = 2.0

    # Mandatory taker fee on both fill modes (exchange fee is real even in book-walk).
    taker_fee_bps: float = 5.0    # Binance perps default ~0.05% per side


@dataclass
class BacktestResult:
    per_pair_returns: pd.DataFrame
    portfolio_returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float] = field(default_factory=dict)


def _class_to_signal(pred_class: np.ndarray, spread_sign: np.ndarray) -> np.ndarray:
    pred_class = np.asarray(pred_class, dtype=int)
    spread_sign = np.asarray(spread_sign, dtype=float)
    signal = np.zeros_like(spread_sign, dtype=float)
    signal[pred_class == 0] = -spread_sign[pred_class == 0]
    signal[pred_class == 2] = spread_sign[pred_class == 2]
    signal[spread_sign == 0] = 0.0
    return signal


def _state_machine_signal(
    pred_class: np.ndarray, spread_z: np.ndarray, spread_sign: np.ndarray,
    entry_z: float, exit_z: float,
) -> np.ndarray:
    pred_class = np.asarray(pred_class, dtype=int)
    z = np.asarray(spread_z, dtype=float)
    s = np.asarray(spread_sign, dtype=float)
    n = len(pred_class)
    position = 0.0; entry_sign = 0.0
    out = np.zeros(n, dtype=float)
    for t in range(n):
        z_t = float(z[t]) if np.isfinite(z[t]) else 0.0
        s_t = float(s[t]) if np.isfinite(s[t]) else 0.0
        cls = int(pred_class[t])
        if position == 0.0:
            if cls == 0 and abs(z_t) >= entry_z and s_t != 0.0:
                position = -s_t; entry_sign = s_t
        else:
            flip = (s_t != 0.0 and np.sign(s_t) != np.sign(entry_sign))
            if abs(z_t) <= exit_z or cls == 2 or flip:
                position = 0.0; entry_sign = 0.0
        out[t] = position
    return out


# -------------------- Book-walking impact --------------------

def walk_book_fill(
    notional: float, side: str, book_row: pd.Series, levels: int = 10,
) -> tuple:
    """
    Walk the L2 book to fill `notional` (USDT) on `side` ('buy' or 'sell').

    Returns (vwap, filled_notional, levels_used). If the book doesn't have
    enough depth, fills what it can and returns the partial vwap.

    book_row expected columns:
      bid_px_1..K, bid_sz_1..K, ask_px_1..K, ask_sz_1..K   (sz in base asset)
    """
    if notional <= 0:
        return float("nan"), 0.0, 0
    if side == "buy":
        px_prefix, sz_prefix = "ask_px", "ask_sz"
    elif side == "sell":
        px_prefix, sz_prefix = "bid_px", "bid_sz"
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    remaining = float(notional)
    total_notional = 0.0
    total_base = 0.0
    levels_used = 0
    for i in range(1, levels + 1):
        px = float(book_row.get(f"{px_prefix}_{i}", np.nan))
        sz = float(book_row.get(f"{sz_prefix}_{i}", np.nan))
        if not np.isfinite(px) or not np.isfinite(sz) or sz <= 0:
            continue
        level_notional = px * sz
        if level_notional >= remaining:
            base_taken = remaining / px
            total_notional += remaining
            total_base += base_taken
            levels_used += 1
            remaining = 0.0
            break
        else:
            total_notional += level_notional
            total_base += sz
            remaining -= level_notional
            levels_used += 1
    if total_base <= 0:
        return float("nan"), 0.0, 0
    vwap = total_notional / total_base
    return vwap, total_notional, levels_used


def _pair_returns(
    pair_id: str, pair_df: pd.DataFrame, beta: float, cfg: BacktestConfig,
) -> pd.DataFrame:
    """
    Per-pair PnL stream. pair_df expected columns:
      timestamp, pred_class, current_spread, latest_spread_z (optional),
      mid_a, mid_b
      Optional book columns for walk-book mode (a / b suffixed):
        bid_px_1..K_a, bid_sz_1..K_a, ask_px_1..K_a, ask_sz_1..K_a (same for _b)
    """
    df = pair_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "pair", "signal", "pnl", "turnover", "slippage_bps"])

    df["mid_a"] = df["mid_a"].astype(float)
    df["mid_b"] = df["mid_b"].astype(float)
    df["ret_a"] = df["mid_a"].pct_change().fillna(0.0)
    df["ret_b"] = df["mid_b"].pct_change().fillna(0.0)
    df["spread_sign"] = np.sign(df["current_spread"].to_numpy(dtype=float))

    pred = df["pred_class"].to_numpy(dtype=int)
    spread_sign = df["spread_sign"].to_numpy(dtype=float)
    if cfg.use_state_machine:
        if "latest_spread_z" not in df.columns:
            raise ValueError("use_state_machine=True requires latest_spread_z column")
        signal = _state_machine_signal(pred, df["latest_spread_z"].to_numpy(dtype=float), spread_sign, cfg.entry_z, cfg.exit_z)
    else:
        signal = _class_to_signal(pred, spread_sign)
        if cfg.entry_z_threshold > 0 and "latest_spread_z" in df.columns:
            gate = np.abs(df["latest_spread_z"].to_numpy(dtype=float)) >= cfg.entry_z_threshold
            signal = np.where(gate, signal, 0.0)
    df["signal"] = signal

    # One-bar execution lag: held position at bar t is signal at t-1.
    held = np.concatenate([[0.0], signal[:-1]])
    df["held_position"] = held

    # MTM PnL per bar on the held position (dollar-hedged $L of A, $L*beta of B)
    df["pnl_mtm"] = df["held_position"] * cfg.leg_notional * (df["ret_a"].to_numpy() - beta * df["ret_b"].to_numpy())

    # Cost on every position change
    delta = signal - held
    cost = np.zeros(len(df), dtype=float)
    slippage_bps = np.full(len(df), np.nan, dtype=float)

    fee_rate = cfg.taker_fee_bps / 1e4
    has_book = cfg.walk_book and all(
        f"{prefix}_{side}_1_{leg}" in df.columns
        for prefix in ("bid_px", "ask_px") for side in ()  # placeholder; check below
        for leg in ("a", "b")
    )
    # Simpler check: walk_book requested AND at least best bid/ask present per leg
    has_book = (
        cfg.walk_book
        and "ask_px_1_a" in df.columns and "bid_px_1_a" in df.columns
        and "ask_px_1_b" in df.columns and "bid_px_1_b" in df.columns
    )

    for t in range(len(df)):
        d = float(delta[t])
        if d == 0.0:
            continue

        # Convert position delta to (side_a, side_b) and notionals per leg.
        # Long-spread (+d): buy A, sell B.   Short-spread (-d): sell A, buy B.
        if d > 0:
            side_a, side_b = "buy", "sell"
        else:
            side_a, side_b = "sell", "buy"
        notional_a = abs(d) * cfg.leg_notional
        notional_b = abs(d) * cfg.leg_notional * abs(beta)

        if has_book:
            row = df.iloc[t]
            book_a = {f"{p}_{i}": row.get(f"{p}_{i}_a", np.nan) for p in ("bid_px", "bid_sz", "ask_px", "ask_sz") for i in range(1, 11)}
            book_b = {f"{p}_{i}": row.get(f"{p}_{i}_b", np.nan) for p in ("bid_px", "bid_sz", "ask_px", "ask_sz") for i in range(1, 11)}
            vwap_a, _, _ = walk_book_fill(notional_a, side_a, pd.Series(book_a))
            vwap_b, _, _ = walk_book_fill(notional_b, side_b, pd.Series(book_b))
            mid_a = float(row["mid_a"]); mid_b = float(row["mid_b"])
            sign_a = 1.0 if side_a == "buy" else -1.0
            sign_b = 1.0 if side_b == "buy" else -1.0
            slip_a = (vwap_a - mid_a) * sign_a if np.isfinite(vwap_a) else mid_a * cfg.fallback_slippage_bps / 1e4
            slip_b = (vwap_b - mid_b) * sign_b if np.isfinite(vwap_b) else mid_b * cfg.fallback_slippage_bps / 1e4
            # Slippage cost in dollars: per-leg slippage per unit base * base size
            base_a = notional_a / mid_a if mid_a > 0 else 0.0
            base_b = notional_b / mid_b if mid_b > 0 else 0.0
            slip_cost = slip_a * base_a + slip_b * base_b
            slippage_bps[t] = ((slip_a / mid_a if mid_a > 0 else 0.0) + (slip_b / mid_b if mid_b > 0 else 0.0)) * 1e4 / 2
        else:
            # Flat fallback
            slip_cost = (notional_a + notional_b) * (cfg.fallback_slippage_bps / 1e4)
            slippage_bps[t] = cfg.fallback_slippage_bps

        fee_cost = (notional_a + notional_b) * fee_rate
        cost[t] = slip_cost + fee_cost

    df["pnl"] = df["pnl_mtm"] - cost
    df["turnover"] = np.abs(delta) * cfg.leg_notional * (1.0 + abs(beta))
    df["slippage_bps"] = slippage_bps

    out = df[["timestamp", "signal", "held_position", "pnl", "turnover", "slippage_bps"]].copy()
    out.insert(1, "pair", pair_id)
    return out


def run_backtest(
    predictions: pd.DataFrame,
    pairs: pd.DataFrame,
    cfg: BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    """
    Required columns in `predictions`:
      sample_id, pair, timestamp, pred_class, current_spread, mid_a, mid_b
    Optional:
      latest_spread_z (required if state machine on)
      Book columns: {bid|ask}_{px|sz}_{1..K}_{a|b}  (needed if cfg.walk_book)
    `pairs`: sym_a, sym_b, beta_a_on_b (and optionally 'pair' = sym_a_sym_b)
    """
    if predictions.empty:
        raise ValueError("predictions is empty")
    if "pair" not in pairs.columns:
        pairs = pairs.assign(pair=pairs["sym_a"] + "_" + pairs["sym_b"])
    pair_lookup = pairs.set_index("pair")[["sym_a", "sym_b", "beta_a_on_b"]].to_dict("index")

    per_pair_frames: List[pd.DataFrame] = []
    trade_rows: List[Dict[str, object]] = []
    for pair_id, pair_df in predictions.groupby("pair", sort=False):
        meta = pair_lookup.get(pair_id)
        if meta is None:
            continue
        beta = float(meta["beta_a_on_b"])
        rows = _pair_returns(pair_id, pair_df, beta, cfg)
        if rows.empty:
            continue
        per_pair_frames.append(rows)
        rows["signal_prev"] = rows["signal"].shift(1).fillna(0.0)
        changes = rows[rows["signal"] != rows["signal_prev"]]
        for _, ch in changes.iterrows():
            trade_rows.append({
                "pair": pair_id, "timestamp": ch["timestamp"],
                "from_signal": float(ch["signal_prev"]), "to_signal": float(ch["signal"]),
                "turnover": float(ch["turnover"]), "slippage_bps": float(ch["slippage_bps"]),
            })

    if not per_pair_frames:
        raise RuntimeError("No pairs produced backtest rows.")

    long_form = pd.concat(per_pair_frames, ignore_index=True)
    per_pair_pnl = long_form.pivot_table(index="timestamp", columns="pair", values="pnl", aggfunc="sum").sort_index().fillna(0.0)
    per_pair_signal = long_form.pivot_table(index="timestamp", columns="pair", values="signal", aggfunc="last").sort_index().fillna(0.0)

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
        per_pair_returns=per_pair_pnl, portfolio_returns=portfolio,
        positions=per_pair_signal, trades=trades, metrics=metrics,
    )


def _portfolio_metrics(
    portfolio: pd.Series, per_pair_pnl: pd.DataFrame,
    positions: pd.DataFrame, trades: pd.DataFrame, cfg: BacktestConfig,
) -> Dict[str, float]:
    pnl = portfolio.to_numpy(dtype=float)
    if len(pnl) == 0:
        return {}
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak

    # Fixed capital base (audit invariant from phase 1).
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
