"""
As-of-time universe construction for L2 / trades.

A symbol is in the universe at t0 iff its local store contains data on every
day in the qualifying window [t0 - min_history_days, t0 - 1d]. Liquidity
ranking is by mean daily notional volume (sum of trade `notional` per day,
averaged across the window) -- a faithful USDT-volume measure now that we
have trades, not just OHLCV.

Same look-ahead-safety contract as phase 1: only data with timestamp < t0
participates in any decision.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.l2_store import L2Config, list_local_symbols as list_l2_symbols
from src.trade_store import TradeConfig, list_local_symbols as list_trade_symbols, load_symbol_range as load_trades_range


def _days_present(symbol: str, root: Path) -> List[pd.Timestamp]:
    sym_dir = root / symbol.upper()
    if not sym_dir.exists():
        return []
    out = []
    for p in sym_dir.glob("*.parquet"):
        try:
            out.append(pd.Timestamp(p.stem, tz="UTC"))
        except Exception:
            continue
    return sorted(out)


def compute_universe_at_time(
    t0: pd.Timestamp,
    l2_cfg: L2Config = L2Config(),
    trade_cfg: TradeConfig = TradeConfig(),
    min_history_days: float = 0.0,
    require_both_feeds: bool = True,
) -> List[str]:
    """
    Symbols whose stored coverage spans [t0 - min_history_days, t0) for the
    required feeds (L2 + trades by default). Daily-file granularity.
    """
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    t0_floor = t0.floor("D")
    last_required = t0_floor - pd.Timedelta(days=1)
    first_required = t0_floor - pd.Timedelta(days=float(min_history_days))

    l2_symbols = set(list_l2_symbols(l2_cfg))
    trade_symbols = set(list_trade_symbols(trade_cfg))
    candidates = sorted(l2_symbols & trade_symbols) if require_both_feeds else sorted(l2_symbols | trade_symbols)

    universe: List[str] = []
    for sym in candidates:
        l2_days = set(_days_present(sym, Path(l2_cfg.data_dir)))
        trade_days = set(_days_present(sym, Path(trade_cfg.data_dir)))
        need = pd.date_range(first_required, last_required, freq="D", tz="UTC")
        has_l2 = all(d in l2_days for d in need)
        has_trades = all(d in trade_days for d in need)
        if require_both_feeds:
            if has_l2 and has_trades:
                universe.append(sym)
        else:
            if has_l2 or has_trades:
                universe.append(sym)
    return universe


def filter_top_n_by_liquidity(
    symbols: List[str],
    start: pd.Timestamp, end_exclusive: pd.Timestamp,
    top_n: int = 50,
    trade_cfg: TradeConfig = TradeConfig(),
) -> List[str]:
    """
    Rank by mean daily USDT (notional) volume across [start, end_exclusive)
    using the trades feed. No look-ahead because the caller passes a
    training-window endpoint.
    """
    if start.tzinfo is None: start = start.tz_localize("UTC")
    if end_exclusive.tzinfo is None: end_exclusive = end_exclusive.tz_localize("UTC")
    scores: List[Tuple[str, float]] = []
    for sym in symbols:
        trades = load_trades_range(sym, start, end_exclusive, cfg=trade_cfg)
        if trades.empty:
            continue
        daily_notional = trades.set_index("timestamp")["notional"].astype(float).resample("1D").sum()
        mu = float(daily_notional.mean())
        if not np.isfinite(mu):
            continue
        scores.append((sym, mu))
    scores.sort(key=lambda t: t[1], reverse=True)
    return [s for s, _ in scores[:top_n]]
