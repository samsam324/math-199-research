"""
L2-derived execution-cost model for the backtester.

Replaces the flat `slippage_bps` assumption in `src/backtest.py` with a cost
that comes from the actual order book: half the quoted spread plus the market
impact of walking the book to fill a given dollar notional. This is the phase-2
upgrade flagged in docs/NOTES.md ("Flat slippage, no market impact. The L2
phase will resolve this.").

A trade to BUY `notional_usd` dollars consumes ask levels from best outward;
the volume-weighted fill price minus the mid, expressed in bps, is the cost. A
SELL consumes bid levels symmetrically. Spread cost falls out naturally because
the best ask sits above the mid (and best bid below it).

If a symbol has no L2 coverage at the requested time, `slippage_bps` returns
None so the caller can fall back to the flat assumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.l2_store import L2Config, load_symbol_range


@dataclass
class _SymbolBook:
    ts_ns: np.ndarray            # int64 sorted snapshot timestamps (UTC ns)
    bid_px: np.ndarray           # shape (n_rows, levels)
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray


class L2CostModel:
    """
    Lazily loads per-symbol L2 books and answers execution-cost queries.

    Parameters
    ----------
    start, end : the time span the backtest will query (used to bound the load).
    levels     : top-K levels to walk (defaults to the L2Config level count).
    """

    def __init__(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        cfg: L2Config = L2Config(),
    ) -> None:
        self.cfg = cfg
        self.start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
        self.end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
        self._cache: Dict[str, Optional[_SymbolBook]] = {}

    def _book(self, symbol: str) -> Optional[_SymbolBook]:
        symbol = symbol.upper()
        if symbol in self._cache:
            return self._cache[symbol]
        df = load_symbol_range(symbol, self.start, self.end, self.cfg)
        if df.empty:
            self._cache[symbol] = None
            return None
        K = self.cfg.levels
        book = _SymbolBook(
            ts_ns=df["timestamp"].astype("int64").to_numpy(),
            bid_px=df[[f"bid_px_{i}" for i in range(1, K + 1)]].to_numpy(dtype=float),
            bid_sz=df[[f"bid_sz_{i}" for i in range(1, K + 1)]].to_numpy(dtype=float),
            ask_px=df[[f"ask_px_{i}" for i in range(1, K + 1)]].to_numpy(dtype=float),
            ask_sz=df[[f"ask_sz_{i}" for i in range(1, K + 1)]].to_numpy(dtype=float),
        )
        self._cache[symbol] = book
        return book

    def has_symbol(self, symbol: str) -> bool:
        return self._book(symbol) is not None

    def slippage_bps(
        self,
        symbol: str,
        ts: pd.Timestamp,
        notional_usd: float,
        side: str,
    ) -> Optional[float]:
        """
        Cost in bps of mid for filling `notional_usd` on `side` ('buy'/'sell')
        using the book as of the latest snapshot at-or-before `ts`. None if the
        symbol has no L2 data. Always >= 0.
        """
        if notional_usd <= 0:
            return 0.0
        book = self._book(symbol)
        if book is None:
            return None

        ts = pd.Timestamp(ts)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        i = int(np.searchsorted(book.ts_ns, ts.value, side="right")) - 1
        if i < 0:
            return None  # trade precedes first snapshot

        if side == "buy":
            px, sz = book.ask_px[i], book.ask_sz[i]
            best, mid = px[0], 0.5 * (book.bid_px[i][0] + px[0])
            signed = 1.0
        elif side == "sell":
            px, sz = book.bid_px[i], book.bid_sz[i]
            best, mid = px[0], 0.5 * (px[0] + book.ask_px[i][0])
            signed = -1.0
        else:
            raise ValueError("side must be 'buy' or 'sell'")
        if not np.isfinite(mid) or mid <= 0:
            return None

        remaining = notional_usd
        cost_vs_mid = 0.0  # dollars paid above mid (buy) / below mid (sell)
        last_px = best
        for lvl in range(len(px)):
            p, s = px[lvl], sz[lvl]
            if not np.isfinite(p) or p <= 0 or not np.isfinite(s) or s <= 0:
                continue
            last_px = p
            level_usd = p * s
            take = min(remaining, level_usd)
            units = take / p
            cost_vs_mid += units * (p - mid) * signed
            remaining -= take
            if remaining <= 0:
                break
        if remaining > 0:
            # Book too thin to fill; charge the residual at the worst seen level.
            units = remaining / last_px
            cost_vs_mid += units * (last_px - mid) * signed

        return float(cost_vs_mid / notional_usd * 1e4)
