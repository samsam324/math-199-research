# src/universe.py
from __future__ import annotations
from typing import List

import pandas as pd

from src.data_store import StoreConfig, load_symbol

def compute_universe_at_time(
    cfg: StoreConfig,
    local_symbols: List[str],
    t0: pd.Timestamp,
) -> List[str]:
    """
    Universe(t0) = symbols that appear to be tradable as of t0 based on local candles only.

    Definition used:
      - Let t0_floor be t0 floored to the bar interval.
      - A symbol is "available at t0" if it has at least one candle with timestamp <= t0_floor - 1 interval,
        i.e., it has traded recently up to the boundary before t0.

    No liquidity filters, no coverage filters, no ranking.
    """
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    else:
        t0 = t0.tz_convert("UTC")

    t0_floor = t0.floor(cfg.interval)
    step = pd.tseries.frequencies.to_offset(cfg.interval)
    last_required = t0_floor - step

    universe: List[str] = []
    for sym in local_symbols:
        df = load_symbol(cfg, sym, columns=["close"])
        if df is None or df.empty:
            continue
        last_ts = df.index.max()
        if last_ts >= last_required:
            universe.append(sym)

    return sorted(universe)