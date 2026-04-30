# src/universe.py
from __future__ import annotations
from typing import List, Tuple

import numpy as np
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


def filter_top_n_by_liquidity(
    cfg: StoreConfig,
    symbols: List[str],
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    top_n: int = 50,
    min_coverage: float = 0.80,
) -> List[str]:
    """
    Rank symbols by mean USDT volume (close * volume) over [start, end_exclusive),
    return the top_n.

    Liquidity score is mean(close * volume) per bar. Symbols whose in-window bar
    count is below min_coverage of the expected calendar are dropped before
    ranking — sparse symbols would otherwise rank on noisy means.

    Uses only data inside the window, so callers passing the training window
    avoid look-ahead.
    """
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end_exclusive.tzinfo is None:
        end_exclusive = end_exclusive.tz_localize("UTC")
    else:
        end_exclusive = end_exclusive.tz_convert("UTC")

    start = start.floor(cfg.interval)
    end_exclusive = end_exclusive.floor(cfg.interval)
    if end_exclusive <= start:
        return []

    step = pd.tseries.frequencies.to_offset(cfg.interval)
    expected_bars = int((end_exclusive - start) / pd.Timedelta(step))
    if expected_bars <= 0:
        return []

    scores: List[Tuple[str, float]] = []
    for sym in symbols:
        df = load_symbol(cfg, sym, columns=["close", "volume"])
        if df is None or df.empty:
            continue
        w = df.loc[(df.index >= start) & (df.index < end_exclusive)]
        if len(w) < min_coverage * expected_bars:
            continue
        usdt_vol = float((w["close"] * w["volume"]).mean())
        if not np.isfinite(usdt_vol):
            continue
        scores.append((sym, usdt_vol))

    scores.sort(key=lambda t: t[1], reverse=True)
    return [s for s, _ in scores[:top_n]]