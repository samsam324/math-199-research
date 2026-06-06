"""
Bar aggregation from L2 + trades to fixed-cadence bars (default 1 second).

A bar at timestamp t represents the interval (t - bar_size, t]. Values inside
the bar are aggregated to a single row indexed at t. The book columns
(microprice, OBI, quoted spread) take the LAST value within the bar -- the
state of the book at bar close. Trade columns sum within the bar.

The bar table is the unit of analysis for the feature builder, the Kalman
hedge, and the model dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.microstructure import (
    FlowConfig, build_signed_trade_frame,
    microprice, order_book_imbalance, quoted_spread_bps,
)


BAR_COLUMNS = [
    "microprice", "midprice", "quoted_spread_bps", "obi_5",
    # Trade flow (totals over the bar)
    "trade_count", "trade_notional",
    "buy_notional", "sell_notional",
    "signed_notional",
    # Size-bucketed signed flow
    "signed_notional_retail", "signed_notional_mid",
    "signed_notional_large", "signed_notional_institutional",
    # Institutional / retail buy ratio (the pre-reg target metric)
    "inst_buy_notional", "inst_sell_notional",
    "retail_buy_notional", "retail_sell_notional",
]


@dataclass(frozen=True)
class BarConfig:
    bar_size: str = "1s"            # pandas-style freq alias
    obi_depth_levels: int = 5
    flow_cfg: FlowConfig = FlowConfig()


def build_bars(
    book: pd.DataFrame, trades: pd.DataFrame, cfg: BarConfig = BarConfig(),
) -> pd.DataFrame:
    """
    Aggregate book + trades to fixed-cadence bars.

    Returns a frame indexed by bar end timestamp (UTC) with BAR_COLUMNS.
    No look-ahead: every column at time t uses only data with timestamp <= t.
    """
    if book.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)

    book = book.copy().sort_values("timestamp").reset_index(drop=True)
    book["microprice"] = microprice(book, level=1)
    book["midprice"] = (book["bid_px_1"].astype(float) + book["ask_px_1"].astype(float)) / 2.0
    book["quoted_spread_bps"] = quoted_spread_bps(book, level=1)
    book["obi_5"] = order_book_imbalance(book, depth_levels=cfg.obi_depth_levels)

    book_idx = book.set_index("timestamp")
    book_bars = book_idx[["microprice", "midprice", "quoted_spread_bps", "obi_5"]].resample(
        cfg.bar_size, label="right", closed="right"
    ).last()

    if trades.empty:
        # Book-only bars: fill flow columns with zeros
        for c in BAR_COLUMNS:
            if c not in book_bars.columns:
                book_bars[c] = 0.0
        return book_bars[BAR_COLUMNS].dropna(subset=["microprice"]).copy()

    enriched = build_signed_trade_frame(trades, book=book, cfg=cfg.flow_cfg)
    enriched = enriched.set_index("timestamp").sort_index()
    buy_mask = enriched["trade_sign"] > 0
    sell_mask = enriched["trade_sign"] < 0

    bar_size = cfg.bar_size

    def _sum_by(series, mask, label):
        return series.where(mask, 0.0).resample(bar_size, label="right", closed="right").sum().rename(label)

    trade_count = enriched["trade_sign"].resample(bar_size, label="right", closed="right").count().rename("trade_count")
    trade_notional = enriched["notional"].resample(bar_size, label="right", closed="right").sum().rename("trade_notional")
    buy_notional = _sum_by(enriched["notional"], buy_mask, "buy_notional")
    sell_notional = _sum_by(enriched["notional"], sell_mask, "sell_notional")
    signed_notional = enriched["signed_notional"].resample(bar_size, label="right", closed="right").sum().rename("signed_notional")

    # By bucket
    bucket_frames: List[pd.Series] = []
    for bucket in ("retail", "mid", "large", "institutional"):
        in_bucket = enriched["bucket"].astype(str) == bucket
        s = _sum_by(enriched["signed_notional"], in_bucket, f"signed_notional_{bucket}")
        bucket_frames.append(s)

    # Institutional / retail buy and sell legs (for the pre-reg metric)
    inst_buy = _sum_by(
        enriched["notional"],
        (enriched["bucket"].astype(str) == "institutional") & buy_mask,
        "inst_buy_notional",
    )
    inst_sell = _sum_by(
        enriched["notional"],
        (enriched["bucket"].astype(str) == "institutional") & sell_mask,
        "inst_sell_notional",
    )
    retail_buy = _sum_by(
        enriched["notional"],
        (enriched["bucket"].astype(str) == "retail") & buy_mask,
        "retail_buy_notional",
    )
    retail_sell = _sum_by(
        enriched["notional"],
        (enriched["bucket"].astype(str) == "retail") & sell_mask,
        "retail_sell_notional",
    )

    flow = pd.concat(
        [trade_count, trade_notional, buy_notional, sell_notional, signed_notional,
         *bucket_frames, inst_buy, inst_sell, retail_buy, retail_sell],
        axis=1,
    ).fillna(0.0)

    bars = book_bars.join(flow, how="outer").sort_index()
    # Book columns should ffill (state is persistent); flow columns stay 0 when empty
    book_cols = ["microprice", "midprice", "quoted_spread_bps", "obi_5"]
    bars[book_cols] = bars[book_cols].ffill()
    flow_cols = [c for c in BAR_COLUMNS if c not in book_cols]
    bars[flow_cols] = bars[flow_cols].fillna(0.0)
    bars = bars.dropna(subset=["microprice"])
    return bars[BAR_COLUMNS].copy()


def build_pair_bars(
    book_a: pd.DataFrame, trades_a: pd.DataFrame,
    book_b: pd.DataFrame, trades_b: pd.DataFrame,
    cfg: BarConfig = BarConfig(),
) -> pd.DataFrame:
    """
    Build per-symbol bars and join on the shared time index. Result has each
    feature suffixed with '_a' / '_b'.
    """
    bars_a = build_bars(book_a, trades_a, cfg).add_suffix("_a")
    bars_b = build_bars(book_b, trades_b, cfg).add_suffix("_b")
    return bars_a.join(bars_b, how="inner").sort_index()
