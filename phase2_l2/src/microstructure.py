"""
Microstructure features over L2 books + trade tapes.

Two families:
  - Book-derived: microprice, quoted spread (bps), order-book imbalance.
  - Trade-derived: aggressor signing, signed volume, size-bucket flow,
                   VPIN, sweep detection.

Bucket boundaries default to dollar-notional thresholds:
  retail < $1k <= mid < $10k <= large < $100k <= institutional
These are USDT-quoted defaults appropriate for Binance perps; override
when calling the constructors if needed for other venues.

The institutional/retail proxy is the pre-registered hypothesis target:
sign + sum trade notional within each bucket, then take the institutional
buy-imbalance per bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------- Book-side features --------------------

def microprice(book: pd.DataFrame, level: int = 1) -> pd.Series:
    """
    Size-weighted mid: (bid_sz * ask_px + ask_sz * bid_px) / (bid_sz + ask_sz).
    Reduces to mid when sizes are equal.
    """
    bp = book[f"bid_px_{level}"].astype(float)
    bs = book[f"bid_sz_{level}"].astype(float)
    ap = book[f"ask_px_{level}"].astype(float)
    as_ = book[f"ask_sz_{level}"].astype(float)
    denom = (bs + as_).replace(0.0, np.nan)
    return ((bs * ap) + (as_ * bp)) / denom


def quoted_spread_bps(book: pd.DataFrame, level: int = 1) -> pd.Series:
    bp = book[f"bid_px_{level}"].astype(float)
    ap = book[f"ask_px_{level}"].astype(float)
    mid = (bp + ap) / 2.0
    return (ap - bp) / mid.replace(0.0, np.nan) * 1e4


def order_book_imbalance(book: pd.DataFrame, depth_levels: int = 5) -> pd.Series:
    """OBI in [-1, 1]: > 0 means more size on bids."""
    bid_sz = sum(book[f"bid_sz_{i}"].astype(float) for i in range(1, depth_levels + 1))
    ask_sz = sum(book[f"ask_sz_{i}"].astype(float) for i in range(1, depth_levels + 1))
    total = (bid_sz + ask_sz).replace(0.0, np.nan)
    return (bid_sz - ask_sz) / total


# -------------------- Trade-side features --------------------

@dataclass(frozen=True)
class SizeBuckets:
    # Notional (USDT) cuts. Pair x_i values <= x_i are in bucket i.
    # Default boundaries: < $1k, < $10k, < $100k, >= $100k.
    retail_max: float = 1_000.0
    mid_max: float = 10_000.0
    large_max: float = 100_000.0


def assign_size_bucket(notional: pd.Series, buckets: SizeBuckets = SizeBuckets()) -> pd.Series:
    """Return categorical bucket label per trade."""
    n = notional.astype(float)
    out = pd.Series(np.empty(len(n), dtype=object), index=n.index)
    out[n < buckets.retail_max] = "retail"
    out[(n >= buckets.retail_max) & (n < buckets.mid_max)] = "mid"
    out[(n >= buckets.mid_max) & (n < buckets.large_max)] = "large"
    out[n >= buckets.large_max] = "institutional"
    return out.astype("category")


def sign_trades_tick_rule(trades: pd.DataFrame) -> pd.Series:
    """
    Tick rule fallback: trade is +1 if price > previous price, -1 if <, else
    carry previous sign. Used when `side` is missing.
    """
    p = trades["price"].astype(float).to_numpy()
    n = len(p)
    sign = np.zeros(n, dtype=np.int8)
    prev_sign = 0
    for i in range(n):
        if i == 0:
            sign[i] = 0
            continue
        if p[i] > p[i-1]:
            sign[i] = 1
        elif p[i] < p[i-1]:
            sign[i] = -1
        else:
            sign[i] = prev_sign
        prev_sign = sign[i]
    return pd.Series(sign, index=trades.index, name="trade_sign")


def sign_trades_quote_rule(
    trades: pd.DataFrame, book: pd.DataFrame,
) -> pd.Series:
    """
    Quote rule (Lee-Ready 1991): trade signed by comparison to midprice at the
    contemporaneous best quote. > mid -> buyer-initiated (+1), < mid -> seller-
    initiated (-1), == mid -> tick-rule fallback. `book` is a top-1 frame with
    columns [timestamp, bid_px_1, ask_px_1]; we reindex to trades via asof-merge.
    """
    book_top = book[["timestamp", "bid_px_1", "ask_px_1"]].copy()
    book_top["mid"] = (book_top["bid_px_1"].astype(float) + book_top["ask_px_1"].astype(float)) / 2.0
    merged = pd.merge_asof(
        trades[["timestamp", "price"]].sort_values("timestamp"),
        book_top.sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    p = merged["price"].astype(float).to_numpy()
    m = merged["mid"].astype(float).to_numpy()
    sign = np.where(p > m, 1, np.where(p < m, -1, 0)).astype(np.int8)
    # fallback to tick rule where mid is NaN or sign == 0
    tick = sign_trades_tick_rule(trades).to_numpy()
    sign = np.where((np.isnan(m)) | (sign == 0), tick, sign)
    return pd.Series(sign.astype(np.int8), index=trades.index, name="trade_sign")


def sign_trades_use_aggressor(trades: pd.DataFrame) -> pd.Series:
    """
    Prefer the exchange-reported aggressor side when present. 'buy' -> +1
    (buyer initiated, taker buying), 'sell' -> -1. Otherwise 0 (caller can
    fall back to tick/quote rule).
    """
    if "side" not in trades.columns:
        return pd.Series(np.zeros(len(trades), dtype=np.int8), index=trades.index, name="trade_sign")
    side = trades["side"].astype(str)
    sign = np.where(side == "buy", 1, np.where(side == "sell", -1, 0)).astype(np.int8)
    return pd.Series(sign, index=trades.index, name="trade_sign")


# -------------------- VPIN --------------------

def vpin(
    signed_notional: pd.Series, bucket_size_notional: float, window_buckets: int = 50,
) -> pd.Series:
    """
    Easley/Lopez de Prado/O'Hara VPIN. Bucket trades by cumulative notional
    (not time). For each bucket compute |buy_vol - sell_vol| / bucket_size.
    VPIN at time t = trailing-mean of bucket imbalances over `window_buckets`.

    Returns a series at trade-cadence (sparsely): the value at the end of each
    completed bucket. Downstream code can ffill onto bar timestamps.
    """
    sn = signed_notional.astype(float)
    notional_abs = sn.abs()
    cum = notional_abs.cumsum()
    bucket_id = (cum // bucket_size_notional).astype(int)

    # Per-bucket sums
    df = pd.DataFrame({"signed_notional": sn, "abs_notional": notional_abs, "bucket": bucket_id})
    grouped = df.groupby("bucket")
    buy_vol = grouped["signed_notional"].apply(lambda s: s[s > 0].sum())
    sell_vol = grouped["signed_notional"].apply(lambda s: -s[s < 0].sum())
    imb = (buy_vol - sell_vol).abs() / bucket_size_notional

    bucket_end_ts = grouped.apply(lambda g: g.index[-1])
    vpin_vals = imb.rolling(window_buckets, min_periods=1).mean()
    out = pd.Series(vpin_vals.values, index=bucket_end_ts.values, name="vpin")
    return out


# -------------------- Sweep detection --------------------

def detect_sweeps(
    trades: pd.DataFrame, book: pd.DataFrame, max_gap_ms: int = 50,
) -> pd.DataFrame:
    """
    A 'sweep' = sequence of consecutive same-side trades within `max_gap_ms` of
    each other that together exceed the best-level resting size (i.e. walked
    multiple levels).

    Returns a DataFrame with one row per sweep:
      [start_ts, end_ts, side, notional, n_trades, levels_walked_est]

    `levels_walked_est` is approximate: count of distinct trade prices in the
    sweep. We don't try to reconstruct the exact book walk here -- that's a
    follow-up.
    """
    if trades.empty:
        return pd.DataFrame(columns=["start_ts", "end_ts", "side", "notional", "n_trades", "levels_walked_est"])

    sign = sign_trades_use_aggressor(trades)
    fallback = sign_trades_tick_rule(trades)
    sign = sign.where(sign != 0, fallback)

    df = trades[["timestamp", "price", "amount", "notional"]].copy()
    df["sign"] = sign.values
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ts_ns"] = df["timestamp"].astype("int64")
    df["gap_ms"] = df["ts_ns"].diff().fillna(0).astype("int64") // 1_000_000
    df["new_seq"] = ((df["sign"] != df["sign"].shift(1)) | (df["gap_ms"] > max_gap_ms)).cumsum()

    grouped = df.groupby("new_seq")
    sweeps = pd.DataFrame({
        "start_ts": grouped["timestamp"].first(),
        "end_ts": grouped["timestamp"].last(),
        "side": grouped["sign"].first(),
        "notional": grouped["notional"].sum(),
        "n_trades": grouped.size(),
        "levels_walked_est": grouped["price"].nunique(),
    }).reset_index(drop=True)
    # Only call it a 'sweep' if multiple trades touched multiple price levels
    sweeps = sweeps[(sweeps["n_trades"] >= 2) & (sweeps["levels_walked_est"] >= 2)]
    return sweeps.reset_index(drop=True)


# -------------------- Combined trade-flow feature builder --------------------

@dataclass(frozen=True)
class FlowConfig:
    buckets: SizeBuckets = SizeBuckets()
    use_quote_rule_for_unsigned: bool = True
    vpin_bucket_notional: float = 1_000_000.0  # $1M buckets
    vpin_window: int = 50


def build_signed_trade_frame(
    trades: pd.DataFrame, book: Optional[pd.DataFrame] = None,
    cfg: FlowConfig = FlowConfig(),
) -> pd.DataFrame:
    """
    Returns trades enriched with sign, signed_notional, and size bucket.
    """
    out = trades.copy().reset_index(drop=True)
    sign_exch = sign_trades_use_aggressor(out)
    if (sign_exch == 0).any() and cfg.use_quote_rule_for_unsigned and book is not None and not book.empty:
        sign_fallback = sign_trades_quote_rule(out, book)
        sign = sign_exch.where(sign_exch != 0, sign_fallback)
    else:
        sign_fallback = sign_trades_tick_rule(out)
        sign = sign_exch.where(sign_exch != 0, sign_fallback)
    out["trade_sign"] = sign.astype(np.int8).values
    out["signed_notional"] = out["trade_sign"].astype(float) * out["notional"].astype(float)
    out["bucket"] = assign_size_bucket(out["notional"], buckets=cfg.buckets).astype(str).values
    return out
