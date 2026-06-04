"""
Volume-as-information microstructure features.

Phase-1 used coarse hourly OHLCV volume (close*volume). This module derives the
finer order-flow / information signals the L2 + tick-trade data unlocks, then
rolls them up to the hourly calendar the pairs pipeline runs on, so they slot
into `src/features.py` as additional per-symbol and per-pair columns.

Inputs are the 1s bars produced by the downloader:
    data/trades/{SYMBOL}/{date}.parquet   (trades_store: volume + signed flow)
    data/l2/{SYMBOL}/{date}.parquet       (l2_store: top-K book)

Signals
-------
order_flow_imbalance   net signed taker volume / total volume, in [-1, 1].
                       The directional-information proxy: persistent one-sided
                       flow is the classic footprint of informed trading.
trade_intensity        trades per hour (count), a liquidity/attention proxy.
volume_quote           total USDT traded in the hour.
vpin                   Volume-Synchronized Probability of Informed Trading
                       (Easley, Lopez de Prado, O'Hara 2012): mean of
                       |buy-sell|/bucket_volume over equal-volume buckets. High
                       VPIN = toxic, one-sided flow.
kyle_lambda            Kyle's (1985) price impact: OLS slope of 1s mid-return on
                       signed volume within the hour. Higher = less liquid /
                       more informative order flow.
quoted_spread_bps      mean top-of-book spread (from L2), a direct liquidity cost.

All of these are computed per symbol per hour; `pair_information_features`
combines the two legs into the per-pair columns the models consume.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.l2_store import L2Config, load_symbol_range as load_book
from src.trades_store import TradesConfig, load_symbol_range as load_trades


def _vpin(buy: np.ndarray, sell: np.ndarray, n_buckets: int = 50) -> float:
    """VPIN over equal-*volume* buckets within the window. NaN if too little volume."""
    total = buy + sell
    vol = total.sum()
    if vol <= 0 or len(total) == 0:
        return np.nan
    bucket = vol / n_buckets
    if bucket <= 0:
        return np.nan
    cum = np.cumsum(total)
    # assign each 1s bar to a volume bucket; aggregate signed imbalance per bucket
    idx = np.minimum((cum / bucket).astype(int), n_buckets - 1)
    imb = np.zeros(n_buckets)
    bvol = np.zeros(n_buckets)
    np.add.at(imb, idx, np.abs(buy - sell))
    np.add.at(bvol, idx, total)
    mask = bvol > 0
    if not mask.any():
        return np.nan
    return float((imb[mask] / bvol[mask]).mean())


def _kyle_lambda(mid: np.ndarray, signed_vol: np.ndarray) -> float:
    """OLS slope of mid return on signed volume. NaN if degenerate."""
    if len(mid) < 10:
        return np.nan
    ret = np.diff(mid) / mid[:-1]
    x = signed_vol[1:]
    good = np.isfinite(ret) & np.isfinite(x)
    if good.sum() < 10 or np.std(x[good]) == 0:
        return np.nan
    x = x[good]; y = ret[good]
    xc = x - x.mean()
    denom = float((xc ** 2).sum())
    if denom == 0:
        return np.nan
    return float((xc * (y - y.mean())).sum() / denom)


def hourly_information_features(
    symbol: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    levels: int = 10,
) -> pd.DataFrame:
    """
    Hourly volume/order-flow/liquidity features for one symbol. Index is the
    hourly UTC bar (right-labelled). Empty frame if no trade data on disk.
    """
    trades = load_trades(symbol, start, end_exclusive, TradesConfig())
    if trades.empty:
        return pd.DataFrame()
    trades = trades.set_index("timestamp").sort_index()
    hour = trades.resample("1h", label="right", closed="right")

    rows = {
        "volume_quote": hour["volume_quote"].sum(),
        "trade_intensity": hour["trade_count"].sum(),
        "signed_volume": hour["signed_volume_base"].sum(),
        "total_volume_base": hour["volume_base"].sum(),
    }
    feat = pd.DataFrame(rows)
    feat["order_flow_imbalance"] = (
        feat["signed_volume"] / feat["total_volume_base"].replace(0.0, np.nan)
    )
    # VPIN per hour from the 1s buy/sell split.
    feat["vpin"] = hour.apply(
        lambda g: _vpin(g["buy_volume_base"].to_numpy(), g["sell_volume_base"].to_numpy())
    )

    # Kyle's lambda needs the 1s mid from the book; join if L2 is present.
    book = load_book(symbol, start, end_exclusive, L2Config(levels=levels))
    if not book.empty:
        b = book.set_index("timestamp").sort_index()
        b = b[~b.index.duplicated(keep="last")]
        mid = (b["bid_px_1"] + b["ask_px_1"]) / 2.0
        spread_bps = (b["ask_px_1"] - b["bid_px_1"]) / mid.replace(0.0, np.nan) * 1e4

        sv = trades["signed_volume_base"]
        sv = sv[~sv.index.duplicated(keep="last")]
        m = pd.DataFrame({"mid": mid}).join(sv.rename("sv"), how="outer").sort_index()
        m["mid"] = m["mid"].ffill()
        m["sv"] = m["sv"].fillna(0.0)
        m = m.dropna(subset=["mid"])

        hourkey = m.index.floor("1h") + pd.Timedelta(hours=1)  # right-label to match feat
        kl = m.groupby(hourkey).apply(lambda g: _kyle_lambda(g["mid"].to_numpy(), g["sv"].to_numpy()))
        feat["kyle_lambda"] = kl
        feat["quoted_spread_bps"] = spread_bps.resample("1h", label="right", closed="right").mean()
    else:
        feat["kyle_lambda"] = np.nan
        feat["quoted_spread_bps"] = np.nan

    return feat.dropna(how="all")


def pair_information_features(
    sym_a: str,
    sym_b: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    levels: int = 10,
) -> pd.DataFrame:
    """
    Per-pair information features on the hourly calendar: each leg's OFI / VPIN /
    Kyle-lambda / spread, plus cross-leg combinations the models can use directly.
    """
    fa = hourly_information_features(sym_a, start, end_exclusive, levels)
    fb = hourly_information_features(sym_b, start, end_exclusive, levels)
    if fa.empty or fb.empty:
        return pd.DataFrame()
    out = fa.add_suffix("_a").join(fb.add_suffix("_b"), how="inner")
    # Cross-leg information signals.
    out["ofi_divergence"] = out["order_flow_imbalance_a"] - out["order_flow_imbalance_b"]
    out["vpin_max"] = out[["vpin_a", "vpin_b"]].max(axis=1)
    out["volume_quote_ratio"] = (
        out["volume_quote_a"] / out["volume_quote_b"].replace(0.0, np.nan)
    )
    return out
