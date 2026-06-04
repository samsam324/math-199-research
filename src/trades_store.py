"""
Tick-level trade storage + volume/order-flow aggregation.

Companion to `src/l2_store.py`. Where the L2 store answers "what did the book
look like," this answers "what actually traded" — the volume-as-information
signal the phase-2 analysis needs. Source is the Tardis `trades` CSV:

    exchange, symbol, timestamp, local_timestamp, id, side, price, amount

`side` is the aggressor: 'buy' = a market buy lifting the ask (positive order
flow), 'sell' = a market sell hitting the bid (negative). That sign is the
whole point — signed volume is a direct proxy for informed/directional flow.

Two layouts, both one file per symbol-day:

    data/trades/{SYMBOL}/{YYYY-MM-DD}.parquet      1s trade bars (default)

The raw per-tick CSV stays in the gz cache (data/l2_raw/.../trades/...), so
nothing is lost — bars can be rebuilt at any cadence. The parquet we keep is
the 1s aggregation, aligned to the L2 1s book bars so features join cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradesConfig:
    data_dir: str = "data/trades"
    resample: str = "1s"


BAR_COLUMNS = [
    "timestamp",        # right-labeled bar time, UTC
    "trade_count",      # number of trades in the bar
    "volume_base",      # sum of base-asset amount
    "volume_quote",     # sum of price * amount (USDT)
    "buy_volume_base",  # aggressor-buy base volume
    "sell_volume_base", # aggressor-sell base volume
    "signed_volume_base",  # buy - sell (order-flow imbalance)
    "vwap",             # volume_quote / volume_base
]


def _bar_dtype(col: str) -> str:
    if col == "timestamp":
        return "datetime64[ns, UTC]"
    if col == "trade_count":
        return "uint64"
    return "float64"


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=_bar_dtype(c)) for c in BAR_COLUMNS})


def parse_raw_trades_csv(path: Path) -> Iterable[pd.DataFrame]:
    """Yield one per-tick DataFrame per symbol-day from a Tardis trades CSV."""
    raw = pd.read_csv(path)
    if raw.empty:
        return
    symbol = str(raw["symbol"].iloc[0]).upper()
    ts = pd.to_datetime(raw["timestamp"].to_numpy(), unit="us", utc=True)
    tick = pd.DataFrame({
        "timestamp": ts,
        "price": raw["price"].astype("float64"),
        "amount": raw["amount"].astype("float64"),
        "side": raw["side"].astype("string"),
    }).sort_values("timestamp").reset_index(drop=True)
    for day, frame in tick.groupby(tick["timestamp"].dt.floor("D")):
        frame = frame.reset_index(drop=True)
        frame.attrs["symbol"] = symbol
        frame.attrs["day"] = pd.Timestamp(day)
        yield frame


def aggregate_to_bars(tick: pd.DataFrame, rule: str = "1s") -> pd.DataFrame:
    """Aggregate per-tick trades into volume/order-flow bars at `rule` cadence."""
    if tick.empty:
        return empty_bars()
    df = tick.copy()
    df["quote"] = df["price"] * df["amount"]
    is_buy = (df["side"] == "buy").to_numpy()
    df["buy_base"] = np.where(is_buy, df["amount"], 0.0)
    df["sell_base"] = np.where(~is_buy, df["amount"], 0.0)

    g = df.set_index("timestamp").resample(rule, label="right", closed="right")
    bars = pd.DataFrame({
        "trade_count": g.size().astype("uint64"),
        "volume_base": g["amount"].sum(),
        "volume_quote": g["quote"].sum(),
        "buy_volume_base": g["buy_base"].sum(),
        "sell_volume_base": g["sell_base"].sum(),
    })
    bars = bars[bars["trade_count"] > 0].copy()
    bars["signed_volume_base"] = bars["buy_volume_base"] - bars["sell_volume_base"]
    bars["vwap"] = bars["volume_quote"] / bars["volume_base"].replace(0.0, np.nan)
    return bars.reset_index()[BAR_COLUMNS]


def write_day(symbol: str, bars: pd.DataFrame, day: pd.Timestamp, cfg: TradesConfig = TradesConfig()) -> Path:
    out_dir = Path(cfg.data_dir) / symbol.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{day.strftime('%Y-%m-%d')}.parquet"
    out = bars[BAR_COLUMNS].copy()
    out.attrs = {}
    out.to_parquet(path, engine="pyarrow", index=False)
    return path


def load_symbol_range(
    symbol: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    cfg: TradesConfig = TradesConfig(),
) -> pd.DataFrame:
    sym_dir = Path(cfg.data_dir) / symbol.upper()
    if not sym_dir.exists():
        return empty_bars()
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end_exclusive.tzinfo is None:
        end_exclusive = end_exclusive.tz_localize("UTC")
    days = pd.date_range(start.floor("D"), end_exclusive.ceil("D"), freq="D", tz="UTC")
    frames: List[pd.DataFrame] = []
    for d in days:
        path = sym_dir / f"{d.strftime('%Y-%m-%d')}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, engine="pyarrow"))
    if not frames:
        return empty_bars()
    out = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    mask = (out["timestamp"] >= start) & (out["timestamp"] < end_exclusive)
    return out.loc[mask].reset_index(drop=True)
