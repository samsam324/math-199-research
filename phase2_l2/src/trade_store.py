"""
Trades canonical storage.

Layout: data/trades/{symbol}/{YYYY-MM-DD}.parquet

Columns:
    timestamp        datetime64[ns, UTC]   exchange timestamp
    local_timestamp  datetime64[ns, UTC]   Tardis receive timestamp (may be null)
    trade_id         string                exchange trade id (may be empty)
    side             category              'buy' (taker buy) or 'sell' (taker sell);
                                           empty if unknown -> downstream signs by tick rule
    price            float64               trade price
    amount           float64               base asset size
    notional         float64               price * amount, denominated in quote (USDT)

`side` is the aggressor side as reported by the exchange. If absent, we leave
it empty and `microstructure.sign_trades_tick_rule` reassigns based on the
midprice trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass(frozen=True)
class TradeConfig:
    data_dir: str = "data/trades"


CANONICAL_COLUMNS = [
    "timestamp", "local_timestamp", "trade_id",
    "side", "price", "amount", "notional",
]


def _dtype_for(col: str):
    if col in ("timestamp", "local_timestamp"):
        return "datetime64[ns, UTC]"
    if col == "trade_id":
        return "string"
    if col == "side":
        return pd.CategoricalDtype(categories=["buy", "sell", ""])
    return "float64"


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=_dtype_for(c)) for c in CANONICAL_COLUMNS})


def validate_frame(df: pd.DataFrame) -> None:
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Trades frame missing columns: {missing}")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Trade timestamps must be monotone non-decreasing")
    if df["timestamp"].dt.tz is None:
        raise ValueError("Trade timestamps must be UTC-aware")
    if (df["price"] <= 0).any():
        raise ValueError("Non-positive trade prices")
    if (df["amount"] < 0).any():
        raise ValueError("Negative trade amount")


def write_day(symbol: str, df: pd.DataFrame, day: pd.Timestamp, cfg: TradeConfig = TradeConfig()) -> Path:
    validate_frame(df)
    out_dir = Path(cfg.data_dir) / symbol.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{day.strftime('%Y-%m-%d')}.parquet"
    df[CANONICAL_COLUMNS].to_parquet(path, engine="pyarrow", index=False)
    return path


def list_local_symbols(cfg: TradeConfig = TradeConfig()) -> List[str]:
    root = Path(cfg.data_dir)
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def load_symbol_range(
    symbol: str, start: pd.Timestamp, end_exclusive: pd.Timestamp,
    cfg: TradeConfig = TradeConfig(),
) -> pd.DataFrame:
    sym_dir = Path(cfg.data_dir) / symbol.upper()
    if not sym_dir.exists():
        return empty_frame()
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end_exclusive.tzinfo is None:
        end_exclusive = end_exclusive.tz_localize("UTC")
    days = pd.date_range(start.floor("D"), end_exclusive.ceil("D"), freq="D", tz="UTC")
    frames: List[pd.DataFrame] = []
    for d in days:
        path = sym_dir / f"{d.strftime('%Y-%m-%d')}.parquet"
        if not path.exists():
            continue
        frames.append(pd.read_parquet(path, engine="pyarrow"))
    if not frames:
        return empty_frame()
    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    mask = (out["timestamp"] >= start) & (out["timestamp"] < end_exclusive)
    return out.loc[mask].reset_index(drop=True)
