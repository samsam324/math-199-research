"""
L2 order-book canonical storage.

Layout: data/l2/{symbol}/{YYYY-MM-DD}.parquet

Columns (K = configurable depth, default 10):
    timestamp        datetime64[ns, UTC]    exchange timestamp
    local_timestamp  datetime64[ns, UTC]    Tardis-side receive timestamp (may be null)
    seq              uint64                 monotone sequence for dedupe (0 if unavailable)
    bid_px_1..K      float64                bid prices, 1 = best
    bid_sz_1..K      float64                bid sizes  (base asset units)
    ask_px_1..K      float64                ask prices, 1 = best
    ask_sz_1..K      float64                ask sizes  (base asset units)
    is_snapshot      bool                   true for initial/refresh snapshot
    is_delta         bool                   true for incremental update (mutually exclusive)

A book row gives the full top-K state at the moment the message was applied.
For book_snapshot_25 from Tardis this is direct. For incremental_book_L2,
the ingestion layer (`tardis_ingest._reconstruct_book_from_incremental`) walks
the diff stream and emits a snapshot row at every update.

Sub-second timestamps matter: the trades/quotes feeds align by exchange
timestamp at microsecond resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class L2Config:
    levels: int = 10
    data_dir: str = "data/l2"
    timezone: str = "UTC"


def expected_columns(cfg: L2Config = L2Config()) -> List[str]:
    cols: List[str] = ["timestamp", "local_timestamp", "seq"]
    for side in ("bid", "ask"):
        for px_or_sz in ("px", "sz"):
            cols.extend([f"{side}_{px_or_sz}_{i}" for i in range(1, cfg.levels + 1)])
    cols.extend(["is_snapshot", "is_delta"])
    return cols


def _dtype_for(col: str) -> str:
    if col in ("timestamp", "local_timestamp"):
        return "datetime64[ns, UTC]"
    if col == "seq":
        return "uint64"
    if col in ("is_snapshot", "is_delta"):
        return "bool"
    return "float64"


def empty_frame(cfg: L2Config = L2Config()) -> pd.DataFrame:
    cols = expected_columns(cfg)
    return pd.DataFrame({c: pd.Series(dtype=_dtype_for(c)) for c in cols})


def validate_frame(df: pd.DataFrame, cfg: L2Config = L2Config()) -> None:
    expected = expected_columns(cfg)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"L2 frame missing columns: {missing}")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("L2 timestamps must be monotone non-decreasing")
    if df["timestamp"].dt.tz is None:
        raise ValueError("L2 timestamps must be UTC-aware")
    if (df["is_snapshot"] & df["is_delta"]).any():
        raise ValueError("is_snapshot and is_delta are mutually exclusive")
    # Bid/ask crossed check: bid_px_1 <= ask_px_1 (when both non-NaN)
    bid1 = df["bid_px_1"]; ask1 = df["ask_px_1"]
    both = bid1.notna() & ask1.notna()
    if (bid1[both] > ask1[both]).any():
        raise ValueError("Crossed book: bid_px_1 > ask_px_1 in some rows")


def write_day(symbol: str, df: pd.DataFrame, day: pd.Timestamp, cfg: L2Config = L2Config()) -> Path:
    validate_frame(df, cfg)
    out_dir = Path(cfg.data_dir) / symbol.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{day.strftime('%Y-%m-%d')}.parquet"
    df[expected_columns(cfg)].to_parquet(path, engine="pyarrow", index=False)
    return path


def list_local_symbols(cfg: L2Config = L2Config()) -> List[str]:
    root = Path(cfg.data_dir)
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def load_symbol_range(
    symbol: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    cfg: L2Config = L2Config(),
) -> pd.DataFrame:
    sym_dir = Path(cfg.data_dir) / symbol.upper()
    if not sym_dir.exists():
        return empty_frame(cfg)
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
        return empty_frame(cfg)
    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values(["timestamp", "seq"]).reset_index(drop=True)
    mask = (out["timestamp"] >= start) & (out["timestamp"] < end_exclusive)
    return out.loc[mask].reset_index(drop=True)


# -------------------- Top-of-book helpers --------------------

def best_bid_ask(book: pd.DataFrame) -> pd.DataFrame:
    """Convenience: return (timestamp, bid, ask, bid_sz, ask_sz) for quick spread math."""
    return book[["timestamp", "bid_px_1", "ask_px_1", "bid_sz_1", "ask_sz_1"]].rename(
        columns={"bid_px_1": "bid", "ask_px_1": "ask", "bid_sz_1": "bid_sz", "ask_sz_1": "ask_sz"}
    )


def midprice(book: pd.DataFrame) -> pd.Series:
    return (book["bid_px_1"].astype(float) + book["ask_px_1"].astype(float)) / 2.0
