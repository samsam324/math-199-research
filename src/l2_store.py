"""
L2 order-book storage layout and ingestion stub.

STATUS: blocked on UCLA data access. This module locks in the schema and the
read/write interface so downstream code (feature engineering, microstructure
backtests) can be built against it now, and so the only thing that has to
change when the data arrives is the ingestion function body in `_parse_raw_*`
helpers.

Layout
------
data/l2/{symbol}/{YYYY-MM-DD}.parquet

One file per symbol per UTC day. Columns:

    timestamp        nanosecond-precision UTC (datetime64[ns, UTC])
    seq              monotone exchange sequence number (uint64) for dedupe
    bid_px_1..K      top-K bid prices (float64), 1 = best bid
    bid_sz_1..K      top-K bid sizes  (float64)
    ask_px_1..K      top-K ask prices (float64)
    ask_sz_1..K      top-K ask sizes  (float64)
    is_snapshot      bool, true for the initial book snapshot at session start
    is_delta         bool, true for incremental updates (mutually exclusive)

K is fixed per dataset (default 10). The schema is intentionally column-wise
and not nested so parquet predicate pushdown works for time-range queries
without parsing variable-length lists.

Sub-second timestamps are required: spread/mid-quote features depend on
millisecond ordering against the trade feed.

Source plan
-----------
UCLA is expected to provide either a Binance-style WebSocket capture (per-
symbol depth update stream, replayed against an initial REST snapshot) or a
flat CSV/parquet dump from a market-data vendor. Both can land in the same
schema; only `_parse_raw_websocket_dump` and `_parse_raw_csv_dump` need to be
filled in.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class L2Config:
    levels: int = 10
    data_dir: str = "data/l2"
    timezone: str = "UTC"


def expected_columns(cfg: L2Config = L2Config()) -> List[str]:
    cols: List[str] = ["timestamp", "seq"]
    for side in ("bid", "ask"):
        for px_or_sz in ("px", "sz"):
            cols.extend([f"{side}_{px_or_sz}_{i}" for i in range(1, cfg.levels + 1)])
    cols.extend(["is_snapshot", "is_delta"])
    return cols


def empty_frame(cfg: L2Config = L2Config()) -> pd.DataFrame:
    """Empty DataFrame with the canonical schema. Useful for tests."""
    cols = expected_columns(cfg)
    df = pd.DataFrame({c: pd.Series(dtype=_dtype_for(c)) for c in cols})
    return df


def _dtype_for(col: str) -> str:
    if col == "timestamp":
        return "datetime64[ns, UTC]"
    if col == "seq":
        return "uint64"
    if col in ("is_snapshot", "is_delta"):
        return "bool"
    return "float64"


def validate_frame(df: pd.DataFrame, cfg: L2Config = L2Config()) -> None:
    """Raise if df does not match the canonical schema."""
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


def write_day(symbol: str, df: pd.DataFrame, day: pd.Timestamp, cfg: L2Config = L2Config()) -> Path:
    """Write one symbol-day file in canonical layout."""
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
    """Concatenate symbol-day files covering [start, end_exclusive)."""
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


# ---------------------------------------------------------------------------
# Microstructure features that L2 unlocks. These are stubbed; once `load_symbol_range`
# returns real data, the feature math below is correct and ready to use.
# ---------------------------------------------------------------------------


def microprice(book: pd.DataFrame, level: int = 1) -> pd.Series:
    """
    Size-weighted mid: (bid_sz * ask_px + ask_sz * bid_px) / (bid_sz + ask_sz).
    Reduces to mid-quote when sizes are equal. Better short-horizon predictor
    of next trade price than the simple mid.
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
    """
    Sum of bid sizes minus ask sizes over the top N levels, scaled by their
    sum. Range [-1, 1]; positive means buying pressure.
    """
    bid_sz = sum(book[f"bid_sz_{i}"].astype(float) for i in range(1, depth_levels + 1))
    ask_sz = sum(book[f"ask_sz_{i}"].astype(float) for i in range(1, depth_levels + 1))
    total = (bid_sz + ask_sz).replace(0.0, np.nan)
    return (bid_sz - ask_sz) / total


# ---------------------------------------------------------------------------
# Ingestion stubs. Body to be filled in once UCLA dump format is known.
# ---------------------------------------------------------------------------


def _parse_raw_websocket_dump(path: Path, cfg: L2Config = L2Config()) -> Iterable[pd.DataFrame]:
    """
    Yield one DataFrame per symbol-day from a Binance-style depth-stream
    capture. BLOCKED: needs sample data + format spec from UCLA.
    """
    raise NotImplementedError("Awaiting UCLA data sample; see docs/l2_data.md when written.")


def _parse_raw_csv_dump(path: Path, cfg: L2Config = L2Config()) -> Iterable[pd.DataFrame]:
    """
    Yield one DataFrame per symbol-day from a vendor CSV dump.
    BLOCKED: needs format spec from UCLA.
    """
    raise NotImplementedError("Awaiting UCLA data sample; see docs/l2_data.md when written.")


def ingest_directory(raw_root: Path, cfg: L2Config = L2Config(), *, fmt: str = "websocket") -> Optional[int]:
    """
    Walk a directory of raw exchange captures and write canonical symbol-day
    parquet files into cfg.data_dir. Returns the count of files written, or
    None if no ingestion ran.
    """
    raw_root = Path(raw_root)
    if not raw_root.exists():
        return None
    parsers = {"websocket": _parse_raw_websocket_dump, "csv": _parse_raw_csv_dump}
    if fmt not in parsers:
        raise ValueError(f"Unknown raw format: {fmt}")
    parse = parsers[fmt]

    written = 0
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        for frame in parse(path, cfg=cfg):
            symbol = str(frame.attrs.get("symbol", "")).upper()
            day = pd.Timestamp(frame.attrs.get("day"))
            if not symbol or pd.isna(day):
                continue
            write_day(symbol, frame, day, cfg=cfg)
            written += 1
    return written
