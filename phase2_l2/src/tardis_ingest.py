"""
Tardis ingestion: download daily CSVs, convert to canonical parquet.

Tardis serves one CSV per (exchange, data_type, symbol, UTC day) at:
    https://datasets.tardis.dev/v1/{exchange}/{data_type}/{YYYY}/{MM}/{DD}/{SYMBOL}.csv.gz

with bearer-token auth on the API key. Free tier returns the first day of each
month without auth.

DATA TYPES we use:
  - book_snapshot_25       : 25-level book snapshots
  - incremental_book_L2    : every L2 update (for higher-fidelity reconstruction)
  - trades                 : individual trades with aggressor side
  - derivative_ticker      : funding rate, OI, mark price (perps only)

THE PARSING SEAM:
  `_tardis_csv_to_l2_canonical` and `_tardis_csv_to_trades_canonical` are the
  ONLY two functions that depend on the exact Tardis CSV column layout. When
  the reference `data.py` from the team arrives (or the docs at
  https://docs.tardis.dev are checked), update these two functions only.
  Everything downstream uses the canonical schema in l2_store / trade_store.

This module avoids hitting the network at import time. The real downloader is
gated behind `download_day()`. With no API key set, calls fail with a clear
message.
"""
from __future__ import annotations

import gzip
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from src.l2_store import L2Config, expected_columns as l2_expected_columns, write_day as l2_write_day
from src.trade_store import TradeConfig, CANONICAL_COLUMNS as TRADE_COLUMNS, write_day as trade_write_day


TARDIS_BASE_URL = "https://datasets.tardis.dev/v1"


@dataclass(frozen=True)
class TardisConfig:
    api_key: Optional[str] = None  # falls back to TARDIS_API_KEY env var
    cache_root: str = "data/cache/tardis"
    request_timeout_s: float = 60.0
    max_retries: int = 5
    user_agent: str = "phase2_l2-ingest/0.1"


def _api_key(cfg: TardisConfig) -> Optional[str]:
    return cfg.api_key or os.environ.get("TARDIS_API_KEY")


def _csv_url(exchange: str, data_type: str, day: pd.Timestamp, symbol: str) -> str:
    y = day.strftime("%Y"); m = day.strftime("%m"); d = day.strftime("%d")
    return f"{TARDIS_BASE_URL}/{exchange}/{data_type}/{y}/{m}/{d}/{symbol.upper()}.csv.gz"


def download_day_raw(
    exchange: str, data_type: str, day: pd.Timestamp, symbol: str,
    cfg: TardisConfig = TardisConfig(),
) -> Optional[bytes]:
    """
    Returns the raw gzipped CSV bytes for a single (exchange, data_type, day, symbol),
    or None if the file is missing on Tardis. Raises on auth/network failures.
    """
    url = _csv_url(exchange, data_type, day, symbol)
    headers = {"User-Agent": cfg.user_agent}
    key = _api_key(cfg)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    last_exc: Optional[Exception] = None
    for attempt in range(cfg.max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=cfg.request_timeout_s)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
            if r.status_code in (401, 403):
                raise RuntimeError(f"Tardis auth failure ({r.status_code}). Check TARDIS_API_KEY.")
            if r.status_code in (429,) or 500 <= r.status_code < 600:
                continue
            raise RuntimeError(f"Tardis HTTP {r.status_code}: {r.text[:200]}")
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to download {url} after {cfg.max_retries} attempts")


def _read_gz_csv(blob: bytes) -> pd.DataFrame:
    """
    Parse a CSV blob. Auto-detects gzip via the magic bytes (0x1f 0x8b) so this
    works whether the server served the raw .csv.gz file or auto-decompressed
    via Content-Encoding: gzip.
    """
    if len(blob) >= 2 and blob[0] == 0x1F and blob[1] == 0x8B:
        with gzip.GzipFile(fileobj=io.BytesIO(blob), mode="rb") as f:
            return pd.read_csv(f)
    return pd.read_csv(io.BytesIO(blob))


def canonical_l2_path(symbol: str, day: pd.Timestamp, cfg: L2Config) -> Path:
    return Path(cfg.data_dir) / symbol.upper() / f"{day.strftime('%Y-%m-%d')}.parquet"


def canonical_trades_path(symbol: str, day: pd.Timestamp, cfg: TradeConfig) -> Path:
    return Path(cfg.data_dir) / symbol.upper() / f"{day.strftime('%Y-%m-%d')}.parquet"


# -------------------- THE PARSING SEAM --------------------
# These two functions hold ALL Tardis-specific column knowledge. If the team's
# `data.py` reveals a different layout, update only here.

def _tardis_book_snapshot_25_to_canonical(raw: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    """
    Convert Tardis `book_snapshot_25` CSV to our canonical L2 schema (top-K).

    Documented Tardis columns:
      exchange, symbol, timestamp (microseconds since epoch),
      local_timestamp (microseconds since epoch),
      asks[0].price, asks[0].amount, ..., asks[24].price, asks[24].amount,
      bids[0].price, bids[0].amount, ..., bids[24].price, bids[24].amount

    We project to top-K bid/ask, rename to our naming convention, convert
    timestamps to UTC.
    """
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(raw["timestamp"].astype("int64"), unit="us", utc=True)
    if "local_timestamp" in raw.columns:
        out["local_timestamp"] = pd.to_datetime(raw["local_timestamp"].astype("int64"), unit="us", utc=True)
    else:
        out["local_timestamp"] = pd.NaT
    out["seq"] = np.uint64(0)  # book_snapshot_25 has no exchange seq; rely on timestamp

    for i in range(1, levels + 1):
        bid_px_col = f"bids[{i-1}].price"
        bid_sz_col = f"bids[{i-1}].amount"
        ask_px_col = f"asks[{i-1}].price"
        ask_sz_col = f"asks[{i-1}].amount"
        out[f"bid_px_{i}"] = raw[bid_px_col].astype(float) if bid_px_col in raw.columns else np.nan
        out[f"bid_sz_{i}"] = raw[bid_sz_col].astype(float) if bid_sz_col in raw.columns else np.nan
        out[f"ask_px_{i}"] = raw[ask_px_col].astype(float) if ask_px_col in raw.columns else np.nan
        out[f"ask_sz_{i}"] = raw[ask_sz_col].astype(float) if ask_sz_col in raw.columns else np.nan

    out["is_snapshot"] = True
    out["is_delta"] = False
    return out.sort_values("timestamp").reset_index(drop=True)


def _tardis_trades_to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Tardis `trades` CSV to our canonical trades schema.

    Documented Tardis columns:
      exchange, symbol, timestamp (us), local_timestamp (us), id, side, price, amount
    """
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(raw["timestamp"].astype("int64"), unit="us", utc=True)
    if "local_timestamp" in raw.columns:
        out["local_timestamp"] = pd.to_datetime(raw["local_timestamp"].astype("int64"), unit="us", utc=True)
    else:
        out["local_timestamp"] = pd.NaT
    out["trade_id"] = raw["id"].astype(str) if "id" in raw.columns else ""
    side_raw = raw["side"].astype(str).str.lower() if "side" in raw.columns else ""
    out["side"] = pd.Categorical(side_raw, categories=["buy", "sell", ""])
    out["price"] = raw["price"].astype(float)
    out["amount"] = raw["amount"].astype(float)
    out["notional"] = out["price"] * out["amount"]
    return out.sort_values("timestamp").reset_index(drop=True)


# -------------------- Public API --------------------

def ingest_book_day(
    exchange: str, day: pd.Timestamp, symbol: str,
    cfg_tardis: TardisConfig = TardisConfig(),
    cfg_l2: L2Config = L2Config(),
    data_type: str = "book_snapshot_25",
    skip_existing: bool = True,
) -> Optional[Path]:
    """Pull one (exchange, day, symbol) of L2 from Tardis, write canonical parquet."""
    out_path = canonical_l2_path(symbol, day, cfg_l2)
    if skip_existing and out_path.exists():
        return out_path
    blob = download_day_raw(exchange, data_type, day, symbol, cfg=cfg_tardis)
    if blob is None:
        return None
    raw = _read_gz_csv(blob)
    canon = _tardis_book_snapshot_25_to_canonical(raw, levels=cfg_l2.levels)
    return l2_write_day(symbol, canon, day, cfg=cfg_l2)


def ingest_trades_day(
    exchange: str, day: pd.Timestamp, symbol: str,
    cfg_tardis: TardisConfig = TardisConfig(),
    cfg_trades: TradeConfig = TradeConfig(),
    skip_existing: bool = True,
) -> Optional[Path]:
    """Pull one (exchange, day, symbol) of trades from Tardis, write canonical parquet."""
    out_path = canonical_trades_path(symbol, day, cfg_trades)
    if skip_existing and out_path.exists():
        return out_path
    blob = download_day_raw(exchange, "trades", day, symbol, cfg=cfg_tardis)
    if blob is None:
        return None
    raw = _read_gz_csv(blob)
    canon = _tardis_trades_to_canonical(raw)
    return trade_write_day(symbol, canon, day, cfg=cfg_trades)


def ingest_range(
    exchange: str, symbols: Iterable[str],
    start_day: pd.Timestamp, end_day_exclusive: pd.Timestamp,
    data_types: Iterable[str] = ("book_snapshot_25", "trades"),
    cfg_tardis: TardisConfig = TardisConfig(),
    cfg_l2: L2Config = L2Config(),
    cfg_trades: TradeConfig = TradeConfig(),
    *,
    max_concurrency: int = 4,
    skip_existing: bool = True,
    show_progress: bool = True,
) -> dict:
    """
    Walk the (symbol x day x data_type) grid and write canonical parquet for
    each successful pull. Concurrent downloads with a thread pool (bounded by
    max_concurrency). Idempotent: by default skips any (symbol, day, data_type)
    whose canonical parquet already exists. Progress bar via tqdm.

    Default max_concurrency=4: book_snapshot_25 days are ~90 MB each and
    decompression is CPU-bound; 4 concurrent keeps memory usage and CPU
    sane on a laptop. Bump higher on a cluster.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None  # type: ignore

    days = pd.date_range(start_day, end_day_exclusive - pd.Timedelta(days=1), freq="D", tz="UTC")

    # Build the task grid. Skip already-done jobs up front so the progress bar
    # reflects only real work.
    tasks: List[Tuple[str, pd.Timestamp, str]] = []
    skipped_existing = 0
    for sym in symbols:
        for day in days:
            for dt in data_types:
                if skip_existing:
                    if dt == "trades":
                        if canonical_trades_path(sym, day, cfg_trades).exists():
                            skipped_existing += 1
                            continue
                    elif dt == "book_snapshot_25":
                        if canonical_l2_path(sym, day, cfg_l2).exists():
                            skipped_existing += 1
                            continue
                tasks.append((sym, day, dt))

    summary = {
        "requested": len(tasks) + skipped_existing,
        "skipped_existing": skipped_existing,
        "written": 0,
        "missing": 0,
        "errors": [],
    }

    def _one(task):
        sym, day, dt = task
        try:
            if dt == "trades":
                path = ingest_trades_day(exchange, day, sym, cfg_tardis, cfg_trades, skip_existing=False)
            elif dt == "book_snapshot_25":
                path = ingest_book_day(exchange, day, sym, cfg_tardis, cfg_l2, data_type=dt, skip_existing=False)
            else:
                raise ValueError(f"Unsupported data_type: {dt}")
            return ("ok" if path is not None else "missing", task, None)
        except Exception as exc:
            return ("error", task, str(exc))

    if not tasks:
        return summary

    iterator: Iterable
    if max_concurrency <= 1:
        iterator = (_one(t) for t in tasks)
    else:
        executor = ThreadPoolExecutor(max_workers=max_concurrency)
        futures = [executor.submit(_one, t) for t in tasks]
        iterator = (f.result() for f in as_completed(futures))

    pbar = tqdm(total=len(tasks), desc="Tardis pulls") if (show_progress and tqdm is not None) else None
    try:
        for status, task, err in iterator:
            sym, day, dt = task
            if status == "ok":
                summary["written"] += 1
            elif status == "missing":
                summary["missing"] += 1
            else:
                summary["errors"].append((sym, str(day.date()), dt, err))
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"written": summary["written"], "miss": summary["missing"], "err": len(summary["errors"])}, refresh=False)
    finally:
        if pbar is not None:
            pbar.close()
        if max_concurrency > 1:
            executor.shutdown(wait=False)

    return summary
