# src/data_store.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


# If api.binance.com is blocked for you, Binance.US is typically accessible in the US.
DEFAULT_BASE_URL = "https://api.binance.us"

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

_INTERVAL_MS: Dict[str, int] = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


@dataclass(frozen=True)
class StoreConfig:
    base_url: str = DEFAULT_BASE_URL
    interval: str = "1h"
    quote_asset: str = "USDT"
    data_dir: str = "data"

    # HTTP behavior
    request_timeout_s: float = 10.0
    max_retries: int = 8
    min_request_spacing_s: float = 0.06
    kline_limit: int = 1000

    # Universe filtering
    exclude_leveraged_tokens: bool = True  # excludes *UP/*DOWN/*BULL/*BEAR
    max_symbols: Optional[int] = None      # set e.g. 50 for testing


def _to_utc_ts(x: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _now_floor(interval: str) -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor(interval)


def _get_json(
    session: requests.Session,
    url: str,
    params: Optional[dict],
    timeout_s: float,
    max_retries: int,
) -> Any:
    params = params or {}
    for attempt in range(max_retries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout_s)
            if r.status_code == 200:
                return r.json()

            # Rate limit / temporary issues
            if r.status_code in (418, 429) or (500 <= r.status_code < 600):
                sleep_s = min(30.0, 0.5 * (2 ** attempt))
                time.sleep(sleep_s)
                continue

            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")

        except (requests.Timeout, requests.ConnectionError):
            sleep_s = min(30.0, 0.5 * (2 ** attempt))
            time.sleep(sleep_s)
            continue

    raise RuntimeError(f"Failed GET {url} after retries")


def data_path(cfg: StoreConfig) -> Path:
    return Path(cfg.data_dir) / f"spot_{cfg.interval}"


def metadata_path(cfg: StoreConfig) -> Path:
    return Path(cfg.data_dir) / "metadata"


def symbol_file(cfg: StoreConfig, symbol: str) -> Path:
    return data_path(cfg) / f"{symbol}.parquet"


def list_local_symbols(cfg: StoreConfig) -> List[str]:
    p = data_path(cfg)
    if not p.exists():
        return []
    return sorted([f.stem for f in p.glob("*.parquet")])


def load_symbol(cfg: StoreConfig, symbol: str, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    f = symbol_file(cfg, symbol)
    if not f.exists():
        return None
    df = pd.read_parquet(f, engine="pyarrow", columns=columns)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    return df


def save_symbol(cfg: StoreConfig, symbol: str, df: pd.DataFrame) -> None:
    out_dir = data_path(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(symbol_file(cfg, symbol), engine="pyarrow", index=True)


def fetch_all_usdt_symbols(cfg: StoreConfig, session: requests.Session) -> List[str]:
    """
    Fetch ALL currently tradable USDT spot symbols from exchangeInfo.

    NOTE: This is current listings only. Delisted historical symbols cannot be retrieved
    via typical free exchange endpoints. We'll document that later in your paper.
    """
    url = f"{cfg.base_url}/api/v3/exchangeInfo"

    # Try with permissions=SPOT (works on Binance.com; Binance.US may ignore)
    try_params = [{"permissions": "SPOT"}, {}]

    info = None
    for params in try_params:
        try:
            info = _get_json(session, url, params=params, timeout_s=cfg.request_timeout_s, max_retries=cfg.max_retries)
            if isinstance(info, dict) and "symbols" in info:
                break
        except Exception:
            continue

    if not info or "symbols" not in info:
        raise RuntimeError("Failed to fetch exchangeInfo symbols.")

    out: List[str] = []
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if s.get("quoteAsset") != cfg.quote_asset:
            continue
        sym = s.get("symbol")
        if not sym:
            continue
        if cfg.exclude_leveraged_tokens and sym.endswith(("UP", "DOWN", "BULL", "BEAR")):
            continue
        out.append(sym)

    out = sorted(set(out))
    if cfg.max_symbols is not None:
        out = out[: cfg.max_symbols]
    return out


def _klines_to_df(rows: List[List[Any]]) -> pd.DataFrame:
    if not rows:
        return ClientEmptyDf()

    arr = np.asarray(rows, dtype=object)
    open_time = pd.to_datetime(arr[:, 0].astype(np.int64), unit="ms", utc=True)

    df = pd.DataFrame(
        {
            "timestamp": open_time,
            "open": arr[:, 1].astype(np.float64),
            "high": arr[:, 2].astype(np.float64),
            "low": arr[:, 3].astype(np.float64),
            "close": arr[:, 4].astype(np.float64),
            "volume": arr[:, 5].astype(np.float64),
        }
    ).set_index("timestamp")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[OHLCV_COLS]


def ClientEmptyDf() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV_COLS).set_index(pd.DatetimeIndex([], tz="UTC", name="timestamp"))


def fetch_klines(
    cfg: StoreConfig,
    session: requests.Session,
    symbol: str,
    start_ms: int,
    end_ms_excl: int,
) -> pd.DataFrame:
    """
    Download spot klines for [start, end) using pagination.
    """
    if cfg.interval not in _INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {cfg.interval}")

    url = f"{cfg.base_url}/api/v3/klines"
    interval_ms = _INTERVAL_MS[cfg.interval]

    all_rows: List[List[Any]] = []
    cur = start_ms
    end_inclusive = end_ms_excl - 1

    while cur < end_ms_excl:
        params = {
            "symbol": symbol,
            "interval": cfg.interval,
            "startTime": int(cur),
            "endTime": int(end_inclusive),
            "limit": int(cfg.kline_limit),
        }
        data = _get_json(session, url, params=params, timeout_s=cfg.request_timeout_s, max_retries=cfg.max_retries)
        time.sleep(cfg.min_request_spacing_s)

        if not data:
            break

        all_rows.extend(data)

        last_open = int(data[-1][0])
        cur = last_open + interval_ms

        if len(data) < cfg.kline_limit:
            break

    return _klines_to_df(all_rows)


def get_last_timestamp_local(cfg: StoreConfig, symbol: str) -> Optional[pd.Timestamp]:
    """
    Reads only the 'close' column to get the latest timestamp efficiently.
    """
    df = load_symbol(cfg, symbol, columns=["close"])
    if df is None or df.empty:
        return None
    return df.index.max()


def merge_and_validate(existing: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge existing and new data, ensure monotonic index, drop duplicates.
    """
    if existing is None or existing.empty:
        out = new.copy()
    else:
        out = pd.concat([existing, new], axis=0)

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # Basic validation: ensure required columns exist
    for c in OHLCV_COLS:
        if c not in out.columns:
            raise RuntimeError(f"Missing column '{c}' after merge.")

    return out[OHLCV_COLS]


def sync_store_all(cfg: StoreConfig) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Automatic "validate and update" behavior:

    - Fetch current tradable symbols.
    - For each symbol:
        - If not present locally: download full history from epoch -> now.
        - If present locally: download missing tail from (last_ts + interval) -> now.
    - Merge, dedupe, validate, write parquet.

    Returns:
      (synced_symbols, failures)
    """
    data_path(cfg).mkdir(parents=True, exist_ok=True)
    metadata_path(cfg).mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    symbols = fetch_all_usdt_symbols(cfg, session=session)
    now_end = _now_floor(cfg.interval)
    end_ms_excl = int(now_end.value // 1_000_000)

    interval_ms = _INTERVAL_MS[cfg.interval]

    synced: List[str] = []
    failures: List[Tuple[str, str]] = []

    print(f"Symbols to sync (current listings): {len(symbols)}")
    print(f"Sync target end (UTC, floored): {now_end} | interval={cfg.interval} | base_url={cfg.base_url}")

    for sym in tqdm(symbols, desc="Syncing symbols"):
        try:
            f = symbol_file(cfg, sym)
            existing = None

            if f.exists():
                # Load full existing file once (we need it to merge)
                existing = load_symbol(cfg, sym)
                last_ts = existing.index.max() if existing is not None and not existing.empty else None
            else:
                last_ts = None

            if last_ts is None:
                # New symbol locally: fetch max available (epoch -> now)
                start_ms = 0
            else:
                # Existing: fetch only the missing tail
                last_ms = int(last_ts.value // 1_000_000)
                start_ms = last_ms + interval_ms

            if start_ms >= end_ms_excl:
                # Already up to date
                if existing is not None and not existing.empty:
                    synced.append(sym)
                else:
                    # edge case: file exists but empty; try full download next time
                    failures.append((sym, "local_empty_or_invalid"))
                continue

            new_df = fetch_klines(cfg, session, sym, start_ms=start_ms, end_ms_excl=end_ms_excl)

            # If we got nothing new, still consider synced
            if new_df.empty:
                if existing is not None and not existing.empty:
                    synced.append(sym)
                else:
                    failures.append((sym, "no_data_returned"))
                continue

            merged = merge_and_validate(existing, new_df)

            # Write back to disk
            save_symbol(cfg, sym, merged)
            synced.append(sym)

        except Exception as e:
            failures.append((sym, str(e)))

    # Write metadata snapshot
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    meta = {
        "run_time_utc": ts,
        "base_url": cfg.base_url,
        "interval": cfg.interval,
        "quote_asset": cfg.quote_asset,
        "symbols_attempted": len(symbols),
        "symbols_synced": len(synced),
        "failures_count": len(failures),
        "failures_sample": failures[:200],
        "end_utc_floored": str(now_end),
        "note": "Sync uses current listings. New symbols are backfilled epoch->now; existing symbols are appended last_ts->now.",
    }
    (metadata_path(cfg) / f"sync_run_{ts}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return synced, failures


def build_close_panel(
    cfg: StoreConfig,
    symbols: List[str],
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    min_symbol_coverage: float = 0.90,
) -> pd.DataFrame:
    """
    Build a close-price panel on a fixed calendar for [start, end) using local data only.

    Design:
      - Reindex each symbol to the expected hourly calendar so gaps become NaN.
      - Compute per-symbol coverage = fraction of calendar timestamps with a close.
      - Drop symbols with coverage < min_symbol_coverage.
      - Forward-fill closes to remove intermittent gaps (explicit choice).
      - Drop remaining NaN rows (usually only the initial prefix before the first close).

    Output:
      index: expected timestamps at cfg.interval
      columns: symbols that pass min_symbol_coverage
      values: close prices (float), mostly gap-free after ffill
    """
    start = _to_utc_ts(start).floor(cfg.interval)
    end_exclusive = _to_utc_ts(end_exclusive).floor(cfg.interval)
    if end_exclusive <= start:
        return pd.DataFrame()

    step = pd.tseries.frequencies.to_offset(cfg.interval)
    end_inclusive = end_exclusive - step
    if end_inclusive < start:
        return pd.DataFrame()

    calendar = pd.date_range(
        start=start,
        end=end_inclusive,
        freq=cfg.interval,
        tz="UTC",
        name="timestamp",
    )

    frames = []
    for sym in symbols:
        df = load_symbol(cfg, sym, columns=["close"])
        if df is None or df.empty:
            continue

        w = df.loc[(df.index >= start) & (df.index < end_exclusive), ["close"]]
        if w.empty:
            continue

        w = w.reindex(calendar).rename(columns={"close": sym})
        frames.append(w)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, axis=1).sort_index()

    # Coverage measured against the fixed calendar
    coverage = panel.notna().mean(axis=0)
    keep = coverage[coverage >= min_symbol_coverage].index.tolist()
    panel = panel[keep]

    if panel.empty or panel.shape[1] < 2:
        return pd.DataFrame()

    # Forward-fill closes to handle intermittent gaps
    panel = panel.ffill()

    # Remaining NaNs are typically only at the very beginning before a symbol’s first close
    panel = panel.dropna(axis=0, how="any")

    return panel