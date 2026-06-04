"""
Download Binance-spot historical data from Tardis.dev and ingest into the
canonical parquet layouts used by the research pipeline. Built to fetch
everything needed to redo the phase-1 results on L2 + tick volume:

  book_snapshot_25  -> data/l2/{SYMBOL}/{date}.parquet      (1s book bars)
  trades            -> data/trades/{SYMBOL}/{date}.parquet  (1s volume/flow bars)

Resilience
----------
* Caching: raw gzipped CSV is kept forever at
      data/l2_raw/{exchange}/{data_type}/{SYMBOL}/{date}.csv.gz
  A file present and gzip-valid is skipped — never re-downloaded ("cache every GB").
* Per-file retry: each download is retried with backoff on any error.
* Resume loop: after a full pass the script recomputes what's still missing and
  loops until everything is cached or --max-passes is hit. If the process dies,
  just run it again — it picks up exactly where it stopped.

Usage
-----
  python scripts/download_tardis_l2.py --symbols-file data/l2_universe_top50.txt \
      --data-types book_snapshot_25 trades --from 2024-01-01 --to 2025-01-01 --resample 1s
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import l2_store, trades_store  # noqa: E402

EXCHANGE = "binance"
RAW_ROOT = REPO_ROOT / "data" / "l2_raw"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MAX_FILE_RETRIES = 4
RETRY_BACKOFF_S = 5.0


def load_api_key() -> Optional[str]:
    import os

    key = os.environ.get("TARDIS_API_KEY")
    if key:
        return key.strip()
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("TARDIS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _gzip_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 20):
                pass
        return True
    except (OSError, EOFError):
        return False


def raw_path(data_type: str, symbol: str, day: pd.Timestamp) -> Path:
    return RAW_ROOT / EXCHANGE / data_type / symbol.upper() / f"{day:%Y-%m-%d}.csv.gz"


def _download_one(data_type: str, symbol: str, day: pd.Timestamp, api_key: Optional[str]) -> bool:
    """Download a single symbol-day with retries. Returns True if cached & valid."""
    from tardis_dev import download_datasets

    dest = raw_path(data_type, symbol, day)
    if _gzip_ok(dest):
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    next_day = (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for attempt in range(1, MAX_FILE_RETRIES + 1):
        try:
            download_datasets(
                exchange=EXCHANGE,
                data_types=[data_type],
                symbols=[symbol],
                from_date=day.strftime("%Y-%m-%d"),
                to_date=next_day,
                api_key=api_key,
                download_dir=str(dest.parent),
                get_filename=lambda *_a, _n=dest.name, **_k: _n,
            )
            if _gzip_ok(dest):
                return True
            return False  # 200 but no data for this symbol-day (e.g. not listed yet)
        except Exception as e:  # network/HTTP/rate-limit -> back off and retry
            if dest.exists() and not _gzip_ok(dest):
                dest.unlink(missing_ok=True)  # drop partial so next attempt re-fetches
            if attempt == MAX_FILE_RETRIES:
                print(f"  FAILED {data_type}/{symbol}/{day:%Y-%m-%d} after {attempt} tries: {e}", flush=True)
                return False
            time.sleep(RETRY_BACKOFF_S * attempt)
    return False


def _expected_jobs(symbols, data_types, days) -> List[Tuple[str, str, pd.Timestamp]]:
    # Chronological / day-major: fill a complete cross-section (all symbols, both
    # datatypes) for each day before advancing. This means the walk-forward can
    # run on a contiguous window as soon as the early days land, instead of
    # waiting for all of one datatype to finish.
    return [(dt, s, d) for d in days for dt in data_types for s in symbols]


def _free_gb(path: Path = RAW_ROOT) -> float:
    target = path if path.exists() else REPO_ROOT
    return shutil.disk_usage(target).free / 1e9


def download_pass(jobs, api_key, min_free_gb: float) -> Tuple[int, List[Tuple[str, str, pd.Timestamp]], bool]:
    """One pass over all jobs. Returns (n_ok, still_missing, stopped_low_disk)."""
    ok = 0
    missing: List[Tuple[str, str, pd.Timestamp]] = []
    low_disk = False
    for i, (dt, sym, day) in enumerate(jobs):
        # Only the disk guard can halt a pass: never let the cache fill the drive.
        if not _gzip_ok(raw_path(dt, sym, day)) and _free_gb() < min_free_gb:
            low_disk = True
            missing.extend(jobs[i:])
            break
        if _download_one(dt, sym, day, api_key):
            ok += 1
        else:
            missing.append((dt, sym, day))
    return ok, missing, low_disk


# --------------------------------------------------------------------------- #
# Ingest: raw csv.gz -> canonical parquet, filtered to the file's exact UTC day
# --------------------------------------------------------------------------- #

def _resample_book(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    idx = frame.set_index("timestamp")
    out = idx.resample(rule, label="right", closed="right").last().dropna(how="all").reset_index()
    out["seq"] = out["seq"].astype("uint64")
    out["is_snapshot"] = out["is_snapshot"].astype(bool)
    out["is_delta"] = out["is_delta"].astype(bool)
    return out


def _select_day(frames, day: pd.Timestamp):
    """From the per-day groups a parser yields, return the frame for `day` only."""
    for fr in frames:
        if pd.Timestamp(fr.attrs.get("day")).date() == day.date():
            return fr
    return None


def ingest_book(path: Path, day: pd.Timestamp, levels: int, resample: Optional[str], data_dir: str) -> bool:
    cfg = l2_store.L2Config(levels=levels, data_dir=data_dir)
    out_parquet = REPO_ROOT / data_dir / path.parent.name / f"{day:%Y-%m-%d}.parquet"
    if out_parquet.exists():
        return False
    frame = _select_day(l2_store._parse_raw_csv_dump(path, cfg=cfg), day)
    if frame is None or frame.empty:
        return False
    if resample:
        attrs = dict(frame.attrs)
        frame = _resample_book(frame, resample)
        frame.attrs.update(attrs)
    l2_store.write_day(path.parent.name, frame, day, cfg=cfg)
    return True


def ingest_trades(path: Path, day: pd.Timestamp, resample: str, data_dir: str) -> bool:
    cfg = trades_store.TradesConfig(data_dir=data_dir, resample=resample)
    out_parquet = REPO_ROOT / data_dir / path.parent.name / f"{day:%Y-%m-%d}.parquet"
    if out_parquet.exists():
        return False
    tick = _select_day(trades_store.parse_raw_trades_csv(path), day)
    if tick is None or tick.empty:
        return False
    bars = trades_store.aggregate_to_bars(tick, resample)
    trades_store.write_day(path.parent.name, bars, day, cfg=cfg)
    return True


def ingest_all(jobs, levels, resample) -> int:
    written = 0
    for dt, sym, day in jobs:
        path = raw_path(dt, sym, day)
        if not _gzip_ok(path):
            continue
        if dt.startswith("book_snapshot"):
            written += int(ingest_book(path, day, levels, resample, "data/l2"))
        elif dt == "trades":
            written += int(ingest_trades(path, day, resample or "1s", "data/trades"))
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--symbols", nargs="+")
    g.add_argument("--symbols-file", help="newline-delimited symbol list")
    ap.add_argument("--data-types", nargs="+", default=["book_snapshot_25", "trades"])
    ap.add_argument("--from", dest="from_date", required=True)
    ap.add_argument("--to", dest="to_date", required=True, help="exclusive")
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--resample", default="1s", help="bar cadence; '' for full tick")
    ap.add_argument("--max-passes", type=int, default=10, help="resume-loop attempts to clear stragglers")
    ap.add_argument("--min-free-gb", type=float, default=25.0, help="halt downloading new files if free disk drops below this")
    ap.add_argument("--no-ingest", action="store_true")
    ap.add_argument("--ingest-only", action="store_true", help="skip download; ingest whatever raw is already cached")
    args = ap.parse_args()

    if args.symbols_file:
        text = Path(args.symbols_file).read_text()
        symbols = [s.strip().upper() for s in text.replace("\r", "\n").split("\n") if s.strip()]
    else:
        symbols = [s.strip().upper() for s in args.symbols if s.strip()]

    days = list(pd.date_range(args.from_date, pd.Timestamp(args.to_date) - pd.Timedelta(days=1), freq="D"))
    jobs = _expected_jobs(symbols, args.data_types, days)
    api_key = load_api_key()
    print(f"API key: {'present' if api_key else 'MISSING (free first-of-month only)'}", flush=True)
    print(f"{len(symbols)} symbols x {len(args.data_types)} types x {len(days)} days = {len(jobs)} files", flush=True)
    print(f"Types: {args.data_types}  Range: [{args.from_date}, {args.to_date})  Resample: {args.resample or 'tick'}", flush=True)

    if args.ingest_only:
        n = ingest_all(jobs, args.levels, args.resample)
        print(f"Ingested {n} new symbol-day parquet files (data/l2 + data/trades)", flush=True)
        return

    remaining = jobs
    for p in range(1, args.max_passes + 1):
        ok, missing, low_disk = download_pass(remaining, api_key, args.min_free_gb)
        raw_bytes = sum(raw_path(dt, s, d).stat().st_size for dt, s, d in jobs if raw_path(dt, s, d).exists())
        print(f"[pass {p}] cached {len(jobs) - len(missing)}/{len(jobs)}  ({raw_bytes/1e9:,.1f} GB raw, {_free_gb():,.0f} GB free)  missing {len(missing)}", flush=True)
        if low_disk:
            print(f"STOPPED: free disk < {args.min_free_gb} GB. {len(missing)} files left. "
                  f"Free space or point the cache at another drive, then rerun to resume.", flush=True)
            break
        if not missing:
            break
        remaining = missing
        time.sleep(RETRY_BACKOFF_S)

    if args.no_ingest:
        return
    n = ingest_all(jobs, args.levels, args.resample)
    print(f"Ingested {n} new symbol-day parquet files (data/l2 + data/trades)", flush=True)


if __name__ == "__main__":
    main()
