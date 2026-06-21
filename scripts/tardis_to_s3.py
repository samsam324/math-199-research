"""Stream Binance-spot L2 from Tardis.dev straight to S3 as Hive-partitioned parquet,
CONCURRENTLY. For each (data_type, symbol, day): download the Tardis csv.gz, convert it
to parquet with DuckDB, upload to

    s3://<bucket>/<data_type>/symbol=<SYM>/year=<YYYY>/<YYYY-MM-DD>.parquet

then delete the temp files. Idempotent (skips days already in S3 via HEAD), so a crash or
Spot reclaim resumes cleanly. A thread pool overlaps the network-bound download/upload with
the CPU-bound convert; each worker has its own DuckDB connection, boto3 client, and asyncio
loop (tardis_dev is async). Local disk stays tiny (one file per in-flight worker).

Examples
  python scripts/tardis_to_s3.py --bucket B --symbols BTCUSDT ETHUSDT --dates 2023-06-15 --workers 8
  python scripts/tardis_to_s3.py --bucket B --symbols-file data/l2_universe_top50.txt \
      --from 2023-01-01 --to 2025-12-31 --workers 10 --storage-class STANDARD_IA
"""
from __future__ import annotations
import argparse
import asyncio
import os
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXCHANGE = "binance"
DATA_TYPES_DEFAULT = ["book_snapshot_25", "trades"]
MAX_RETRIES, BACKOFF = 4, 5.0
_local = threading.local()


def load_api_key() -> str:
    key = os.environ.get("TARDIS_API_KEY")
    if not key:
        env = REPO / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip().startswith("TARDIS_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        sys.exit("no TARDIS_API_KEY (env or .env)")
    return key


def s3_key(data_type, symbol, day):
    return f"{data_type}/symbol={symbol}/year={day[:4]}/{day}.parquet"


def fetch_csv(data_type, symbol, day, api_key, tmp: Path):
    from tardis_dev import download_datasets
    dest = tmp / f"{data_type}_{symbol}_{day}.csv.gz"
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            download_datasets(exchange=EXCHANGE, data_types=[data_type], symbols=[symbol],
                              from_date=day, to_date=nxt, api_key=api_key,
                              download_dir=str(tmp), get_filename=lambda *_a, _n=dest.name, **_k: _n)
            return dest if dest.exists() and dest.stat().st_size > 0 else None
        except Exception:
            if attempt == MAX_RETRIES:
                return None
            time.sleep(BACKOFF * attempt)
    return None


def _init_worker(region):
    asyncio.set_event_loop(asyncio.new_event_loop())   # tardis_dev is async; one loop per thread
    _local.con = duckdb.connect()
    _local.con.execute("PRAGMA threads=2")
    _local.s3 = boto3.client("s3", region_name=region)


def process_one(job, bucket, storage_class, api_key):
    dt, sym, day = job
    key = s3_key(dt, sym, day)
    try:
        _local.s3.head_object(Bucket=bucket, Key=key)
        return "skip"
    except Exception:
        pass
    tmpd = Path(tempfile.mkdtemp())
    try:
        csv = fetch_csv(dt, sym, day, api_key, tmpd)
        if csv is None:
            return "nodata"
        pq = tmpd / "out.parquet"
        try:
            _local.con.execute(
                f"COPY (SELECT * FROM read_csv_auto('{csv.as_posix()}', compression='gzip', "
                f"sample_size=-1)) TO '{pq.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
        except Exception:
            return "convfail"
        _local.s3.upload_file(str(pq), bucket, key, ExtraArgs={"StorageClass": storage_class})
        return "ok"
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--symbols-file")
    ap.add_argument("--data-types", nargs="*", default=DATA_TYPES_DEFAULT)
    ap.add_argument("--from", dest="d_from")
    ap.add_argument("--to", dest="d_to")
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--storage-class", default="STANDARD")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    if a.symbols_file:
        syms = [l.strip() for l in open(REPO / a.symbols_file) if l.strip()]
    elif a.symbols:
        syms = [s.upper() for s in a.symbols]
    else:
        sys.exit("need --symbols or --symbols-file")
    if a.dates:
        days = list(a.dates)
    elif a.d_from and a.d_to:
        days = [d.strftime("%Y-%m-%d") for d in pd.date_range(a.d_from, a.d_to, freq="D")]
    else:
        sys.exit("need --dates or --from/--to")

    api_key = load_api_key()
    # day-major job order: a contiguous window is queryable as soon as early days finish
    jobs = [(dt, sym, day) for day in days for dt in a.data_types for sym in syms]
    print(f"plan: {len(syms)} symbols x {len(days)} days x {len(a.data_types)} types = {len(jobs)} files "
          f"-> s3://{a.bucket}  ({a.workers} workers)", flush=True)

    counts, t0 = Counter(), time.time()
    with ThreadPoolExecutor(max_workers=a.workers, initializer=_init_worker, initargs=(a.region,)) as ex:
        futs = [ex.submit(process_one, j, a.bucket, a.storage_class, api_key) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                counts[fut.result()] += 1
            except Exception:
                counts["error"] += 1
            if k % 100 == 0:
                rate = k / max(1e-9, time.time() - t0)
                eta_h = (len(jobs) - k) / max(1e-9, rate) / 3600
                print(f"  {k}/{len(jobs)} | ok {counts['ok']} skip {counts['skip']} "
                      f"nodata {counts['nodata']} fail {counts['convfail']+counts['error']} "
                      f"| {rate:.1f} files/s ETA {eta_h:.1f}h", flush=True)
    print(f"DONE in {time.time()-t0:.0f}s: {dict(counts)}", flush=True)


if __name__ == "__main__":
    main()
