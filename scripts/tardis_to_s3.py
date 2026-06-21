"""Stream Binance-spot L2 from Tardis.dev straight to S3 as Hive-partitioned parquet.

For each (data_type, symbol, day): download the Tardis csv.gz to a temp dir, convert
it to parquet with DuckDB (fast, multi-threaded), upload to

    s3://<bucket>/<data_type>/symbol=<SYM>/year=<YYYY>/<YYYY-MM-DD>.parquet

then delete the temp files. Idempotent (skips days already in S3 via HEAD), so a crash
or Spot reclaim resumes cleanly. Local disk stays tiny (one symbol-day at a time), which
is what lets the same script run on a laptop or a 30 GB EC2 box.

Examples
  # validate: two paid mid-month days (proves 2023+2025 plan coverage), BTC+ETH
  python scripts/tardis_to_s3.py --bucket math199-l2-873750256216 \
      --symbols BTCUSDT ETHUSDT --dates 2023-06-15 2025-06-15

  # full 3-year pull, top-50 universe (run on EC2)
  python scripts/tardis_to_s3.py --bucket math199-l2-873750256216 \
      --symbols-file data/l2_universe_top50.txt --from 2023-01-01 --to 2025-12-31
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import tempfile
from pathlib import Path

import boto3
import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXCHANGE = "binance"
DATA_TYPES_DEFAULT = ["book_snapshot_25", "trades"]
MAX_RETRIES, BACKOFF = 4, 5.0


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


def s3_key(data_type: str, symbol: str, day: str) -> str:
    return f"{data_type}/symbol={symbol}/year={day[:4]}/{day}.parquet"


def exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def fetch_csv(data_type: str, symbol: str, day: str, api_key: str, tmp: Path) -> Path | None:
    """Download one Tardis symbol-day csv.gz to tmp. Returns path, or None if no data."""
    from tardis_dev import download_datasets
    dest = tmp / f"{data_type}_{symbol}_{day}.csv.gz"
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            download_datasets(
                exchange=EXCHANGE, data_types=[data_type], symbols=[symbol],
                from_date=day, to_date=nxt, api_key=api_key,
                download_dir=str(tmp), get_filename=lambda *_a, _n=dest.name, **_k: _n)
            return dest if dest.exists() and dest.stat().st_size > 0 else None
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"  FAIL {data_type}/{symbol}/{day}: {e}", flush=True)
                return None
            time.sleep(BACKOFF * attempt)
    return None


def to_parquet(con, csv_gz: Path, pq: Path) -> bool:
    try:
        con.execute(
            f"COPY (SELECT * FROM read_csv_auto('{csv_gz.as_posix()}', "
            f"compression='gzip', sample_size=-1)) "
            f"TO '{pq.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
        return pq.exists() and pq.stat().st_size > 0
    except Exception as e:
        print(f"  CONVERT FAIL {csv_gz.name}: {e}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--symbols-file")
    ap.add_argument("--data-types", nargs="*", default=DATA_TYPES_DEFAULT)
    ap.add_argument("--from", dest="d_from")
    ap.add_argument("--to", dest="d_to")
    ap.add_argument("--dates", nargs="*", help="explicit days instead of a --from/--to range")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--storage-class", default="STANDARD",
                    help="STANDARD (instant, ~$18/mo per TB) or STANDARD_IA (cheaper store, small retrieval fee)")
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
    s3 = boto3.client("s3", region_name=a.region)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    total = len(days) * len(a.data_types) * len(syms)
    print(f"plan: {len(syms)} symbols x {len(days)} days x {len(a.data_types)} types = {total} files "
          f"-> s3://{a.bucket}", flush=True)
    done = skipped = uploaded = nodata = 0
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # day-major so a contiguous window is queryable as soon as early days land
        for day in days:
            for dt in a.data_types:
                for sym in syms:
                    done += 1
                    key = s3_key(dt, sym, day)
                    if exists(s3, a.bucket, key):
                        skipped += 1
                        continue
                    csv_gz = fetch_csv(dt, sym, day, api_key, tmp)
                    if csv_gz is None:
                        nodata += 1
                        continue
                    pq = tmp / f"{dt}_{sym}_{day}.parquet"
                    if to_parquet(con, csv_gz, pq):
                        s3.upload_file(str(pq), a.bucket, key,
                                       ExtraArgs={"StorageClass": a.storage_class})
                        uploaded += 1
                        pq.unlink(missing_ok=True)
                    csv_gz.unlink(missing_ok=True)
                    if uploaded and uploaded % 25 == 0:
                        rate = uploaded / max(1e-9, time.time() - t0)
                        print(f"  {done}/{total} | uploaded {uploaded} skip {skipped} nodata {nodata} "
                              f"| {rate:.2f} files/s", flush=True)
    print(f"DONE: {done} processed, {uploaded} uploaded, {skipped} already present, "
          f"{nodata} no-data, in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
