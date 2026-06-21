# 3-year Binance L2 on S3 — and how to analyze it

Three full years (2023–2025) of Binance-spot Level-2 data for the top-50 USDT universe,
pulled from Tardis.dev and stored as Hive-partitioned parquet so you can query any
symbol/date range with DuckDB without downloading the whole thing.

- **Bucket:** `s3://math199-l2-873750256216/` (region `us-west-2`, **private**)
- **Size:** ~1.1 TB (Standard-IA, ~$15/mo). Do not make public — Tardis-licensed.
- **Coverage:** 50 symbols × 2023-01-01…2025-12-31 × {`book_snapshot_25`, `trades`}.
  `nodata` gaps exist where a coin was not yet listed (e.g. FLOKI in early 2023).

## Layout

```
s3://math199-l2-873750256216/
  book_snapshot_25/symbol=BTCUSDT/year=2024/2024-03-13.parquet   # tick-level, top-25 book, 104 cols
  trades/symbol=BTCUSDT/year=2024/2024-03-13.parquet             # tick-level trades, 8 cols
```
One parquet per symbol-day; partitioned by `symbol` and `year` so a single-symbol query
prunes to just those files. Each file is the raw Tardis rows converted to parquet (lossless,
zstd) — `timestamp`/`local_timestamp` are int64 microseconds UTC.

## Query it (no download)

`scripts/l2_query.py` wires DuckDB to the bucket via the AWS credential chain:

```python
from scripts.l2_query import open_l2, load_day, query_range
con = open_l2()

book = load_day(con, "BTCUSDT", "2024-03-13")                  # one symbol-day, full book
tr   = load_day(con, "BTCUSDT", "2024-03-13", dataset="trades")
q1   = query_range(con, "ETHUSDT", "2023-06-01", "2023-06-30") # a date range, one symbol

# raw SQL with partition pruning + column projection (only scans BTC/2024 files):
con.execute('''
  SELECT date_trunc('day', to_timestamp(timestamp/1e6)) d,
         avg("asks[0].price" - "bids[0].price") avg_spread
  FROM read_parquet('s3://math199-l2-873750256216/book_snapshot_25/symbol=*/year=*/*.parquet',
                    hive_partitioning=true)
  WHERE symbol='BTCUSDT' AND year=2024
  GROUP BY 1 ORDER BY 1
''').df()
```
Tips: filter on the bare partition columns (`symbol`, `year`) so DuckDB prunes files;
select only the columns you need; for repeated work on one slice, `COPY ... TO 'local.parquet'`
once and iterate locally. Heavy multi-symbol scans are faster from an EC2 box in `us-west-2`.

## How it was pulled (reproducible)

`scripts/aws/launch_ec2_pull.sh` launches a self-terminating m7g.2xlarge that runs
`scripts/tardis_to_s3.py` (concurrent stream: Tardis csv.gz → DuckDB parquet → S3, idempotent).
Re-running the puller skips days already in S3. To extend (more years/symbols), adjust
`--from/--to`/`--symbols-file` and relaunch; the Tardis key lives in SSM `/tardis/api_key`.

## Sharing

Private bucket; mint per-file links for collaborators with `scripts/presign.py`
(`python scripts/presign.py book_snapshot_25/symbol=BTCUSDT/year=2024/2024-03-13.parquet`),
or grant a read-only IAM policy on the bucket.
