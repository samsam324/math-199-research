"""Easy analysis over the 3-year L2 parquet on S3, via DuckDB partition pruning.

Layout queried:  s3://<bucket>/<dataset>/symbol=<SYM>/year=<YYYY>/<YYYY-MM-DD>.parquet
where <dataset> is 'book_snapshot_25' or 'trades'.

    from scripts.l2_query import open_l2, load_day, query_range
    con = open_l2()                                   # wired to S3 via the AWS cred chain
    df  = load_day(con, "BTCUSDT", "2024-03-13")      # one symbol-day of the book
    tr  = load_day(con, "BTCUSDT", "2024-03-13", dataset="trades")
    rng = query_range(con, "BTCUSDT", "2024-01-01", "2024-03-31",
                      columns='timestamp, "bids[0].price", "asks[0].price"')

CLI:
    python scripts/l2_query.py --symbol BTCUSDT --date 2024-03-13           # peek a day
    python scripts/l2_query.py --sql "SELECT count(*) FROM read_parquet('s3://.../*.parquet')"
"""
from __future__ import annotations
import argparse
import duckdb

BUCKET = "math199-l2-873750256216"
REGION = "us-west-2"


def open_l2(bucket: str = BUCKET, region: str = REGION) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    # no keys in code: walk the standard AWS credential chain (env, ~/.aws, instance role)
    con.execute(f"""
        CREATE OR REPLACE SECRET s3_l2 (
            TYPE s3, PROVIDER credential_chain, REGION '{region}', SCOPE 's3://{bucket}'
        );
    """)
    return con


def _path(bucket, dataset, symbol, year):
    return f"s3://{bucket}/{dataset}/symbol={symbol}/year={year}/*.parquet"


def load_day(con, symbol, date, dataset="book_snapshot_25", columns="*", bucket=BUCKET):
    """One symbol-day. Reads exactly one parquet file."""
    key = f"s3://{bucket}/{dataset}/symbol={symbol}/year={date[:4]}/{date}.parquet"
    return con.execute(f"SELECT {columns} FROM read_parquet('{key}')").df()


def query_range(con, symbol, start, end, dataset="book_snapshot_25", columns="*", bucket=BUCKET):
    """One symbol over [start, end]. Globs only that symbol's year folders (prunes to
    one of ~50 symbols x the spanned years), then filters by the microsecond timestamp."""
    years = range(int(start[:4]), int(end[:4]) + 1)
    globs = ",".join(f"'{_path(bucket, dataset, symbol, y)}'" for y in years)
    lo = int(__import__("pandas").Timestamp(start).value // 1000)        # us since epoch
    hi = int(__import__("pandas").Timestamp(end).value // 1000) + 86_400_000_000
    return con.execute(
        f"SELECT {columns} FROM read_parquet([{globs}]) "
        f"WHERE timestamp >= {lo} AND timestamp < {hi} ORDER BY timestamp").df()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol")
    ap.add_argument("--date")
    ap.add_argument("--dataset", default="book_snapshot_25")
    ap.add_argument("--sql")
    a = ap.parse_args()
    con = open_l2()
    if a.sql:
        print(con.execute(a.sql).df().to_string())
    elif a.symbol and a.date:
        df = load_day(con, a.symbol, a.date, a.dataset)
        print(f"{a.symbol} {a.date} {a.dataset}: {len(df):,} rows, {df.shape[1]} cols")
        print(df.head(3).to_string())
    else:
        ap.error("give --symbol+--date, or --sql")


if __name__ == "__main__":
    main()
