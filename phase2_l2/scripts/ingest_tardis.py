"""
CLI: pull a range of (exchange, symbols, days, data_types) from Tardis and
write canonical parquet under data/.

Requires TARDIS_API_KEY env var (or --api-key flag). For the free tier (first
day of each month), the key can be omitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tardis_ingest import TardisConfig, ingest_range
from src.l2_store import L2Config
from src.trade_store import TradeConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Tardis daily CSVs into canonical parquet.")
    p.add_argument("--exchange", default="binance-futures", help="Tardis exchange code (e.g. binance-futures, binance, deribit)")
    p.add_argument("--symbols", nargs="+", required=True, help="Symbols like BTCUSDT ETHUSDT")
    p.add_argument("--start-day", required=True, help="UTC date YYYY-MM-DD")
    p.add_argument("--end-day-exclusive", required=True, help="UTC date YYYY-MM-DD")
    p.add_argument("--data-types", nargs="+", default=["book_snapshot_25", "trades"])
    p.add_argument("--api-key", default=None, help="Tardis API key (otherwise reads $TARDIS_API_KEY).")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--levels", type=int, default=10)
    p.add_argument("--max-concurrency", type=int, default=4, help="Concurrent downloads. 1 = sequential. Bump on a cluster.")
    p.add_argument("--force", action="store_true", help="Re-download even if canonical parquet already exists.")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    p.add_argument("--out-summary", default="data/metadata/ingest_summary.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start_day, tz="UTC")
    end = pd.Timestamp(args.end_day_exclusive, tz="UTC")

    cfg_tardis = TardisConfig(api_key=args.api_key) if args.api_key else TardisConfig()
    cfg_l2 = L2Config(levels=args.levels, data_dir=str(Path(args.data_dir) / "l2"))
    cfg_trades = TradeConfig(data_dir=str(Path(args.data_dir) / "trades"))

    summary = ingest_range(
        exchange=args.exchange, symbols=args.symbols,
        start_day=start, end_day_exclusive=end,
        data_types=args.data_types,
        cfg_tardis=cfg_tardis, cfg_l2=cfg_l2, cfg_trades=cfg_trades,
        max_concurrency=args.max_concurrency,
        skip_existing=not args.force,
        show_progress=not args.no_progress,
    )

    out_path = Path(args.out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out = {
        "exchange": args.exchange,
        "symbols": args.symbols,
        "start_day": args.start_day,
        "end_day_exclusive": args.end_day_exclusive,
        "data_types": args.data_types,
        **summary,
    }
    # errors list can be long; truncate
    summary_out["errors"] = summary_out.get("errors", [])[:200]
    out_path.write_text(json.dumps(summary_out, indent=2, default=str), encoding="utf-8")

    print(f"Requested:        {summary['requested']}")
    print(f"Skipped existing: {summary.get('skipped_existing', 0)}")
    print(f"Written:          {summary['written']}")
    print(f"Missing:          {summary['missing']}")
    print(f"Errors:           {len(summary['errors'])}")
    print(f"Summary written to {out_path}")


if __name__ == "__main__":
    main()
