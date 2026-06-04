"""
Precompute hourly volume-as-information panels from the downloaded L2 + tick
trades, one parquet per symbol:

    data/microstructure/{SYMBOL}.parquet

Columns are the hourly information features (order_flow_imbalance, vpin,
kyle_lambda, quoted_spread_bps, trade_intensity, volume_quote, ...). These feed
`src.microstructure_features.merge_into_pair_features`, which joins each leg's
panel onto the pair feature store so the ML walk-forward can use them.

Caching it once here keeps `run_first_branch.py` cheap: the expensive 1s->1h
aggregation runs a single time per symbol, not per pair per run.

Usage
-----
  # all symbols that have trades parquet, over their full coverage
  python scripts/build_microstructure_panel.py

  # explicit symbols + window
  python scripts/build_microstructure_panel.py --symbols BTCUSDT ETHUSDT \
      --from 2024-01-01 --to 2024-04-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.microstructure_features import hourly_information_features  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "microstructure"
TRADES_DIR = REPO_ROOT / "data" / "trades"


def _coverage(symbol: str):
    sym_dir = TRADES_DIR / symbol.upper()
    days = sorted(p.stem for p in sym_dir.glob("*.parquet")) if sym_dir.exists() else []
    if not days:
        return None
    return pd.Timestamp(days[0], tz="UTC"), pd.Timestamp(days[-1], tz="UTC") + pd.Timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", help="default: every symbol with trades parquet")
    ap.add_argument("--from", dest="from_date", default=None)
    ap.add_argument("--to", dest="to_date", default=None)
    ap.add_argument("--levels", type=int, default=10)
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols]
    else:
        symbols = sorted(p.name for p in TRADES_DIR.iterdir() if p.is_dir()) if TRADES_DIR.exists() else []
    if not symbols:
        print("No symbols with trades data found. Run the downloader first.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        cov = _coverage(sym)
        if cov is None:
            print(f"  {sym}: no trades data, skipping")
            continue
        start = pd.Timestamp(args.from_date, tz="UTC") if args.from_date else cov[0]
        end = pd.Timestamp(args.to_date, tz="UTC") if args.to_date else cov[1]
        feat = hourly_information_features(sym, start, end, levels=args.levels)
        if feat.empty:
            print(f"  {sym}: no hourly features in window")
            continue
        out = feat.reset_index().rename(columns={"index": "timestamp"})
        out.to_parquet(OUT_DIR / f"{sym}.parquet", engine="pyarrow", index=False)
        print(f"  {sym}: {len(out):,} hourly rows -> {OUT_DIR / (sym + '.parquet')}")


if __name__ == "__main__":
    main()
