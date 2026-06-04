"""
Tardis CSV -> canonical schema parsing tests. Uses synthetic CSV strings in
the documented Tardis layout; round-trips through the parsing seam.

When the real `data.py` reference arrives, replace these synthetic CSV
strings with sample bytes from a real Tardis pull and the parser only
breaks at this one boundary.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tardis_ingest import (
    _tardis_book_snapshot_25_to_canonical,
    _tardis_trades_to_canonical,
)


def test_book_snapshot_25_csv_parses():
    # Two rows, 3 bid levels and 3 ask levels populated; remaining levels NaN
    rows = []
    base_ts_us = 1_704_067_200_000_000  # 2024-01-01 00:00:00 UTC in microseconds
    for i in range(2):
        row = {
            "exchange": "binance-futures",
            "symbol": "BTCUSDT",
            "timestamp": base_ts_us + i * 1_000_000,
            "local_timestamp": base_ts_us + i * 1_000_000 + 5_000,
        }
        # Levels 0..24 expected; we populate 0..2 with real values, rest with 0
        for level in range(25):
            row[f"asks[{level}].price"] = 100.0 + level * 0.1 if level < 3 else 0.0
            row[f"asks[{level}].amount"] = 1.0 + level if level < 3 else 0.0
            row[f"bids[{level}].price"] = 99.9 - level * 0.1 if level < 3 else 0.0
            row[f"bids[{level}].amount"] = 1.0 + level if level < 3 else 0.0
        rows.append(row)
    raw = pd.DataFrame(rows)
    canon = _tardis_book_snapshot_25_to_canonical(raw, levels=10)
    assert len(canon) == 2
    assert canon["bid_px_1"].iloc[0] == 99.9
    assert canon["ask_px_1"].iloc[0] == 100.0
    assert canon["bid_px_3"].iloc[0] == 99.7
    assert canon["ask_px_3"].iloc[0] == 100.2
    # Timestamp roundtrip
    assert canon["timestamp"].dt.tz is not None
    assert canon["timestamp"].iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert canon["is_snapshot"].all()
    assert (~canon["is_delta"]).all()
    print("test_book_snapshot_25_csv_parses: PASS")


def test_trades_csv_parses():
    base_ts_us = 1_704_067_200_000_000
    raw = pd.DataFrame([
        {"exchange": "binance-futures", "symbol": "BTCUSDT", "timestamp": base_ts_us, "local_timestamp": base_ts_us + 1000, "id": "1", "side": "buy", "price": 100.0, "amount": 1.5},
        {"exchange": "binance-futures", "symbol": "BTCUSDT", "timestamp": base_ts_us + 500_000, "local_timestamp": base_ts_us + 501_000, "id": "2", "side": "sell", "price": 99.95, "amount": 0.8},
    ])
    canon = _tardis_trades_to_canonical(raw)
    assert len(canon) == 2
    assert canon["timestamp"].iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert canon["price"].iloc[0] == 100.0
    assert canon["amount"].iloc[1] == 0.8
    assert abs(canon["notional"].iloc[0] - 150.0) < 1e-9
    assert canon["side"].iloc[0] == "buy"
    print("test_trades_csv_parses: PASS")


def test_trades_canonical_validates_via_trade_store():
    from src.trade_store import validate_frame
    base_ts_us = 1_704_067_200_000_000
    raw = pd.DataFrame([
        {"exchange": "binance-futures", "symbol": "BTCUSDT", "timestamp": base_ts_us, "local_timestamp": base_ts_us + 1000, "id": "1", "side": "buy", "price": 100.0, "amount": 1.5},
    ])
    canon = _tardis_trades_to_canonical(raw)
    validate_frame(canon)
    print("test_trades_canonical_validates_via_trade_store: PASS")


if __name__ == "__main__":
    test_book_snapshot_25_csv_parses()
    test_trades_csv_parses()
    test_trades_canonical_validates_via_trade_store()
    print("\nAll ingest parsing tests PASSED.")
