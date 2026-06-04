"""
Schema + roundtrip tests for l2_store and trade_store.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.l2_store import L2Config, expected_columns as l2_expected_columns, validate_frame as l2_validate, write_day as l2_write, load_symbol_range as l2_load
from src.trade_store import TradeConfig, CANONICAL_COLUMNS as TRADE_COLUMNS, validate_frame as trade_validate, write_day as trade_write, load_symbol_range as trade_load
from tests.synthetic import make_synthetic_book, make_synthetic_trades


def test_l2_roundtrip():
    book = make_synthetic_book(n_rows=100)
    l2_validate(book)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = L2Config(data_dir=str(Path(tmp) / "l2"))
        day = book["timestamp"].iloc[0].floor("D")
        l2_write("BTCUSDT", book, day, cfg=cfg)
        back = l2_load("BTCUSDT", day, day + pd.Timedelta(days=1), cfg=cfg)
        assert len(back) == len(book), f"{len(back)} != {len(book)}"
        for c in l2_expected_columns(cfg):
            assert c in back.columns, c
    print("test_l2_roundtrip: PASS")


def test_trades_roundtrip():
    book = make_synthetic_book(n_rows=50)
    trades = make_synthetic_trades(book)
    if trades.empty:
        # Bump avg_trades to guarantee non-empty
        trades = make_synthetic_trades(book, avg_trades_per_bar=10.0)
    trade_validate(trades)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = TradeConfig(data_dir=str(Path(tmp) / "trades"))
        day = trades["timestamp"].iloc[0].floor("D")
        trade_write("BTCUSDT", trades, day, cfg=cfg)
        back = trade_load("BTCUSDT", day, day + pd.Timedelta(days=1), cfg=cfg)
        assert len(back) == len(trades)
        for c in TRADE_COLUMNS:
            assert c in back.columns
    print("test_trades_roundtrip: PASS")


def test_l2_validate_rejects_crossed_book():
    book = make_synthetic_book(n_rows=10)
    book.loc[5, "bid_px_1"] = book.loc[5, "ask_px_1"] + 1.0  # cross it
    try:
        l2_validate(book)
        raise AssertionError("Validation should have failed for crossed book")
    except ValueError as exc:
        assert "Crossed book" in str(exc)
    print("test_l2_validate_rejects_crossed_book: PASS")


if __name__ == "__main__":
    test_l2_roundtrip()
    test_trades_roundtrip()
    test_l2_validate_rejects_crossed_book()
    print("\nAll store tests PASSED.")
