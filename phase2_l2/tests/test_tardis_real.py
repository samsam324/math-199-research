"""
Real-data integration test against Tardis.

Hits the free tier (no API key required) and verifies:
  1. download_day_raw returns valid bytes for a known free-tier file
  2. _read_gz_csv autodetects gzip and parses
  3. Canonical parsers produce frames that validate against l2_store / trade_store
  4. End-to-end ingest_book_day + ingest_trades_day round-trip through parquet
  5. Idempotency: re-running ingest_book_day skips redundant work
  6. ingest_range concurrent path works for a tiny grid

Skipped if no network (e.g. CI without egress). Set environment variable
PHASE2_SKIP_NETWORK_TESTS=1 to skip explicitly.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tardis_ingest import (
    TardisConfig, canonical_l2_path, canonical_trades_path,
    download_day_raw, ingest_book_day, ingest_range, ingest_trades_day,
    _read_gz_csv,
    _tardis_book_snapshot_25_to_canonical, _tardis_trades_to_canonical,
)
from src.l2_store import L2Config, validate_frame as l2_validate, load_symbol_range as load_l2
from src.trade_store import TradeConfig, validate_frame as t_validate, load_symbol_range as load_trades


# Tardis free tier: first day of any month, no auth required.
FREE_TIER_DAY = pd.Timestamp("2024-04-01", tz="UTC")
FREE_TIER_EXCHANGE = "binance-futures"
FREE_TIER_SYMBOL = "BTCUSDT"


def _network_available() -> bool:
    if os.environ.get("PHASE2_SKIP_NETWORK_TESTS"):
        return False
    try:
        import requests
        # Quick HEAD-style ping; Tardis serves their docs site
        r = requests.head("https://datasets.tardis.dev", timeout=5)
        return r.status_code < 500
    except Exception:
        return False


def _skip_if_no_network() -> bool:
    if not _network_available():
        print("SKIP: no network or PHASE2_SKIP_NETWORK_TESTS=1")
        return True
    return False


def test_download_day_raw_book_freetier():
    if _skip_if_no_network():
        return
    blob = download_day_raw(FREE_TIER_EXCHANGE, "book_snapshot_25", FREE_TIER_DAY, FREE_TIER_SYMBOL, cfg=TardisConfig())
    assert blob is not None and len(blob) > 1_000_000, f"expected a sizeable blob, got {len(blob) if blob else 0}"
    # First two bytes should be the gzip magic
    assert blob[0] == 0x1F and blob[1] == 0x8B, "expected gzip magic bytes"
    print(f"test_download_day_raw_book_freetier: PASS ({len(blob)/1e6:.1f} MB)")


def test_read_gz_csv_autodetect_gzip():
    """Round-trip a tiny CSV both gzipped and plain through _read_gz_csv."""
    import gzip, io

    csv_text = "a,b,c\n1,2,3\n4,5,6\n"
    plain_blob = csv_text.encode("utf-8")
    gz_buf = io.BytesIO()
    with gzip.GzipFile(fileobj=gz_buf, mode="wb") as gz:
        gz.write(plain_blob)
    gz_blob = gz_buf.getvalue()

    df_plain = _read_gz_csv(plain_blob)
    df_gz = _read_gz_csv(gz_blob)
    assert df_plain.equals(df_gz)
    assert list(df_plain.columns) == ["a", "b", "c"]
    assert len(df_plain) == 2
    print("test_read_gz_csv_autodetect_gzip: PASS")


def test_book_parser_on_real_sample():
    if _skip_if_no_network():
        return
    blob = download_day_raw(FREE_TIER_EXCHANGE, "book_snapshot_25", FREE_TIER_DAY, FREE_TIER_SYMBOL, cfg=TardisConfig())
    raw = _read_gz_csv(blob)
    # Sample just the first 5000 rows to keep the test fast
    raw_small = raw.head(5000).copy()
    canon = _tardis_book_snapshot_25_to_canonical(raw_small, levels=10)
    l2_validate(canon, L2Config(levels=10))
    # Sanity: bid_px_1 < ask_px_1 on every row
    crossed = (canon["bid_px_1"] >= canon["ask_px_1"]).sum()
    assert crossed == 0, f"{crossed} crossed rows in real book data"
    # Sanity: timestamps monotone non-decreasing
    assert canon["timestamp"].is_monotonic_increasing
    print(f"test_book_parser_on_real_sample: PASS ({len(canon)} rows validated)")


def test_trades_parser_on_real_sample():
    if _skip_if_no_network():
        return
    blob = download_day_raw(FREE_TIER_EXCHANGE, "trades", FREE_TIER_DAY, FREE_TIER_SYMBOL, cfg=TardisConfig())
    raw = _read_gz_csv(blob)
    raw_small = raw.head(10_000).copy()
    canon = _tardis_trades_to_canonical(raw_small)
    t_validate(canon)
    # Sanity: notional == price * amount
    expected_notional = canon["price"].astype(float) * canon["amount"].astype(float)
    assert (canon["notional"].sub(expected_notional).abs() < 1e-6).all()
    # Sanity: side is in {buy, sell}
    sides = set(canon["side"].astype(str).unique())
    assert sides.issubset({"buy", "sell", ""}), sides
    print(f"test_trades_parser_on_real_sample: PASS ({len(canon)} rows validated)")


def test_ingest_book_day_idempotent():
    if _skip_if_no_network():
        return
    with tempfile.TemporaryDirectory() as tmp:
        cfg_l2 = L2Config(levels=10, data_dir=str(Path(tmp) / "l2"))
        cfg_t = TardisConfig()
        t0 = time.perf_counter()
        p1 = ingest_book_day(FREE_TIER_EXCHANGE, FREE_TIER_DAY, FREE_TIER_SYMBOL, cfg_t, cfg_l2)
        dt_first = time.perf_counter() - t0
        assert p1 is not None and p1.exists()
        # Second call should skip — must be much faster (no download)
        t1 = time.perf_counter()
        p2 = ingest_book_day(FREE_TIER_EXCHANGE, FREE_TIER_DAY, FREE_TIER_SYMBOL, cfg_t, cfg_l2)
        dt_second = time.perf_counter() - t1
        assert p2 == p1
        # Idempotent path is just a file-exists check; should be sub-second.
        assert dt_second < max(0.5, dt_first / 20), f"second call too slow: {dt_second:.2f}s vs first {dt_first:.2f}s"
        # canonical_l2_path agrees
        assert canonical_l2_path(FREE_TIER_SYMBOL, FREE_TIER_DAY, cfg_l2) == p1
    print(f"test_ingest_book_day_idempotent: PASS (first {dt_first:.1f}s, second {dt_second:.3f}s)")


def test_ingest_range_concurrent_smoke():
    """
    Tiny smoke test: pull trades only (smaller files) for one symbol-day grid,
    verify ingest_range runs concurrent path without errors and the summary is
    consistent. Uses only the trades data_type to keep total bytes modest.
    """
    if _skip_if_no_network():
        return
    with tempfile.TemporaryDirectory() as tmp:
        cfg_l2 = L2Config(levels=10, data_dir=str(Path(tmp) / "l2"))
        cfg_tr = TradeConfig(data_dir=str(Path(tmp) / "trades"))
        cfg_t = TardisConfig()
        # Just one symbol, one day so this stays small and fast.
        summary = ingest_range(
            exchange=FREE_TIER_EXCHANGE, symbols=[FREE_TIER_SYMBOL],
            start_day=FREE_TIER_DAY, end_day_exclusive=FREE_TIER_DAY + pd.Timedelta(days=1),
            data_types=["trades"],
            cfg_tardis=cfg_t, cfg_l2=cfg_l2, cfg_trades=cfg_tr,
            max_concurrency=2, show_progress=False,
        )
        assert summary["requested"] == 1
        assert summary["written"] == 1, summary
        assert len(summary["errors"]) == 0, summary["errors"]
        # Second run is a pure skip-existing path
        summary2 = ingest_range(
            exchange=FREE_TIER_EXCHANGE, symbols=[FREE_TIER_SYMBOL],
            start_day=FREE_TIER_DAY, end_day_exclusive=FREE_TIER_DAY + pd.Timedelta(days=1),
            data_types=["trades"],
            cfg_tardis=cfg_t, cfg_l2=cfg_l2, cfg_trades=cfg_tr,
            max_concurrency=2, show_progress=False,
        )
        assert summary2["skipped_existing"] == 1
        assert summary2["written"] == 0
    print("test_ingest_range_concurrent_smoke: PASS")


if __name__ == "__main__":
    test_download_day_raw_book_freetier()
    test_read_gz_csv_autodetect_gzip()
    test_book_parser_on_real_sample()
    test_trades_parser_on_real_sample()
    test_ingest_book_day_idempotent()
    test_ingest_range_concurrent_smoke()
    print("\nAll Tardis real-data tests PASSED.")
