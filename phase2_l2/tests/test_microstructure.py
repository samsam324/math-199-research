"""
Microstructure feature correctness: microprice, OBI, quoted spread, trade
signing, size buckets, bar aggregation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.microstructure import (
    SizeBuckets, assign_size_bucket, build_signed_trade_frame, microprice,
    order_book_imbalance, quoted_spread_bps,
    sign_trades_quote_rule, sign_trades_tick_rule, sign_trades_use_aggressor,
)
from src.bars import BarConfig, build_bars
from tests.synthetic import make_synthetic_book, make_synthetic_trades


def test_microprice_reduces_to_mid_when_sizes_equal():
    book = pd.DataFrame({
        "bid_px_1": [100.0, 100.0],
        "bid_sz_1": [10.0, 10.0],
        "ask_px_1": [100.10, 100.10],
        "ask_sz_1": [10.0, 10.0],
    })
    mp = microprice(book)
    mid = (book["bid_px_1"] + book["ask_px_1"]) / 2.0
    assert np.allclose(mp.values, mid.values), f"{mp.values} != {mid.values}"
    print("test_microprice_reduces_to_mid_when_sizes_equal: PASS")


def test_microprice_skews_toward_thicker_side():
    book = pd.DataFrame({
        "bid_px_1": [100.0],  # bid is THIN
        "bid_sz_1": [1.0],
        "ask_px_1": [100.10], # ask is THICK -> microprice should pull DOWN toward bid (more demand on opposite side)
        "ask_sz_1": [100.0],
    })
    mp = float(microprice(book).iloc[0])
    mid = 100.05
    # microprice = (bs * ap + as * bp) / (bs + as) = (1 * 100.10 + 100 * 100.0) / 101 = 100.0010
    # Thicker ask -> microprice pulls toward bid (selling pressure dominates)
    expected = (1.0 * 100.10 + 100.0 * 100.0) / 101.0
    assert abs(mp - expected) < 1e-9, f"{mp} != {expected}"
    assert mp < mid, f"thicker ask should pull microprice below mid: {mp} >= {mid}"
    print("test_microprice_skews_toward_thicker_side: PASS")


def test_quoted_spread_bps_arithmetic():
    book = pd.DataFrame({
        "bid_px_1": [100.0],
        "ask_px_1": [100.10],
    })
    spread = float(quoted_spread_bps(book).iloc[0])
    # 0.10 / 100.05 * 1e4 = ~9.995 bps
    expected = 0.10 / 100.05 * 1e4
    assert abs(spread - expected) < 1e-6, f"{spread} != {expected}"
    print(f"test_quoted_spread_bps_arithmetic: PASS ({spread:.4f} bps)")


def test_obi_bounds():
    book = pd.DataFrame({
        **{f"bid_sz_{i}": [10.0] for i in range(1, 6)},
        **{f"ask_sz_{i}": [10.0] for i in range(1, 6)},
    })
    obi = float(order_book_imbalance(book, depth_levels=5).iloc[0])
    assert abs(obi) < 1e-9, f"Equal sizes should give OBI=0, got {obi}"

    book2 = pd.DataFrame({
        **{f"bid_sz_{i}": [10.0] for i in range(1, 6)},
        **{f"ask_sz_{i}": [0.0] for i in range(1, 6)},
    })
    obi2 = float(order_book_imbalance(book2, depth_levels=5).iloc[0])
    assert abs(obi2 - 1.0) < 1e-9, f"All bid side -> OBI=+1, got {obi2}"
    print("test_obi_bounds: PASS")


def test_size_bucket_assignment():
    notional = pd.Series([100.0, 500.0, 5_000.0, 50_000.0, 500_000.0])
    buckets = assign_size_bucket(notional)
    assert list(buckets.astype(str)) == ["retail", "retail", "mid", "large", "institutional"]
    print("test_size_bucket_assignment: PASS")


def test_tick_rule_signing():
    trades = pd.DataFrame({"price": [100.0, 100.1, 100.0, 100.0, 99.9, 99.9, 100.1]})
    sign = sign_trades_tick_rule(trades).to_numpy()
    # 0: 0 (no prev); 1: +1 (up); 2: -1 (down); 3: -1 (carry); 4: -1 (down);
    # 5: -1 (carry); 6: +1 (up)
    expected = np.array([0, 1, -1, -1, -1, -1, 1], dtype=np.int8)
    assert np.array_equal(sign, expected), f"{sign} != {expected}"
    print("test_tick_rule_signing: PASS")


def test_aggressor_signing_uses_side_when_present():
    trades = pd.DataFrame({
        "side": pd.Categorical(["buy", "sell", "buy", ""], categories=["buy", "sell", ""]),
        "price": [100.0, 100.0, 100.0, 100.0],
    })
    sign = sign_trades_use_aggressor(trades).to_numpy()
    expected = np.array([1, -1, 1, 0], dtype=np.int8)
    assert np.array_equal(sign, expected), f"{sign} != {expected}"
    print("test_aggressor_signing_uses_side_when_present: PASS")


def test_quote_rule_signing():
    book = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="1s", tz="UTC"),
        "bid_px_1": [100.0, 100.0, 100.0],
        "ask_px_1": [100.10, 100.10, 100.10],
    })
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 00:00:00.5", periods=3, freq="1s", tz="UTC"),
        "price": [100.10, 100.0, 100.05],  # at ask, at bid, at mid
    })
    sign = sign_trades_quote_rule(trades, book).to_numpy()
    # at ask = +1 (buyer initiated), at bid = -1, at mid -> tick rule fallback (0 here for first)
    assert sign[0] == 1 and sign[1] == -1
    print("test_quote_rule_signing: PASS")


def test_build_signed_trade_frame_uses_aggressor_first():
    book = make_synthetic_book(n_rows=20)
    trades = make_synthetic_trades(book, avg_trades_per_bar=5)
    out = build_signed_trade_frame(trades, book=book)
    # All synthetic trades have side set, so trade_sign should match
    assert (out["trade_sign"] != 0).all(), "All synthetic trades have explicit side; sign must be nonzero"
    # signed_notional sign matches trade_sign sign
    sn = out["signed_notional"].to_numpy()
    ts = out["trade_sign"].to_numpy()
    assert np.all(np.sign(sn) == ts), "signed_notional sign mismatch"
    print("test_build_signed_trade_frame_uses_aggressor_first: PASS")


def test_bars_no_lookahead():
    """
    Inject a giant trade at bar index 200. Bar features at index 100 must be
    identical to a clean run.
    """
    book = make_synthetic_book(n_rows=400, seed=7)
    trades = make_synthetic_trades(book, seed=7, avg_trades_per_bar=3)

    def bars_of(spike: bool):
        b = book.copy(); t = trades.copy()
        if spike:
            # Add a $10M buy at bar 200's timestamp
            ts_200 = b["timestamp"].iloc[200]
            spike_row = pd.DataFrame([{
                "timestamp": ts_200, "local_timestamp": ts_200,
                "trade_id": "spike", "side": "buy",
                "price": float(b["ask_px_1"].iloc[200]),
                "amount": 1.0, "notional": 10_000_000.0,
            }])
            spike_row["side"] = pd.Categorical(spike_row["side"], categories=["buy", "sell", ""])
            t = pd.concat([t, spike_row], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        return build_bars(b, t, cfg=BarConfig(bar_size="1s"))

    clean = bars_of(False)
    spiked = bars_of(True)
    # Bars at index 100 (long before spike at 200) must be identical
    cutoff_ts = book["timestamp"].iloc[200]
    pre_clean = clean[clean.index < cutoff_ts]
    pre_spiked = spiked[spiked.index < cutoff_ts]
    common = pre_clean.index.intersection(pre_spiked.index)
    assert len(common) > 50
    # Check the last common timestamp (most stringent)
    last = common[-1]
    for col in clean.columns:
        v1 = clean.loc[last, col]; v2 = spiked.loc[last, col]
        if pd.isna(v1) and pd.isna(v2):
            continue
        assert abs(float(v1) - float(v2)) < 1e-9, f"LEAK: bar feature '{col}' at {last} differs ({v1} vs {v2})"
    print(f"test_bars_no_lookahead: PASS ({len(clean.columns)} columns verified at {last})")


if __name__ == "__main__":
    test_microprice_reduces_to_mid_when_sizes_equal()
    test_microprice_skews_toward_thicker_side()
    test_quoted_spread_bps_arithmetic()
    test_obi_bounds()
    test_size_bucket_assignment()
    test_tick_rule_signing()
    test_aggressor_signing_uses_side_when_present()
    test_quote_rule_signing()
    test_build_signed_trade_frame_uses_aggressor_first()
    test_bars_no_lookahead()
    print("\nAll microstructure tests PASSED.")
