"""
Backtester sanity tests adapted for the L2-aware backtester.

6 hand-computable cases:
  1. Flat signal -> zero PnL, zero trades, zero cost.
  2. State machine opens + closes exactly once on a synthetic z trajectory.
  3. Walk-the-book fill arithmetic: VWAP across two levels of known size.
  4. One-bar execution lag: signal flip at t doesn't capture the t-bar return.
  5. Fixed capital base Sharpe arithmetic.
  6. Flat fallback when no book columns are present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import BacktestConfig, run_backtest, walk_book_fill


def _make_preds(n: int, pair: str, pred_class: np.ndarray, current_spread: np.ndarray, mid_a, mid_b, spread_z=None) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=n, freq="1s", tz="UTC")
    df = pd.DataFrame({
        "sample_id": np.arange(n), "pair": pair, "timestamp": ts,
        "current_spread": current_spread,
        "y_class": pred_class.astype(int), "pred_class": pred_class.astype(int),
        "model": "test",
        "mid_a": mid_a if hasattr(mid_a, "__len__") else np.full(n, mid_a),
        "mid_b": mid_b if hasattr(mid_b, "__len__") else np.full(n, mid_b),
    })
    if spread_z is not None:
        df["latest_spread_z"] = spread_z
    return df


def _make_pairs(beta: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame([{"sym_a": "AAA", "sym_b": "BBB", "beta_a_on_b": beta, "pair": "AAA_BBB"}])


def test_flat_signal_zero_pnl():
    n = 100
    pred = np.ones(n, dtype=int)
    preds = _make_preds(n, "AAA_BBB", pred, np.full(n, 0.01), 100.0, 100.0)
    result = run_backtest(preds, _make_pairs(), BacktestConfig(walk_book=False))
    assert abs(result.metrics["total_pnl_dollars"]) < 1e-9
    assert result.metrics["trades"] == 0
    print("test_flat_signal_zero_pnl: PASS")


def test_state_machine_opens_and_closes_once():
    n = 200
    spread = np.zeros(n); spread[20:80] = 0.05
    spread_z = np.zeros(n); spread_z[20:80] = 3.0
    pred = np.zeros(n, dtype=int)
    preds = _make_preds(n, "AAA_BBB", pred, spread, 100.0, 100.0, spread_z=spread_z)
    cfg = BacktestConfig(walk_book=False, use_state_machine=True, entry_z=2.0, exit_z=0.5, taker_fee_bps=0, fallback_taker_fee_bps=0, fallback_slippage_bps=0)
    result = run_backtest(preds, _make_pairs(), cfg)
    assert result.metrics["trades"] == 2, f"Expected 2 transitions, got {result.metrics['trades']}"
    print("test_state_machine_opens_and_closes_once: PASS")


def test_walk_book_fill_vwap_arithmetic():
    """
    Walk 2 levels on the ask side: $1000 fills entirely at L1 ($150 worth) +
    $850 at L2.
    L1: ask_px_1 = 100, ask_sz_1 = 1.5 -> $150
    L2: ask_px_2 = 101, ask_sz_2 = 100 -> plenty
    Need $1000 total: $150 at 100 + $850 at 101
    Base bought = 1.5 + 850/101 = 1.5 + 8.4158... = 9.9158
    VWAP = 1000 / 9.9158 = 100.8487
    """
    book_row = pd.Series({
        "ask_px_1": 100.0, "ask_sz_1": 1.5,
        "ask_px_2": 101.0, "ask_sz_2": 100.0,
        **{f"ask_px_{i}": np.nan for i in range(3, 11)},
        **{f"ask_sz_{i}": np.nan for i in range(3, 11)},
        **{f"bid_px_{i}": np.nan for i in range(1, 11)},
        **{f"bid_sz_{i}": np.nan for i in range(1, 11)},
    })
    vwap, filled, levels = walk_book_fill(1000.0, "buy", book_row)
    expected_base = 1.5 + 850.0 / 101.0
    expected_vwap = 1000.0 / expected_base
    assert abs(vwap - expected_vwap) < 1e-6, f"{vwap} != {expected_vwap}"
    assert abs(filled - 1000.0) < 1e-6
    assert levels == 2
    print(f"test_walk_book_fill_vwap_arithmetic: PASS (VWAP {vwap:.4f}, expected {expected_vwap:.4f})")


def test_one_bar_execution_lag():
    n = 100
    # Constant price except a +1% jump in A at bar 50
    mid_a = np.full(n, 100.0); mid_a[50:] = 101.0
    mid_b = np.full(n, 100.0)
    pred = np.ones(n, dtype=int); pred[50:] = 2  # diverge from bar 50 onward
    spread = np.full(n, 0.01)
    preds = _make_preds(n, "AAA_BBB", pred, spread, mid_a, mid_b)
    # beta=0 so leg B has no PnL contribution; we measure leg A only
    cfg = BacktestConfig(walk_book=False, taker_fee_bps=0, fallback_taker_fee_bps=0, fallback_slippage_bps=0)
    result = run_backtest(preds, _make_pairs(beta=0.0), cfg)
    # Bar 50 return is +1% on A. Position at bar 50 was set by signal at bar 49 (which was flat).
    # So pnl[50] should be 0. Position takes effect at bar 51 -- no more return there.
    total = result.metrics["total_pnl_dollars"]
    assert abs(total) < 1e-6, f"LEAK: expected $0 (one-bar lag), got ${total:.4f}"
    print("test_one_bar_execution_lag: PASS")


def test_sharpe_annualization_uses_bars_per_year():
    rng = np.random.default_rng(11)
    n = 5000
    mid_a_rets = rng.normal(0.0001, 0.001, n)
    mid_a = 100.0 * np.exp(np.cumsum(mid_a_rets))
    mid_b = np.full(n, 100.0)
    pred = np.full(n, 2, dtype=int)
    spread = np.full(n, 0.01)
    preds = _make_preds(n, "AAA_BBB", pred, spread, mid_a, mid_b)
    cfg = BacktestConfig(walk_book=False, taker_fee_bps=0, fallback_taker_fee_bps=0, fallback_slippage_bps=0, bars_per_year=24*365*60*60)
    result = run_backtest(preds, _make_pairs(beta=0.0), cfg)
    # Expected per-bar return on the capital base ~ 0.0001 / 2 (capital base = 1 * 10k * 2 = 20k, leg notional = 10k -> r_a * 10k / 20k)
    # Sharpe = (0.0001/2) / (0.001/2) * sqrt(bars_per_year) = 0.1 * sqrt(bars_per_year)
    expected = 0.1 * np.sqrt(24 * 365 * 60 * 60)
    actual = result.metrics["sharpe_annualized"]
    rel_err = abs(actual - expected) / abs(expected) if expected != 0 else float("inf")
    assert rel_err < 0.30, f"Sharpe annualization off: expected {expected:.2f}, got {actual:.2f}, {rel_err*100:.1f}%"
    print(f"test_sharpe_annualization_uses_bars_per_year: PASS (expected {expected:.0f}, got {actual:.0f})")


def test_flat_fallback_when_no_book_columns():
    n = 20
    pred = np.zeros(n, dtype=int); pred[0] = 1  # open at bar 1
    spread = np.full(n, 0.05)
    preds = _make_preds(n, "AAA_BBB", pred, spread, 100.0, 100.0)
    # walk_book=True but no book columns -> falls back to flat bps
    cfg = BacktestConfig(walk_book=True, taker_fee_bps=10, fallback_slippage_bps=5)
    result = run_backtest(preds, _make_pairs(beta=0.5), cfg)
    # One position change: from 0 to -1 at bar 1.
    # Fee + flat slippage cost = (notional_a + notional_b) * (10 + 5) / 1e4
    # notional_a = 1 * 10000 = 10000, notional_b = 1 * 10000 * 0.5 = 5000
    # cost = 15000 * 0.0015 = $22.50
    expected_cost = 15000.0 * (10 + 5) / 1e4
    actual = -result.metrics["total_pnl_dollars"]
    assert abs(actual - expected_cost) < 1e-6, f"Expected ${expected_cost}, got ${actual}"
    print(f"test_flat_fallback_when_no_book_columns: PASS (cost ${actual:.2f})")


if __name__ == "__main__":
    test_flat_signal_zero_pnl()
    test_state_machine_opens_and_closes_once()
    test_walk_book_fill_vwap_arithmetic()
    test_one_bar_execution_lag()
    test_sharpe_annualization_uses_bars_per_year()
    test_flat_fallback_when_no_book_columns()
    print("\nAll 6 backtester sanity tests PASSED.")
