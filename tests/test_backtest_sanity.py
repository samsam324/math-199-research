"""
Independent sanity checks for the backtester in src/backtest.py.

These do not aim to validate against a published equity result; they aim to
establish that the backtester returns numbers consistent with simple
known-correct cases. We construct synthetic scenarios where the answer is
hand-computable, then check the backtester agrees.

Run with:
    python3 -m pytest tests/test_backtest_sanity.py -v
or directly:
    python3 tests/test_backtest_sanity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import BacktestConfig, run_backtest


def _make_pair(sym_a: str, sym_b: str, beta: float = 1.0, n: int = 100) -> tuple:
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Constant prices => zero returns; perfect setting for cost-only tests
    close = pd.DataFrame({sym_a: np.full(n, 100.0), sym_b: np.full(n, 100.0)}, index=ts)
    pairs = pd.DataFrame([{"pair": f"{sym_a}_{sym_b}", "sym_a": sym_a, "sym_b": sym_b, "beta_a_on_b": beta}])
    return ts, close, pairs


def _make_preds(ts: pd.DatetimeIndex, pair: str, pred_class: np.ndarray, current_spread: np.ndarray, spread_z: np.ndarray = None) -> pd.DataFrame:
    df = pd.DataFrame({
        "sample_id": np.arange(len(ts)),
        "pair": pair,
        "timestamp": ts,
        "current_spread": current_spread,
        "y_class": pred_class.astype(int),
        "pred_class": pred_class.astype(int),
        "model": "test",
    })
    if spread_z is not None:
        df["latest_spread_z"] = spread_z
    return df


# ---------------------------------------------------------------------------
# Test 1: Flat signal -> zero PnL, zero trades, zero cost.
# ---------------------------------------------------------------------------
def test_flat_signal_zero_pnl():
    ts, close, pairs = _make_pair("AAA", "BBB", n=100)
    pred = np.ones(100, dtype=int)  # all class 1 = flat
    preds = _make_preds(ts, "AAA_BBB", pred, np.full(100, 0.01))
    result = run_backtest(preds, close, pairs, BacktestConfig(taker_fee_bps=10, slippage_bps=5))
    assert abs(result.metrics["total_pnl_dollars"]) < 1e-9, f"Expected ~0 PnL, got {result.metrics['total_pnl_dollars']}"
    assert result.metrics["trades"] == 0, f"Expected 0 trades, got {result.metrics['trades']}"
    print("test_flat_signal_zero_pnl: PASS")


# ---------------------------------------------------------------------------
# Test 2: Constant prices, persistent non-flat signal -> zero pnl from returns
#         (both legs zero) but exactly two cost charges (open at t=1, never close).
# ---------------------------------------------------------------------------
def test_constant_prices_pnl_is_only_cost():
    ts, close, pairs = _make_pair("AAA", "BBB", beta=0.5, n=10)
    # signal=-1 from t=1 onward (pred_class 0 with current_spread > 0)
    pred = np.zeros(10, dtype=int)  # all class 0 = revert
    pred[0] = 1  # first bar flat so we can observe the open at t=1
    spread = np.full(10, 0.05)  # positive spread => sign +1 => pred 0 means position -1
    preds = _make_preds(ts, "AAA_BBB", pred, spread)
    cfg = BacktestConfig(taker_fee_bps=10, slippage_bps=5, leg_notional=10_000)
    result = run_backtest(preds, close, pairs, cfg)

    # Returns are exactly zero (constant prices) so total PnL = -costs only.
    # Cost on a position change from 0 to -1 at bar 1:
    #   notional_traded = 1 * 10000 * (1 + |0.5|) = 15000
    #   cost = 15000 * 0.0015 = 22.50
    # No further changes => total cost = 22.50.
    expected_cost = 1 * 10_000 * (1 + 0.5) * (15 / 1e4)
    actual = -result.metrics["total_pnl_dollars"]  # PnL is negative of cost
    assert abs(actual - expected_cost) < 1e-6, f"Expected cost {expected_cost}, got {actual}"
    print(f"test_constant_prices_pnl_is_only_cost: PASS (cost = ${actual:.4f})")


# ---------------------------------------------------------------------------
# Test 3: Known return, known position -> hand-computed PnL.
# ---------------------------------------------------------------------------
def test_known_return_pnl():
    n = 5
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Leg A: 100, 101 (+1%), 101, 101, 101
    # Leg B: 100, 100, 100, 100, 100 (no change)
    close = pd.DataFrame({
        "AAA": [100, 101, 101, 101, 101],
        "BBB": [100, 100, 100, 100, 100],
    }, index=ts)
    pairs = pd.DataFrame([{"pair": "AAA_BBB", "sym_a": "AAA", "sym_b": "BBB", "beta_a_on_b": 0.5}])
    # Open long-spread at bar 0 (signal=+1 from t=1 onward); position held through end.
    pred = np.array([2, 2, 2, 2, 2], dtype=int)  # class 2 = diverge => signal = +sign(spread)
    spread = np.full(n, 0.01)  # positive spread, sign=+1, so signal=+1
    preds = _make_preds(ts, "AAA_BBB", pred, spread)
    cfg = BacktestConfig(taker_fee_bps=0, slippage_bps=0, leg_notional=10_000)  # ignore costs
    result = run_backtest(preds, close, pairs, cfg)

    # Position from bar 1 onward = +1. Bar 1 return on A = (101-100)/100 = 0.01.
    # pnl[bar 1] = held_position * L * (r_a - beta * r_b) = 1 * 10000 * (0.01 - 0.5*0) = $100
    # pnl[bar 2..4] = 0 (no further return)
    expected_total = 100.0
    actual = result.metrics["total_pnl_dollars"]
    assert abs(actual - expected_total) < 1e-6, f"Expected ${expected_total}, got ${actual}"
    print(f"test_known_return_pnl: PASS (PnL = ${actual:.4f})")


# ---------------------------------------------------------------------------
# Test 4: Sharpe sign and sanity. Symmetric random pnl => mean ~ 0, Sharpe ~ 0.
# ---------------------------------------------------------------------------
def test_random_signal_sharpe_near_zero():
    rng = np.random.default_rng(7)
    n = 2000
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Random walk prices with no drift
    r_a = rng.normal(0, 0.005, n)
    r_b = rng.normal(0, 0.005, n)
    p_a = 100 * np.exp(np.cumsum(r_a))
    p_b = 100 * np.exp(np.cumsum(r_b))
    close = pd.DataFrame({"AAA": p_a, "BBB": p_b}, index=ts)
    pairs = pd.DataFrame([{"pair": "AAA_BBB", "sym_a": "AAA", "sym_b": "BBB", "beta_a_on_b": 1.0}])
    # Random signal: uniform class in {0,1,2}
    pred = rng.integers(0, 3, n)
    # Random spread sign
    spread = rng.choice([-0.05, 0.05], size=n)
    preds = _make_preds(ts, "AAA_BBB", pred, spread)
    cfg = BacktestConfig(taker_fee_bps=0, slippage_bps=0, leg_notional=10_000)
    result = run_backtest(preds, close, pairs, cfg)
    sharpe = result.metrics["sharpe_annualized"]
    # With true mean 0 and ~bars/year=8760 of samples (we have 2000), Sharpe should be within roughly
    # +/- z * sqrt(8760 / n) ~ 2 * sqrt(8760/2000) ~ 4.2 with 95% probability.
    assert abs(sharpe) < 5.0, f"Random signal Sharpe should be near zero, got {sharpe}"
    print(f"test_random_signal_sharpe_near_zero: PASS (Sharpe = {sharpe:.3f})")


# ---------------------------------------------------------------------------
# Test 5: State machine - opens once, closes once, exactly two transitions.
# ---------------------------------------------------------------------------
def test_state_machine_open_close_exactly_once():
    n = 200
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = pd.DataFrame({"AAA": np.full(n, 100.0), "BBB": np.full(n, 100.0)}, index=ts)
    pairs = pd.DataFrame([{"pair": "AAA_BBB", "sym_a": "AAA", "sym_b": "BBB", "beta_a_on_b": 1.0}])

    # Spread z is high in middle (above entry), zero everywhere else.
    spread = np.zeros(n)
    spread[20:80] = 0.05  # spread positive in this range
    spread_z = np.zeros(n)
    spread_z[20:80] = 3.0  # |z| above entry threshold of 2.0
    # Then between 80 and end, spread_z is 0.0 which is < exit_z=0.5, triggering close.

    pred = np.zeros(n, dtype=int)  # everyone says class 0 (revert)
    preds = _make_preds(ts, "AAA_BBB", pred, spread, spread_z)
    cfg = BacktestConfig(
        taker_fee_bps=0, slippage_bps=0,
        use_state_machine=True, entry_z=2.0, exit_z=0.5,
    )
    result = run_backtest(preds, close, pairs, cfg)

    # Should open at first bar where |z|>=2 (t=20) and close at first bar where |z|<=0.5 (t=80).
    # That's exactly 2 transitions.
    n_trans = result.metrics["trades"]
    assert n_trans == 2, f"Expected exactly 2 transitions (open + close), got {n_trans}"
    print(f"test_state_machine_open_close_exactly_once: PASS ({n_trans} transitions)")


# ---------------------------------------------------------------------------
# Test 6: Annualization is consistent. If we report Sharpe N and the bar
#         interval matches bars_per_year, then per-bar return std=1 gives
#         Sharpe = sqrt(bars_per_year).
# ---------------------------------------------------------------------------
def test_sharpe_annualization_factor():
    # Build a deterministic per-bar return stream with known mean and std,
    # via a known position * known returns.
    n = 1000
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(11)
    # Per-bar leg return = mu + sigma * z, with mu=0.0001, sigma=0.001
    mu, sigma = 0.0001, 0.001
    r_a = rng.normal(mu, sigma, n)
    r_b = np.zeros(n)
    p_a = 100 * np.exp(np.cumsum(r_a))
    p_b = np.full(n, 100.0)
    close = pd.DataFrame({"AAA": p_a, "BBB": p_b}, index=ts)
    pairs = pd.DataFrame([{"pair": "AAA_BBB", "sym_a": "AAA", "sym_b": "BBB", "beta_a_on_b": 1.0}])
    pred = np.full(n, 2, dtype=int)  # class 2 = diverge => sign(+1) since spread>0 => signal=+1
    spread = np.full(n, 0.01)
    preds = _make_preds(ts, "AAA_BBB", pred, spread)
    cfg = BacktestConfig(taker_fee_bps=0, slippage_bps=0, leg_notional=10_000, bars_per_year=24*365)
    result = run_backtest(preds, close, pairs, cfg)

    # Expected: per-bar return on capital base = (leg_notional * r_a) / (1 * leg_notional * 2) = r_a / 2.
    # mean(per-bar return) ~ mu/2, std(per-bar return) ~ sigma/2.
    # Sharpe = mean / std * sqrt(bars_per_year) ~ (mu/sigma) * sqrt(8760).
    expected = (mu / sigma) * np.sqrt(24 * 365)
    actual = result.metrics["sharpe_annualized"]
    rel_err = abs(actual - expected) / abs(expected)
    # n=1000 samples means SE on mean is mu/(sigma*sqrt(n)) of the mean estimate
    # itself, so Sharpe can drift ~25-30% from the limit. Tolerance set accordingly.
    assert rel_err < 0.25, f"Sharpe annualization off: expected ~{expected:.2f}, got {actual:.2f} ({rel_err*100:.1f}% off)"
    print(f"test_sharpe_annualization_factor: PASS (expected {expected:.2f}, got {actual:.2f}, {rel_err*100:.1f}% off)")


if __name__ == "__main__":
    test_flat_signal_zero_pnl()
    test_constant_prices_pnl_is_only_cost()
    test_known_return_pnl()
    test_random_signal_sharpe_near_zero()
    test_state_machine_open_close_exactly_once()
    test_sharpe_annualization_factor()
    print("\nAll 6 backtester sanity checks PASSED.")
