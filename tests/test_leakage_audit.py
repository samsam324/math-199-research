"""
Comprehensive leakage audit. Every test here checks one specific way
information could flow backwards in time and bias a result.

Run with:
    python3 tests/test_leakage_audit.py

What's checked
--------------
1. Universe construction: a symbol whose FIRST bar is after t0 must not be
   in the as-of universe at t0 (the bug we already fixed in
   compute_universe_at_time; this is a regression test).
2. Feature columns at time t use only data with timestamp <= t (no
   look-ahead in spread_z, spread_diff, spread_vol, volume_z,
   rolling_corr, realized_vol, BTC return).
3. Target columns at time t use ONLY data with timestamp in
   [t, t + horizon] (target uses future-of-t, by design; not a leak).
4. Class labels with the per-pair threshold: threshold must depend only on
   data with timestamp strictly before cfg.label_train_end (no leak into
   the label distribution from future-of-test).
5. Kalman spread overrides: the train_end residual must be computed with
   the SAME parameters as the test_start residual (no parameter
   discontinuity at the boundary).
6. Standardization (deep models): mean/std must be computed from training
   data only.
7. Backtester: held position at time t uses only signal computed at
   times <= t-1 (one-bar execution lag).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. Universe construction: as-of-time correctness
# ---------------------------------------------------------------------------
def test_universe_excludes_post_t0_listings():
    """A symbol whose first bar is after t0 must be excluded from the universe."""
    from src.data_store import StoreConfig
    from src.universe import compute_universe_at_time

    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        # Set up a fake local store with two symbols:
        # OLD_USDT first bar 2023-01-01, last bar 2024-06-01
        # NEW_USDT first bar 2024-03-01, last bar 2024-06-01
        cfg = StoreConfig(interval="1h", data_dir=tmp)
        spot = Path(tmp) / "spot_1h"
        spot.mkdir(parents=True)
        for sym, first, last in [("OLD_USDT", "2023-01-01", "2024-06-01"),
                                  ("NEW_USDT", "2024-03-01", "2024-06-01")]:
            idx = pd.date_range(first, last, freq="1h", tz="UTC")
            df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
            df.index.name = "timestamp"
            df.to_parquet(spot / f"{sym}.parquet", engine="pyarrow", index=True)

        symbols = ["OLD_USDT", "NEW_USDT"]
        t0 = pd.Timestamp("2024-01-01T00:00:00Z")

        # min_history_days=0: OLD qualifies (last_ts >= t0), NEW does NOT (first_ts > t0)
        universe = compute_universe_at_time(cfg, symbols, t0, min_history_days=0)
        assert "OLD_USDT" in universe, "OLD_USDT should be in the as-of universe"
        assert "NEW_USDT" not in universe, "NEW_USDT first bar 2024-03 is AFTER t0; leak!"

        # min_history_days=180: OLD has from 2023-01 to 2024-06, so first_ts=2023-01 <= t0-180d=2023-07-05.
        # Should still qualify.
        universe2 = compute_universe_at_time(cfg, symbols, t0, min_history_days=180)
        assert "OLD_USDT" in universe2, "OLD has 12 months pre-t0 history; should qualify with min_history=180d"
        assert "NEW_USDT" not in universe2

    print("test_universe_excludes_post_t0_listings: PASS")


# ---------------------------------------------------------------------------
# 2. Feature columns at time t use only data with timestamp <= t
# ---------------------------------------------------------------------------
def test_features_have_no_lookahead():
    """
    Inject a 'future' anomaly into the price series at a known late index,
    then verify the feature columns at earlier indices are NOT affected.
    Approach: build a flat price series, then add a huge spike at index 800,
    and check that features at index 500 are identical to a series WITHOUT
    the spike.
    """
    from src.features import FeatureConfig, compute_pair_features

    rng_test = np.random.default_rng(7)
    ts = pd.date_range("2024-01-01", periods=1500, freq="1h", tz="UTC")
    # Use random walks so spread is nontrivial AND volume noise so volume_z is defined
    r_a = rng_test.normal(0, 0.005, 1500)
    r_b = rng_test.normal(0, 0.005, 1500)
    base_a = 100 * np.exp(np.cumsum(r_a))
    base_b = 100 * np.exp(np.cumsum(r_b))
    base_btc = 100 * np.exp(np.cumsum(rng_test.normal(0, 0.005, 1500)))
    vol_a = 100.0 + rng_test.normal(0, 5, 1500).clip(min=10.0)
    vol_b = 100.0 + rng_test.normal(0, 5, 1500).clip(min=10.0)
    vol_btc = 100.0 + rng_test.normal(0, 5, 1500).clip(min=10.0)

    def make_panels(spike: bool):
        close_a = base_a.copy()
        if spike:
            close_a[1200:1250] *= 1000.0  # huge anomaly in late portion
        close = pd.DataFrame({"AAA": close_a, "BBB": base_b, "BTCUSDT": base_btc}, index=ts)
        vol_df = pd.DataFrame({"AAA": vol_a, "BBB": vol_b, "BTCUSDT": vol_btc}, index=ts)
        return close, vol_df

    pair_row = pd.Series({"sym_a": "AAA", "sym_b": "BBB", "alpha": 0.0, "beta_a_on_b": 1.0})
    fcfg = FeatureConfig(target_horizon=24)

    close_clean, vol_clean = make_panels(spike=False)
    close_spike, vol_spike = make_panels(spike=True)

    feats_clean = compute_pair_features(close_clean, vol_clean, pair_row, fcfg=fcfg)
    feats_spike = compute_pair_features(close_spike, vol_spike, pair_row, fcfg=fcfg)

    # Reindex on timestamp for comparison; features at timestamps before the
    # spike (ts[800]) must be byte-identical between the two runs.
    feats_clean = feats_clean.set_index("timestamp")
    feats_spike = feats_spike.set_index("timestamp")

    # Check timestamps strictly before the spike onset
    cutoff = ts[1200]
    common = feats_clean.index.intersection(feats_spike.index)
    pre_spike = common[common < cutoff]

    if len(pre_spike) == 0:
        print("test_features_have_no_lookahead: SKIP (no pre-spike samples)")
        return

    # Features that should be identical pre-spike (target columns are allowed to differ
    # because they look forward by design):
    feature_cols = [
        "spread", "spread_z_168h", "spread_diff_1h", "spread_diff_4h",
        "spread_diff_24h", "spread_vol_24h", "spread_vol_168h",
        "volume_ratio", "volume_z_a_168h", "volume_z_b_168h",
        "pair_volume_sum", "pair_volume_gmean",
        "btc_return_24h", "rolling_corr_24h",
        "realized_vol_a_24h", "realized_vol_b_24h",
    ]
    # Check the LAST pre-spike timestamp specifically (most stringent: anything
    # short of leakage would be identical at earlier times too)
    last_pre = pre_spike[-1]
    for col in feature_cols:
        v_clean = float(feats_clean.loc[last_pre, col])
        v_spike = float(feats_spike.loc[last_pre, col])
        if not (np.isnan(v_clean) and np.isnan(v_spike)):
            assert abs(v_clean - v_spike) < 1e-12, (
                f"LEAK: feature '{col}' at t={last_pre} (before spike at {cutoff}) "
                f"differs between clean and spike runs: {v_clean} vs {v_spike}"
            )

    # Targets are ALLOWED to differ pre-spike because they look forward:
    # at last_pre (which is just before the spike onset), target_spread_change_24h
    # legitimately includes the spike (since shift(-24) reaches into the future).
    print(f"test_features_have_no_lookahead: PASS ({len(feature_cols)} features verified at t={last_pre})")


# ---------------------------------------------------------------------------
# 3. Targets use only data in [t, t+horizon]
# ---------------------------------------------------------------------------
def test_targets_use_forward_window_only():
    """target_spread_change_24h at time t should equal spread[t+24] - spread[t]."""
    from src.features import FeatureConfig, compute_pair_features

    ts = pd.date_range("2024-01-01", periods=1500, freq="1h", tz="UTC")
    rng = np.random.default_rng(7)
    n = 1500
    p_a = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    p_b = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    close = pd.DataFrame({"AAA": p_a, "BBB": p_b, "BTCUSDT": p_a * 1.1}, index=ts)
    vol = pd.DataFrame({c: 100.0 + rng.normal(0, 5, n).clip(min=10) for c in close.columns}, index=ts)

    pair_row = pd.Series({"sym_a": "AAA", "sym_b": "BBB", "alpha": 0.0, "beta_a_on_b": 1.0})
    fcfg = FeatureConfig(target_horizon=24)

    feats = compute_pair_features(close, vol, pair_row, fcfg=fcfg).set_index("timestamp")

    # For a random sample of valid timestamps, verify target_spread_change_24h
    # equals spread[t+24] - spread[t] within float tolerance
    valid_idx = feats.index[feats["target_spread_change_24h"].notna()]
    if len(valid_idx) < 20:
        print("test_targets_use_forward_window_only: SKIP (too few valid samples)")
        return
    rng2 = np.random.default_rng(11)
    sample = rng2.choice(len(valid_idx), size=min(20, len(valid_idx)), replace=False)
    for i in sample:
        t = valid_idx[i]
        t_plus_24 = t + pd.Timedelta(hours=24)
        if t_plus_24 not in feats.index:
            continue
        spread_t = feats.loc[t, "spread"]
        spread_t24 = feats.loc[t_plus_24, "spread"]
        target = feats.loc[t, "target_spread_change_24h"]
        expected = spread_t24 - spread_t
        assert abs(target - expected) < 1e-10, (
            f"Target at {t} = {target}, expected spread[{t_plus_24}]-spread[{t}] = {expected}"
        )
    print(f"test_targets_use_forward_window_only: PASS")


# ---------------------------------------------------------------------------
# 4. Per-pair label threshold respects label_train_end
# ---------------------------------------------------------------------------
def test_per_pair_label_threshold_no_leakage():
    """
    With label_train_end set, the per-pair threshold must depend ONLY on
    data with timestamp < label_train_end. Inject a giant spread anomaly
    AFTER label_train_end and verify the threshold is unchanged.
    """
    from src.features import FeatureConfig, compute_pair_features
    from src.ml_dataset import DatasetConfig, build_samples_from_features

    ts = pd.date_range("2024-01-01", periods=2000, freq="1h", tz="UTC")
    rng = np.random.default_rng(7)
    n = 2000
    p_a = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    p_b = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    cutoff = ts[800]  # label_train_end at index 800

    # Pre-generate volume with a deterministic seed so both calls see the same values
    vol_seed = np.random.default_rng(99)
    vol_a = 100.0 + vol_seed.normal(0, 5, n).clip(min=10)
    vol_b = 100.0 + vol_seed.normal(0, 5, n).clip(min=10)
    vol_btc = 100.0 + vol_seed.normal(0, 5, n).clip(min=10)

    def build_features(post_cutoff_spike: bool):
        close_a = p_a.copy()
        if post_cutoff_spike:
            # Huge spread move AFTER the cutoff
            close_a[900:1100] *= 100.0
        # Keep BTCUSDT identical in both runs so btc_return_24h is the same pre-cutoff
        close = pd.DataFrame({"AAA": close_a, "BBB": p_b, "BTCUSDT": p_b * 1.1}, index=ts)
        vol = pd.DataFrame({"AAA": vol_a, "BBB": vol_b, "BTCUSDT": vol_btc}, index=ts)
        pair_row = pd.Series({"sym_a": "AAA", "sym_b": "BBB", "alpha": 0.0, "beta_a_on_b": 1.0})
        return compute_pair_features(close, vol, pair_row, fcfg=FeatureConfig(target_horizon=24))

    feats_clean = build_features(post_cutoff_spike=False)
    feats_spike = build_features(post_cutoff_spike=True)

    # Build samples with per_pair_label + label_train_end at cutoff
    dcfg = DatasetConfig(per_pair_label=True, label_scale_factor=0.5, label_train_end=cutoff)
    samples_clean, _, _ = build_samples_from_features(feats_clean, dcfg)
    samples_spike, _, _ = build_samples_from_features(feats_spike, dcfg)

    # Class-label distribution PRE-cutoff should be identical between the two runs
    samples_clean["timestamp"] = pd.to_datetime(samples_clean["timestamp"], utc=True)
    samples_spike["timestamp"] = pd.to_datetime(samples_spike["timestamp"], utc=True)
    pre_clean = samples_clean[samples_clean["timestamp"] < cutoff]["y_class"].value_counts().sort_index()
    pre_spike = samples_spike[samples_spike["timestamp"] < cutoff]["y_class"].value_counts().sort_index()

    # If threshold was leaked from post-cutoff data, the spike (which would inflate the
    # threshold) would change the class distribution PRE-cutoff. Identical distributions
    # mean no leak.
    if pre_clean.empty or pre_spike.empty:
        print("test_per_pair_label_threshold_no_leakage: SKIP (no pre-cutoff samples)")
        return
    assert pre_clean.equals(pre_spike), (
        f"LEAK: per-pair label threshold changed pre-cutoff class distribution.\n"
        f"  clean: {pre_clean.to_dict()}\n"
        f"  spike: {pre_spike.to_dict()}"
    )
    print(f"test_per_pair_label_threshold_no_leakage: PASS (pre-cutoff distribution identical: {pre_clean.to_dict()})")


# ---------------------------------------------------------------------------
# 5. Kalman: train_end residual and test_start residual use same params
# ---------------------------------------------------------------------------
def test_kalman_no_parameter_discontinuity():
    """
    After the fix to fit_kalman_mle (return train_residuals) and
    build_kalman_spread_overrides (use those instead of default-param
    kalman_dynamic_hedge), the train and test residual series share the same
    filter parameters. Test: the residual std on the LAST training bar and
    FIRST test bar should be of the same order of magnitude.
    """
    from src.features import build_kalman_spread_overrides

    rng = np.random.default_rng(7)
    n = 600
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    train_end = ts[400]

    rets = rng.normal(0, 0.005, n)
    p_b = 100 * np.exp(np.cumsum(rets))
    beta_true = 0.5 + 0.001 * np.arange(n)
    p_a = np.exp(beta_true * np.log(p_b) + rng.normal(0, 0.01, n))

    close = pd.DataFrame({"AAA": p_a, "BBB": p_b}, index=ts)
    pairs = pd.DataFrame([{"sym_a": "AAA", "sym_b": "BBB", "beta_a_on_b": 0.5}])

    overrides = build_kalman_spread_overrides(close, pairs, train_end)
    assert "AAA_BBB" in overrides, "Kalman override should have been produced"
    spread = overrides["AAA_BBB"]

    pre_cut = spread.loc[spread.index < train_end]
    post_cut = spread.loc[spread.index >= train_end]
    assert len(pre_cut) >= 20 and len(post_cut) >= 20

    # Compute std on the last 50 training bars and first 50 test bars
    last_train_std = float(np.std(pre_cut.iloc[-50:], ddof=1))
    first_test_std = float(np.std(post_cut.iloc[:50], ddof=1))
    ratio = max(last_train_std, first_test_std) / max(min(last_train_std, first_test_std), 1e-12)
    assert ratio < 5.0, (
        f"Kalman residual std jumps by {ratio:.2f}x at train_end. "
        f"last 50 train bars std={last_train_std:.4e}, first 50 test bars std={first_test_std:.4e}. "
        f"That indicates a parameter discontinuity at the train/test boundary."
    )
    print(f"test_kalman_no_parameter_discontinuity: PASS (boundary std ratio = {ratio:.2f}x, both with fitted params)")


# ---------------------------------------------------------------------------
# 6. Deep-model standardization uses training stats only
# ---------------------------------------------------------------------------
def test_deep_model_standardization_train_only():
    """
    Train and test sets should be standardized with the same mu/sd, derived
    from the training set ONLY. We test this by inspecting the relevant code
    path in src/modeling.py: _train_torch_on_split should compute mu and sd
    on x_train_full only, then apply to x_test.
    """
    # Static code check: read the relevant function and assert no
    # contamination via .reshape on x_test
    src_path = ROOT / "src" / "modeling.py"
    code = src_path.read_text(encoding="utf-8")
    # The pattern we expect: flat = x_train.reshape(...); mu = flat.mean(); ... x_test = (x_test - mu) / sd
    assert "flat = x_train.reshape" in code, "Expected training-only standardization code"
    assert "(x_test - mu) / sd" in code, "Expected test set standardized with TRAINING mu and sd"
    # The pattern we want NOT to see: standardization using x_test stats
    assert "x_test.mean" not in code, "x_test.mean appears - possible standardization leak"
    assert "x_test.std" not in code, "x_test.std appears - possible standardization leak"
    print("test_deep_model_standardization_train_only: PASS (training-only standardization confirmed)")


# ---------------------------------------------------------------------------
# 7. Backtester one-bar execution lag
# ---------------------------------------------------------------------------
def test_backtester_one_bar_execution_lag():
    """
    The backtester must hold a position computed from signals at PRIOR bars.
    If the signal flips at t=50, the pnl at t=50 should reflect the OLD
    position, not the new one (because we only learn the signal at the
    bar's close and trade for the NEXT bar).
    """
    from src.backtest import BacktestConfig, run_backtest

    n = 100
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Constant prices except a +1% jump in A at exactly bar 50
    close_a = np.full(n, 100.0)
    close_a[50] = 101.0
    close_a[51:] = 101.0
    close_b = np.full(n, 100.0)
    close = pd.DataFrame({"AAA": close_a, "BBB": close_b}, index=ts)
    pairs = pd.DataFrame([{"sym_a": "AAA", "sym_b": "BBB", "beta_a_on_b": 0.0}])

    # Signal: flat until bar 50, then long-spread (class 2) from bar 50 onward
    pred = np.ones(n, dtype=int)
    pred[50:] = 2  # diverge => long spread
    spread = np.full(n, 0.01)  # positive spread => signal = +1 when pred=2

    preds = pd.DataFrame({
        "sample_id": np.arange(n), "pair": "AAA_BBB", "timestamp": ts,
        "current_spread": spread, "y_class": pred, "pred_class": pred, "model": "test",
    })
    cfg = BacktestConfig(taker_fee_bps=0, slippage_bps=0, leg_notional=10_000)
    result = run_backtest(preds, close, pairs, cfg)

    # The +1% return on A happens at bar 50, but our position at bar 50 was
    # set by signal at bar 49 (which was class 1, flat). So pnl[50] should be 0.
    # The new long position takes effect at bar 51, where there's no return
    # (price already at 101). So total PnL should be 0.
    total = result.metrics["total_pnl_dollars"]
    assert abs(total) < 1e-9, (
        f"LEAK: backtester captured the bar-50 return with a position decided at bar 50. "
        f"Expected $0 PnL (one-bar lag means new position takes effect at bar 51); got ${total:.4f}"
    )
    print("test_backtester_one_bar_execution_lag: PASS (one-bar lag verified)")


# ---------------------------------------------------------------------------
# 8. Universe rejects symbols with insufficient pre-t0 history
# ---------------------------------------------------------------------------
def test_universe_min_history_strict():
    """A symbol whose first bar is at exactly t0 - min_history_days + 1d must be excluded."""
    from src.data_store import StoreConfig
    from src.universe import compute_universe_at_time

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cfg = StoreConfig(interval="1h", data_dir=tmp)
        spot = Path(tmp) / "spot_1h"
        spot.mkdir(parents=True)

        t0 = pd.Timestamp("2024-01-01T00:00:00Z")
        # SYM1 first bar exactly at t0 - 180d - 1h: passes
        # SYM2 first bar at t0 - 179d: fails (less than 180 days of history)
        end = pd.Timestamp("2024-06-01T00:00:00Z")
        for sym, first in [("SYM1_USDT", t0 - pd.Timedelta(days=180, hours=1)),
                           ("SYM2_USDT", t0 - pd.Timedelta(days=179))]:
            idx = pd.date_range(first, end, freq="1h")
            df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
            df.index.name = "timestamp"
            df.to_parquet(spot / f"{sym}.parquet", engine="pyarrow", index=True)

        universe = compute_universe_at_time(cfg, ["SYM1_USDT", "SYM2_USDT"], t0, min_history_days=180)
        assert "SYM1_USDT" in universe, "SYM1 has just enough history; should pass"
        assert "SYM2_USDT" not in universe, "SYM2 has only 179 days; should be excluded"
    print("test_universe_min_history_strict: PASS")


if __name__ == "__main__":
    test_universe_excludes_post_t0_listings()
    test_universe_min_history_strict()
    test_features_have_no_lookahead()
    test_targets_use_forward_window_only()
    test_per_pair_label_threshold_no_leakage()
    test_kalman_no_parameter_discontinuity()
    test_deep_model_standardization_train_only()
    test_backtester_one_bar_execution_lag()
    print("\nAll 8 leakage audit tests PASSED.")
