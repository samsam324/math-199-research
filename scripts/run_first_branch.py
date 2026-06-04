from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_store import StoreConfig, build_close_panel, list_local_symbols, load_symbol
from src.features import FeatureConfig, build_feature_store, build_kalman_spread_overrides, load_ohlcv_panel
from src.ml_dataset import DatasetConfig, add_walk_forward_splits, build_samples_from_features, save_dataset
from src.modeling import (
    TrainingConfig,
    baseline_majority_class,
    baseline_persist_class,
    baseline_random_stratified,
    baseline_zscore_rule,
    combine_results,
    train_lstm,
    train_transformer,
    train_xgboost_baseline,
)
from src.pair_selection import PairConfig, rank_pairs_by_correlation, score_pairs
from src.universe import compute_universe_at_time, filter_top_n_by_liquidity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Binance statistical-arbitrage branch: features, ML dataset, models, evaluation.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="artifacts/first_branch")
    p.add_argument("--t0", default="2026-01-01T00:00:00Z")
    p.add_argument("--train-days", type=int, default=180)
    p.add_argument("--liquid-top-n", type=int, default=50)
    p.add_argument("--top-pairs", type=int, default=20)
    p.add_argument("--max-pairs-to-test", type=int, default=6000)
    p.add_argument("--window", type=int, default=168)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--max-train-samples", type=int, default=12000)
    p.add_argument("--max-test-samples", type=int, default=5000)
    p.add_argument("--dl-epochs", type=int, default=5)
    p.add_argument("--skip-deep", action="store_true", help="Run feature, dataset, XGBoost/fallback, and z-score baseline only.")
    p.add_argument("--use-kalman", action="store_true", help="Use Kalman dynamic-beta spreads (MLE-fit on training data) as the spread input to features.")
    p.add_argument("--test-days", type=int, default=60, help="OOS test window length in days. Capped by the available data tail.")
    p.add_argument("--min-history-days", type=float, default=0.0, help="Minimum days of history before t0 for a symbol to enter the as-of universe.")
    p.add_argument("--per-pair-label", action="store_true", help="Use per-pair-z-scored 3-class label threshold instead of fixed 0.001.")
    p.add_argument("--label-scale-factor", type=float, default=0.5, help="Scale factor on per-pair std(|d(|spread|)|) for the label threshold.")
    p.add_argument("--pairs-path", default=None, help="Optional pre-selected pairs parquet (e.g. from run_kalman_pair_screen). Bypasses score_pairs / correlation fallback.")
    p.add_argument("--with-micro", action="store_true", help="Enrich the feature store with volume-as-information features (data/microstructure panels) so the saved dataset carries micro_* columns for run_walk_forward --with-micro.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = StoreConfig(interval="1h", data_dir=args.data_dir)
    local_symbols = list_local_symbols(cfg)
    if not local_symbols:
        raise RuntimeError("No local data found. Run: python3 store_data.py")

    t0 = pd.Timestamp(args.t0)
    train_start = (t0 - pd.Timedelta(days=args.train_days)).floor(cfg.interval)
    train_end = t0.floor(cfg.interval)
    btc = load_symbol(cfg, "BTCUSDT", columns=["close"])
    if btc is None or btc.empty:
        raise RuntimeError("BTCUSDT is required for market-context features.")
    test_end = min(btc.index.max() + pd.Timedelta(hours=1), train_end + pd.Timedelta(days=args.test_days))

    print(f"Local symbols: {len(local_symbols)}", flush=True)
    universe = compute_universe_at_time(cfg, local_symbols, t0, min_history_days=args.min_history_days)
    print(f"Universe at {t0} (min_history_days={args.min_history_days}): {len(universe)}", flush=True)

    liquid = filter_top_n_by_liquidity(cfg, universe, train_start, train_end, top_n=args.liquid_top_n, min_coverage=0.80)
    print(f"Liquidity-filtered universe: {len(liquid)}", flush=True)
    (out_dir / "liquid_universe.json").write_text(json.dumps(liquid, indent=2), encoding="utf-8")

    close_panel = build_close_panel(cfg, liquid, train_start, train_end, min_symbol_coverage=0.80)
    print(f"Training close panel: {close_panel.shape}", flush=True)
    if close_panel.shape[1] < 2:
        raise RuntimeError("Not enough symbols after liquidity and coverage filtering.")

    if args.pairs_path:
        print(f"Loading pre-selected pairs from {args.pairs_path}", flush=True)
        pairs = pd.read_parquet(args.pairs_path, engine="pyarrow")
        # Restrict to symbols present in our liquid close_panel, otherwise downstream breaks
        keep_mask = pairs["sym_a"].isin(close_panel.columns) & pairs["sym_b"].isin(close_panel.columns)
        dropped = (~keep_mask).sum()
        if dropped:
            print(f"  dropping {dropped} pairs whose symbols aren't in the liquidity-filtered close panel", flush=True)
        pairs = pairs[keep_mask].reset_index(drop=True)
    else:
        pcfg = PairConfig(max_pairs_to_test=args.max_pairs_to_test, min_obs=1200)
        pairs = score_pairs(close_panel, pcfg)
        if pairs.empty:
            print("No pairs passed strict filters; using correlation fallback for downstream pipeline.", flush=True)
            pairs = rank_pairs_by_correlation(close_panel, top_n=max(args.top_pairs, 20), min_corr=0.20)
        else:
            pairs["selection_method"] = "strict_cointegration"
    if pairs.empty:
        raise RuntimeError("No pairs available after strict and fallback selection.")
    pairs_out = out_dir / "selected_pairs.parquet"
    pairs.to_parquet(pairs_out, engine="pyarrow", index=False)
    print(f"Selected pairs: {len(pairs)}; saved {pairs_out}", flush=True)
    print(pairs.head(args.top_pairs).to_string(index=False), flush=True)

    panels = load_ohlcv_panel(cfg, sorted(set(liquid + ["BTCUSDT"])), train_start, test_end, min_symbol_coverage=0.75)
    fcfg = FeatureConfig(target_horizon=args.horizon)
    features_dir = out_dir / "features"

    spread_overrides = None
    if args.use_kalman:
        print(f"Building Kalman dynamic-beta spreads (MLE-fit on data < {train_end})...", flush=True)
        spread_overrides = build_kalman_spread_overrides(
            close=panels["close"],
            pairs=pairs,
            train_end=train_end,
            top_pairs=args.top_pairs,
        )
        print(f"Kalman spreads built for {len(spread_overrides)} pairs.", flush=True)

    features = build_feature_store(
        panels["close"], panels["volume_usdt"], pairs, features_dir,
        fcfg=fcfg, top_pairs=args.top_pairs, spread_overrides=spread_overrides,
    )
    if features.empty:
        raise RuntimeError("Feature store is empty.")
    print(f"Feature rows: {len(features)}; pairs: {features['pair'].nunique()}", flush=True)

    if args.with_micro:
        from src.microstructure_features import merge_into_pair_features
        features = merge_into_pair_features(features)
        micro_cols = [c for c in features.columns if c.startswith("micro_")]
        cov = float(features[micro_cols].notna().any(axis=1).mean()) if micro_cols else 0.0
        print(f"Microstructure features merged: {len(micro_cols)} cols, {cov:.1%} of rows have L2 coverage.", flush=True)

    dcfg = DatasetConfig(
        window=args.window,
        horizon=args.horizon,
        per_pair_label=args.per_pair_label,
        label_scale_factor=args.label_scale_factor,
        # Anchor the per-pair label threshold to data strictly before t0
        # (which is the latest training boundary across all walk-forward
        # splits). Without this anchor, the threshold leaked future-of-test
        # spread vol into the label distribution.
        label_train_end=train_end if args.per_pair_label else None,
    )
    samples, sequences, _ = build_samples_from_features(features, dcfg)
    samples = add_walk_forward_splits(samples, dcfg)
    dataset_dir = out_dir / "dataset"
    save_dataset(samples, sequences, dataset_dir, dcfg)
    if samples.empty:
        raise RuntimeError("ML dataset is empty.")
    print(f"Dataset samples: {len(samples)}; sequences: {sequences.shape}; saved {dataset_dir}", flush=True)

    tcfg = TrainingConfig(
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        dl_epochs=args.dl_epochs,
    )
    results = [
        baseline_persist_class(samples, tcfg),
        baseline_majority_class(samples, tcfg),
        baseline_random_stratified(samples, tcfg),
        baseline_zscore_rule(samples, tcfg),
        train_xgboost_baseline(samples, tcfg),
    ]
    if not args.skip_deep:
        results.append(train_lstm(samples, sequences, tcfg))
        results.append(train_transformer(samples, sequences, tcfg))

    preds, metrics = combine_results(results)
    preds.to_parquet(out_dir / "predictions.parquet", engine="pyarrow", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    print("Metrics:", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print(f"Saved outputs under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
