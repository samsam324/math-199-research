# main.py
from __future__ import annotations

import pandas as pd

from src.data_store import StoreConfig, list_local_symbols, build_close_panel
from src.universe import compute_universe_at_time, filter_top_n_by_liquidity
from src.pair_selection import PairConfig, score_pairs


def main():
    cfg = StoreConfig(
        base_url="https://api.binance.us",  # not used in main; main is local-only
        interval="1h",
        data_dir="data",
    )

    local_symbols = list_local_symbols(cfg)
    if not local_symbols:
        raise RuntimeError("No local data found. Run: python store_data.py")

    # As-of time for universe and training cutoff
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")

    # Universe(t0) = symbols available at t0 (no ranking/filtering)
    universe = compute_universe_at_time(cfg, local_symbols, t0)
    print(f"Local symbols: {len(local_symbols)}")
    print(f"Universe at t0={t0}: {len(universe)}")

    # Training window (only uses data before t0)
    train_days = 180
    train_start = (t0 - pd.Timedelta(days=train_days)).floor(cfg.interval)
    train_end = t0.floor(cfg.interval)

    # Liquidity filter: keep top N by mean USDT volume over the training window.
    # Computed only on training data, so no look-ahead into the test period.
    liquid_top_n = 50
    universe = filter_top_n_by_liquidity(
        cfg=cfg,
        symbols=universe,
        start=train_start,
        end_exclusive=train_end,
        top_n=liquid_top_n,
        min_coverage=0.80,
    )
    print(f"After liquidity filter (top {liquid_top_n} by mean USDT vol): {len(universe)}")

    print(f"Building training close panel [{train_start}, {train_end})...")
    close_panel = build_close_panel(
        cfg=cfg,
        symbols=universe,
        start=train_start,
        end_exclusive=train_end,
        min_symbol_coverage=0.80,
    )
    print(f"Close panel shape: {close_panel.shape} (rows x symbols)")

    if close_panel.empty or close_panel.shape[1] < 2:
        print("Not enough symbols with full coverage in the training window.")
        print("Increase stored history, shorten train_days, or move t0 earlier/later.")
        return

    pcfg = PairConfig(
        min_corr=0.5,
        max_pairs_to_test=9999999999999999999,
        max_adf_pvalue=0.05,
        half_life_min_hours=4.0,
        half_life_max_hours=240.0,
        half_life_target_hours=48.0,
        beta_stability_segments=4,
        max_beta_instability=0.35,
        min_spread_vol=0.002,
        min_obs=1200,
    )

    print("Scoring pairs...")
    pairs = score_pairs(close_panel, pcfg)

    if pairs.empty:
        print("No pairs passed filters. Try min_corr=0.5 or max_adf_pvalue=0.10 or max_half_life_hours=500.")
        return

    print("\nTop pairs:")
    print(pairs.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
