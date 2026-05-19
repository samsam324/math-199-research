"""
Per-pair spread plot with HMM-decoded states colored as background bands.

Fits a 2-state Gaussian HMM on the full available pair history, identifies the
mean-reverting state, and overlays state assignments on the spread.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hmm_filter import HMMConfig, HMM_FEATURE_COLUMNS, fit_hmm, decode_states, identify_mean_reverting_state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot pair spread with HMM-decoded states.")
    p.add_argument("--features-path", required=True, help="all_pair_features.parquet")
    p.add_argument("--pairs", nargs="+", required=True, help="Pair names like ARBUSDT_LINKUSDT")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-states", type=int, default=2)
    p.add_argument("--n-iter", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(args.features_path, engine="pyarrow")
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)

    cfg = HMMConfig(n_states=args.n_states, n_iter=args.n_iter, seed=args.seed)

    for pair in args.pairs:
        pair_features = features[features["pair"] == pair].sort_values("timestamp").reset_index(drop=True)
        if pair_features.empty:
            print(f"skipping {pair}: no features")
            continue
        try:
            model = fit_hmm(pair_features, cfg=cfg, columns=HMM_FEATURE_COLUMNS)
        except Exception as exc:
            print(f"skipping {pair}: HMM fit failed: {exc}")
            continue
        states = decode_states(model, pair_features, columns=HMM_FEATURE_COLUMNS)
        mean_rev = identify_mean_reverting_state(model, pair_features, columns=HMM_FEATURE_COLUMNS)

        ts = pair_features["timestamp"].to_numpy()
        spread = pair_features["spread"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ts, spread, color="black", linewidth=0.7)
        # Color background by state
        boundaries = np.where(np.diff(states) != 0)[0] + 1
        segments = np.split(np.arange(len(states)), boundaries)
        palette = {mean_rev: "#9ecae1", **{s: "#fdae6b" for s in range(args.n_states) if s != mean_rev}}
        for seg in segments:
            if len(seg) == 0:
                continue
            s = int(states[seg[0]])
            ax.axvspan(ts[seg[0]], ts[seg[-1]], color=palette.get(s, "#cccccc"), alpha=0.35, linewidth=0)

        ax.set_title(f"{pair}: spread with HMM regimes (blue = mean-reverting, orange = diverging)")
        ax.set_ylabel("spread")
        ax.set_xlabel("time (UTC)")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
        fig.autofmt_xdate()
        fig.tight_layout()
        out_path = out_dir / f"hmm_states_{pair}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
