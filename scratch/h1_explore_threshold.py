"""
EXPLORATORY (NOT the locked H1; hypothesis-generating only): the locked test was
underpowered because the $100k institutional bucket is ~empty at 1s on the
cleanly-cointegrated thin alts (X 99.3% zero, verified). The collaborator's NOTES
flag that the $100k threshold may need a liquidity-appropriate (ADV-percentile)
adjustment. Here we rebuild the over-leg imbalance with a >=$10k bucket
(signed_notional_large + signed_notional_institutional, from the cached leg bars)
and run the same within-pair HAC Spearman test. This does NOT re-run or replace
the locked H1 (FAIL); it only tells us whether ANY institutional-vs-retail flow
signal exists on these pairs that would justify a NEW pre-registration with a
tradeable threshold.

Run:  python scratch/h1_explore_threshold.py
"""
import os, sys, glob
import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"C:\Users\jackw\Desktop\math-199-research"
sys.path.insert(0, os.path.join(ROOT, "phase2_l2"))
from src.statistics import hac_long_run_var   # noqa
from src.features import _zscore, FeatureConfig   # noqa  (phase2 z-score, z_window=3600)

# reuse the locked test's stats helpers
import importlib.util
_h = {}
exec(open(os.path.join(ROOT, "scratch", "h1_test.py")).read().split("def main")[0], _h)
combined_hac, block_boot, std_ranks = _h["combined_hac"], _h["block_boot"], _h["std_ranks"]

BOUND = pd.Timestamp("2024-10-19T19:12:00Z")
W = 60
FC = FeatureConfig()
BARS = os.path.join(ROOT, "scratch", "h1_bars")


def imb10(d, suf):
    num = d[f"signed_notional_large_{suf}"] + d[f"signed_notional_institutional_{suf}"]
    den = d[f"trade_notional_{suf}"].replace(0.0, np.nan)
    return num / den


def main():
    sel = pd.read_csv(os.path.join(ROOT, "scratch", "h1_selected_pairs.csv"))
    prod_by_pair = []
    zero_fracs, per_pair = [], []
    for _, r in sel.iterrows():
        a, b = r["sym_a"], r["sym_b"]
        fa, fb = os.path.join(BARS, f"{a}.parquet"), os.path.join(BARS, f"{b}.parquet")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        ba = pd.read_parquet(fa).add_suffix("_a"); bb = pd.read_parquet(fb).add_suffix("_b")
        d = ba.join(bb, how="inner").sort_index()
        spread = np.log(d["microprice_a"]) - r["alpha"] - r["beta_a_on_b"] * np.log(d["microprice_b"])
        sz = _zscore(spread, FC.z_window_bars)
        i10a, i10b = imb10(d, "a"), imb10(d, "b")
        over_a = (sz > 0).astype(float)
        xpb = over_a * i10a + (1 - over_a) * i10b
        tgt = spread.shift(-FC.target_horizon_bars) - spread
        f = pd.DataFrame({"sz": sz, "xpb": xpb, "tgt": tgt, "ts": d.index})
        f = f.dropna(subset=["sz", "xpb", "tgt"]).reset_index(drop=True)
        f["X"] = f["xpb"].rolling(W, min_periods=W).mean()
        f = f[(pd.to_datetime(f["ts"], utc=True) >= BOUND) & (f["sz"].abs() >= 2) & f["X"].notna()]
        if len(f) < 30:
            continue
        zero_fracs.append((np.abs(f["X"].values) < 1e-12).mean())
        rr, pp = stats.spearmanr(f["X"], f["tgt"])
        per_pair.append((f"{a}_{b}", len(f), rr))
        prod_by_pair.append(std_ranks(f["X"].to_numpy()) * std_ranks(f["tgt"].to_numpy()))

    print("=" * 80)
    print("EXPLORATORY (NOT the locked H1): over-leg imbalance with a >=$10k bucket")
    print(f"mean frac X==0 across pairs: {np.mean(zero_fracs):.3f}  (vs 0.993 at the locked $100k)")
    print("=" * 80)
    print("per-pair Spearman:")
    for p, n, rr in per_pair:
        print(f"  {p:<22} n={n:>6}  rho={rr:+.4f}")
    rho = sum(p.sum() for p in prod_by_pair) / sum(len(p) for p in prod_by_pair)
    print(f"\nwithin-pair combined rho={rho:+.5f}  one-sided(neg) p @lag: ", end="")
    for L in [60, 120, 240]:
        _, _, t, p, _ = combined_hac(prod_by_pair, L)
        print(f"L{L}={p:.3g}(t{t:+.2f}) ", end="")
    _, _, bp = block_boot(prod_by_pair)
    print(f"| boot p={bp:.3g}")
    print("NOTE: exploratory only — generates a hypothesis for a NEW pre-registration; "
          "does NOT change the locked H1 verdict (FAIL).")


if __name__ == "__main__":
    main()
