"""
H1 task — STEP 2: build held-out-tail 1s features for the 10 selected pairs,
using the collaborator's phase2_l2 code (to match the locked pre-registration).

Held-out tail = bars with timestamp >= HELDOUT_BOUNDARY (2024-10-19 19:12 UTC),
i.e. the last 20% of the 2024 1s L2 range. We load raw L2 from one warmup day
before the boundary so spread_z (3600-bar roll) and the W=60 mean are warm at
the boundary; only bars >= boundary are kept for the test.

Per leg (cached once; legs are shared across pairs): load raw book_snapshot_25
top-of-book + trades, map to phase2 schema, build 1s bars via phase2 bars.build_bars.
Per pair: join legs, compute phase2 features.compute_pair_features (beta from
h1_select, alpha=0 — irrelevant since spread_z is a rolling-3600 z), then form the
locked test feature X = trailing W=60 mean of inst_buy_imbalance_over_leg, and
Y = target_spread_change. Save [timestamp, spread_z, X, Y] per pair.

Env knobs for smoke test: H1_DAYS (e.g. "2024-10-18,2024-10-19"), H1_PAIRS (int n).

Run:  python scratch/h1_build_features.py
"""
import os, sys, glob, time
import numpy as np
import pandas as pd

ROOT = r"C:\Users\jackw\Desktop\math-199-research"
sys.path.insert(0, os.path.join(ROOT, "phase2_l2"))   # so `import src.*` = phase2_l2/src
from src.bars import build_pair_bars, BarConfig          # noqa: E402
from src.features import compute_pair_features, FeatureConfig  # noqa: E402
from src.microstructure import FlowConfig                 # noqa: E402

RAW = os.path.join(ROOT, "data", "l2_raw", "binance")
BARS_DIR = os.path.join(ROOT, "scratch", "h1_bars")
FEAT_DIR = os.path.join(ROOT, "scratch", "h1_feat")
os.makedirs(BARS_DIR, exist_ok=True); os.makedirs(FEAT_DIR, exist_ok=True)

HELDOUT_BOUNDARY = pd.Timestamp("2024-10-19T19:12:00Z")
WARMUP_DAYS = ["2024-10-17", "2024-10-18"]   # warm spread_z (3600-bar) + W=60 mean at the boundary
W = 60                          # rolling-mean window for the over-leg imbalance (locked)
FCFG = FeatureConfig()          # z_window_bars=3600, target_horizon_bars=60 (locked defaults)
BCFG = BarConfig(obi_depth_levels=1, flow_cfg=FlowConfig())   # obi not used by H1; level-1 only


def all_heldout_days():
    days = pd.date_range("2024-10-19", "2024-12-31", freq="D").strftime("%Y-%m-%d").tolist()
    return WARMUP_DAYS + days


def load_leg_raw_day(sym, date):
    """Return (book, trades) phase2-schema frames for one symbol-day, or (None,None)."""
    bp = os.path.join(RAW, "book_snapshot_25", sym, f"{date}.csv.gz")
    tp = os.path.join(RAW, "trades", sym, f"{date}.csv.gz")
    if not (os.path.exists(bp) and os.path.exists(tp)):
        return None, None
    b = pd.read_csv(bp, usecols=["timestamp", "asks[0].price", "asks[0].amount",
                                 "bids[0].price", "bids[0].amount"])
    b = b.rename(columns={"asks[0].price": "ask_px_1", "asks[0].amount": "ask_sz_1",
                          "bids[0].price": "bid_px_1", "bids[0].amount": "bid_sz_1"})
    b = b.dropna(subset=["bid_px_1", "ask_px_1"])
    b = b[(b["bid_px_1"] > 0) & (b["ask_px_1"] > 0)]
    b["timestamp"] = pd.to_datetime(b["timestamp"], unit="us", utc=True)
    t = pd.read_csv(tp, usecols=["timestamp", "side", "price", "amount"])
    t["timestamp"] = pd.to_datetime(t["timestamp"], unit="us", utc=True)
    t["notional"] = t["price"].astype(float) * t["amount"].astype(float)
    return b, t


def build_leg_bars(sym, days):
    """Build & cache 1s bars (phase2 single-symbol schema) for one leg over `days`."""
    cache = os.path.join(BARS_DIR, f"{sym}.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    from src.bars import build_bars
    frames = []
    for d in days:
        b, t = load_leg_raw_day(sym, d)
        if b is None:
            print(f"    [warn] missing {sym} {d}", flush=True); continue
        bars = build_bars(b, t, BCFG)
        if not bars.empty:
            frames.append(bars)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.to_parquet(cache)
    return out


def main():
    t0 = time.time()
    sel = pd.read_csv(os.path.join(ROOT, "scratch", "h1_selected_pairs.csv"))
    days_env = os.environ.get("H1_DAYS")
    days = days_env.split(",") if days_env else all_heldout_days()
    npairs = int(os.environ.get("H1_PAIRS", len(sel)))
    sel = sel.head(npairs)
    print(f"Building H1 held-out features: {len(sel)} pairs, {len(days)} days "
          f"(boundary {HELDOUT_BOUNDARY}); W={W}", flush=True)

    # unique legs (shared across pairs) -> build once
    legs = sorted(set(sel["sym_a"]).union(sel["sym_b"]))
    leg_bars = {}
    for i, leg in enumerate(legs):
        print(f"  [{i+1}/{len(legs)}] leg bars: {leg} ... (elapsed {time.time()-t0:.0f}s)", flush=True)
        leg_bars[leg] = build_leg_bars(leg, days)

    for _, r in sel.iterrows():
        a, b = r["sym_a"], r["sym_b"]
        pair = f"{a}_{b}"
        ba, bb = leg_bars.get(a), leg_bars.get(b)
        if ba is None or bb is None or ba.empty or bb.empty:
            print(f"  [skip] {pair}: missing leg bars", flush=True); continue
        pair_bars = ba.add_suffix("_a").join(bb.add_suffix("_b"), how="inner").sort_index()
        # alpha centers the spread so the collaborator's raw-spread over-leg test is valid;
        # primary over-leg uses spread_z (alpha-invariant + consistent with the |z|>=2 entry).
        pair_row = pd.Series({"sym_a": a, "sym_b": b, "alpha": float(r.get("alpha", 0.0)),
                              "beta_a_on_b": float(r["beta_a_on_b"])})
        feats = compute_pair_features(pair_bars, pair_row, fcfg=FCFG)
        if feats.empty:
            print(f"  [skip] {pair}: no features", flush=True); continue
        feats = feats.sort_values("timestamp").reset_index(drop=True)
        # PRIMARY over-leg = expensive leg = spread_z>0 (robust to the alpha/centering issue
        # that makes raw spread>0 stuck on one leg for different-priced legs).
        over_a = (feats["spread_z"] > 0).astype(float)
        x_perbar = over_a * feats["inst_buy_imbalance_a"] + (1 - over_a) * feats["inst_buy_imbalance_b"]
        feats["X_inst_over_leg_W60"] = x_perbar.rolling(W, min_periods=W).mean()
        # robustness: the collaborator's raw-spread over-leg (now with proper alpha)
        feats["X_rawspread_W60"] = feats["inst_buy_imbalance_over_leg"].rolling(W, min_periods=W).mean()
        keep = feats[["timestamp", "spread_z", "X_inst_over_leg_W60", "X_rawspread_W60",
                      "target_spread_change", "inst_buy_imbalance_over_leg",
                      "inst_buy_imbalance_a", "inst_buy_imbalance_b"]].copy()
        keep["pair"] = pair
        keep = keep[pd.to_datetime(keep["timestamp"], utc=True) >= HELDOUT_BOUNDARY]
        keep.to_parquet(os.path.join(FEAT_DIR, f"{pair}.parquet"), index=False)
        n_evt = int((keep["spread_z"].abs() >= 2).sum())
        print(f"  [ok] {pair}: {len(keep)} held-out bars, {n_evt} with |z|>=2 "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)
    print(f"DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
