"""
PART 2 — Does the mean-reversion premise hold WITH POWER on genuinely
cointegrated pairs (clean Engle-Granger OOS selection from Part 1)?

For each selected pair, on the FULL multi-year hourly history:
  - static OLS hedge -> log-price spread
  - rolling z-score (lookback=240h), no look-ahead
  - events: |z|>2 crossings with a cooldown (>= horizon) to avoid overlap
  - outcome: signed reversion = -sign(z_entry) * (z_{t+h} - z_entry)
            i.e. positive = spread moved back toward 0 over next h hours
  - horizons h in {24, 48, 72}

Inference:
  - pool all events; mean reversion per horizon
  - pair-CLUSTERED SE (cluster bootstrap over pairs) -> the test that killed
    the premise in the L2 log
  - also a block bootstrap over events
  - Bonferroni across the 3 horizons

Contrast: same test on correlation-fallback pairs (top return-correlation,
NOT cointegrated) to show the foundation matters.
"""
from __future__ import annotations
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

RNG = np.random.default_rng(7)
UNI = [l.strip() for l in open(ROOT / "data/l2_universe_top50.txt") if l.strip()]
LOOKBACK = 240          # rolling z window (hours)
ZTHR = 2.0
HORIZONS = [24, 48, 72]
COOLDOWN = 72           # min hours between accepted events per pair (>= max horizon)


def load_close(sym):
    df = pd.read_parquet(ROOT / f"data/spot_1h/{sym}.parquet")
    return df["close"]


def build_full_panel(symbols):
    cols = {s: load_close(s) for s in symbols}
    panel = pd.DataFrame(cols).sort_index()
    panel = panel.dropna(how="any")
    return panel


def pair_events(y, x):
    """y,x are aligned log-price arrays for the full history. Returns list of
    (z_entry, reversion_by_horizon dict)."""
    # static OLS hedge over full history (foundation premise = static spread reverts)
    ols = OLS(y, add_constant(x)).fit()
    a, b = float(ols.params[0]), float(ols.params[1])
    spread = y - (a + b * x)
    s = pd.Series(spread)
    mu = s.rolling(LOOKBACK).mean()
    sd = s.rolling(LOOKBACK).std()
    z = (s - mu) / sd
    zv = z.to_numpy()
    n = len(zv)
    events = []
    last = -10**9
    maxh = max(HORIZONS)
    for t in range(LOOKBACK, n - maxh):
        zt = zv[t]
        if not np.isfinite(zt) or abs(zt) <= ZTHR:
            continue
        if t - last < COOLDOWN:
            continue
        last = t
        rev = {}
        for h in HORIZONS:
            zf = zv[t + h]
            if np.isfinite(zf):
                rev[h] = -np.sign(zt) * (zf - zt)  # positive = reverted toward 0
            else:
                rev[h] = np.nan
        events.append((zt, rev))
    return events


def collect(pairs, panel):
    logp = np.log(panel)
    rows = []
    for a, b in pairs:
        if a not in panel.columns or b not in panel.columns:
            continue
        y = logp[a].to_numpy(); x = logp[b].to_numpy()
        evs = pair_events(y, x)
        for zt, rev in evs:
            row = {"pair": f"{a}_{b}", "z_entry": zt}
            for h in HORIZONS:
                row[f"rev_{h}"] = rev[h]
            rows.append(row)
    return pd.DataFrame(rows)


def cluster_bootstrap_mean(df, col, n_boot=2000):
    """Pair-clustered bootstrap: resample pairs with replacement, take all their events."""
    pairs = df["pair"].unique()
    groups = {p: df[df["pair"] == p][col].dropna().to_numpy() for p in pairs}
    point = np.nanmean(df[col].to_numpy())
    boots = np.empty(n_boot)
    for i in range(n_boot):
        samp = RNG.choice(pairs, size=len(pairs), replace=True)
        vals = np.concatenate([groups[p] for p in samp if len(groups[p])])
        boots[i] = np.mean(vals) if len(vals) else np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    # two-sided p: fraction of boots on opposite side of 0 reflected
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return point, lo, hi, p, len(pairs)


def block_bootstrap_mean(df, col, block=24, n_boot=2000):
    x = df[col].dropna().to_numpy()
    n = len(x)
    if n < block * 2:
        return np.nan, np.nan, np.nan, np.nan
    nblocks = n // block
    point = np.mean(x)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        starts = RNG.integers(0, n - block, size=nblocks)
        samp = np.concatenate([x[s:s+block] for s in starts])
        boots[i] = np.mean(samp)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    return point, lo, hi, p


def report(label, df):
    print(f"\n=== {label} ===", flush=True)
    if df.empty:
        print("  no events", flush=True)
        return
    n_pairs = df["pair"].nunique()
    print(f"  pairs={n_pairs}  total events={len(df)}", flush=True)
    nH = len(HORIZONS)
    for h in HORIZONS:
        col = f"rev_{h}"
        pt, lo, hi, pcl, npairs = cluster_bootstrap_mean(df, col)
        _, blo, bhi, pbl = block_bootstrap_mean(df, col)
        nev = int(df[col].notna().sum())
        bonf = min(1.0, pcl * nH)
        sig = "SIG" if bonf < 0.05 else "ns"
        print(f"  h={h}h  meanRev={pt:+.4f}z  pair-cluster95%CI[{lo:+.4f},{hi:+.4f}] p={pcl:.3f} (Bonf {bonf:.3f} {sig})"
              f"  | block95%CI[{blo:+.4f},{bhi:+.4f}] p={pbl:.3f}  n_ev={nev}", flush=True)


def main():
    # Clean cointegrated pairs from Part 1
    clean = pd.read_csv(ROOT / "scratch/clean_coint_pairs.csv")
    clean_sig = clean[clean["eg_pvalue"] < 0.05]
    clean_pairs = list(clean_sig[["sym_a", "sym_b"]].itertuples(index=False, name=None))
    print(f"Clean-cointegrated pairs (EG OOS p<0.05): {len(clean_pairs)}", flush=True)

    # Correlation-fallback pairs: top return-correlation among top-50, NOT in clean set
    panel = build_full_panel(UNI)
    print(f"Full panel: {panel.shape[0]} bars x {panel.shape[1]} symbols, "
          f"{panel.index.min()} .. {panel.index.max()}", flush=True)
    logp = np.log(panel)
    rets = logp.diff().dropna()
    corr = rets.corr()
    clean_set = {frozenset(p) for p in clean_pairs}
    corr_rows = []
    for a, b in combinations(panel.columns, 2):
        if frozenset((a, b)) in clean_set:
            continue
        c = float(corr.loc[a, b])
        if np.isfinite(c):
            corr_rows.append((a, b, c))
    corr_rows.sort(key=lambda t: t[2], reverse=True)
    n_take = max(len(clean_pairs), 20)
    corr_pairs = [(a, b) for a, b, _ in corr_rows[:n_take]]
    print(f"Correlation-fallback pairs (top corr, non-clean): {len(corr_pairs)}", flush=True)

    # restrict to symbols present
    clean_pairs = [(a, b) for a, b in clean_pairs if a in panel.columns and b in panel.columns]

    df_clean = collect(clean_pairs, panel)
    df_corr = collect(corr_pairs, panel)

    report("CLEAN COINTEGRATED pairs — reversion premise (powered)", df_clean)
    report("CORRELATION-FALLBACK pairs — reversion premise", df_corr)

    df_clean.to_csv(ROOT / "scratch/part2_clean_events.csv", index=False)
    df_corr.to_csv(ROOT / "scratch/part2_corr_events.csv", index=False)
    print("\nSaved event sets to scratch/part2_*.csv", flush=True)


if __name__ == "__main__":
    main()
