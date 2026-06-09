"""
H1 task — STEP 3: the LOCKED one-shot evaluation. Run ONCE. No re-tuning.

Tests the pre-registered hypothesis: a NEGATIVE Spearman correlation between
  X = over-leg institutional buy-imbalance, trailing W=60-bar mean
  Y = target_spread_change(t, t+60) = spread(t+60) - spread(t)
on the held-out tail (timestamp >= 2024-10-19 19:12 UTC), bars with |spread_z|>=2.

Statistical methodology incorporates an adversarial-audit (2026-06-08) of the naive
"pooled ranks + Newey-West lag 60" plan, which UNDERSTATED the SE (overstated
significance):
  (1) Y is an overlapping 60-ahead target and X a 60-bar mean, so the rank-product
      series is autocorrelated well beyond 60 lags; Bartlett-kernel HAC at lag 60
      truncates too early -> SE biased DOWN. We report HAC at lags {60(pre-reg),
      120, 240} and treat the larger-lag p as the honest one.
  (2) HAC is computed PER PAIR and combined (pairs independent), not on one
      concatenated series (which fabricates cross-pair autocovariances).
        rho = (1/N) Sum_i Sum_t p_i(t);  Var(rho) = (1/N^2) Sum_i n_i * LRV_i(lag)
      where p_i(t) = standardized-rank-product within pair i, LRV_i = Newey-West.
  (3) PRIMARY estimand = WITHIN-pair standardized ranks (the hypothesis is about the
      monotonic relation WITHIN a pair; pooled ranks mix per-pair Y-scales =>
      Simpson's-paradox risk). Pooled ranks (literal pre-reg) reported as secondary.
  (4) A block bootstrap (block=300, per pair) gives a tie/overlap-robust SE cross-
      check; we also report the fraction of tied/zero X.

Pass = one-sided p < 0.05 in the predicted (negative) direction. Headline verdict
uses the within-pair estimand at the conservative lag (240) + the bootstrap; the
pre-registered (pooled, lag 60) value is reported for faithfulness.

Run ONCE:  python scratch/h1_test.py
"""
import os, sys, glob
import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"C:\Users\jackw\Desktop\math-199-research"
sys.path.insert(0, os.path.join(ROOT, "phase2_l2"))
from src.statistics import hac_long_run_var   # noqa: E402

HELDOUT_BOUNDARY = pd.Timestamp("2024-10-19T19:12:00Z")
LAGS = [60, 120, 240]
FEAT_DIR = os.path.join(ROOT, "scratch", "h1_feat")


def std_ranks(v):
    r = stats.rankdata(v)
    sd = r.std()
    return (r - r.mean()) / sd if sd > 0 else np.zeros_like(r)


def combined_hac(prod_by_pair, lag):
    """rho (n-weighted) + per-pair-combined Newey-West SE + one-sided (negative) p."""
    N = sum(len(p) for p in prod_by_pair)
    rho = sum(p.sum() for p in prod_by_pair) / N
    var = sum(len(p) * hac_long_run_var(p, min(lag, len(p) - 1)) for p in prod_by_pair) / (N ** 2)
    se = float(np.sqrt(var))
    t = rho / se if se > 0 else np.nan
    return rho, se, t, float(stats.norm.cdf(t)), N


def block_boot(prod_by_pair, block=300, nboot=2000, seed=12345):
    rng = np.random.default_rng(seed)
    N = sum(len(p) for p in prod_by_pair)
    boot = np.empty(nboot)
    for b in range(nboot):
        tot = 0.0
        for p in prod_by_pair:
            n = len(p); nb = int(np.ceil(n / block))
            starts = rng.integers(0, max(1, n - block + 1), size=nb)
            samp = np.concatenate([p[s:s + block] for s in starts])[:n]
            tot += samp.sum()
        boot[b] = tot / N
    rho = sum(p.sum() for p in prod_by_pair) / N
    se = float(boot.std(ddof=1))
    t = rho / se if se > 0 else np.nan
    return rho, se, float(stats.norm.cdf(t))


def products(s, xcol, within):
    """Per-pair standardized-rank-product arrays for X=xcol vs Y=target."""
    out = []
    if within:
        for _, g in s.groupby("pair"):
            if len(g) < 30:
                continue
            out.append(std_ranks(g[xcol].to_numpy()) * std_ranks(g["target_spread_change"].to_numpy()))
    else:  # pooled ranks, then split by pair for the per-pair HAC combination
        zx = std_ranks(s[xcol].to_numpy()); zy = std_ranks(s["target_spread_change"].to_numpy())
        s2 = s.assign(_p=zx * zy)
        for _, g in s2.groupby("pair"):
            if len(g) < 30:
                continue
            out.append(g["_p"].to_numpy())
    return out


def report(s, xcol, label):
    print(f"\n----- {label}  (X = {xcol}) -----")
    for within, tag in [(True, "WITHIN-pair ranks (primary)"), (False, "pooled ranks (pre-reg literal)")]:
        pb = products(s, xcol, within)
        rho = sum(p.sum() for p in pb) / sum(len(p) for p in pb)
        line = f"  {tag:<34} rho={rho:+.5f} | HAC one-sided p @lag: "
        for L in LAGS:
            _, _, t, p, N = combined_hac(pb, L)
            line += f"L{L}={p:.3g}(t{t:+.2f}) "
        brho, bse, bp = block_boot(pb)
        line += f"| block-boot p={bp:.3g}"
        print(line)


def main():
    files = sorted(glob.glob(os.path.join(FEAT_DIR, "*.parquet")))
    frames = []
    for f in files:
        d = pd.read_parquet(f); d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    m = (df["timestamp"] >= HELDOUT_BOUNDARY) & (df["spread_z"].abs() >= 2.0) \
        & df["X_inst_over_leg_W60"].notna() & df["target_spread_change"].notna()
    s = df[m].sort_values(["pair", "timestamp"]).reset_index(drop=True)

    print("=" * 88)
    print("H1 LOCKED ONE-SHOT TEST  (audit-corrected HAC: per-pair combine, lags 60/120/240, bootstrap)")
    print(f"held-out boundary: {HELDOUT_BOUNDARY} | pairs={df['pair'].nunique()} | pooled |z|>=2 obs={len(s)}")
    print("per-pair obs:")
    print(s.groupby("pair").size().to_string())
    xz = s["X_inst_over_leg_W60"].to_numpy()
    print(f"\nX diagnostics (tie/sparsity): frac X==0 = {(np.abs(xz)<1e-12).mean():.3f} | "
          f"unique X / n = {len(np.unique(xz))}/{len(xz)} = {len(np.unique(xz))/len(xz):.3f}")
    print("=" * 88)

    print("\n[per-pair Spearman breakdown — the raw within-pair signal]")
    for pair, g in s.groupby("pair"):
        if len(g) >= 30:
            rr, pp = stats.spearmanr(g["X_inst_over_leg_W60"], g["target_spread_change"])
            print(f"  {pair:<22} n={len(g):>6}  rho={rr:+.4f} (iid two-sided p={pp:.3g})")

    report(s, "X_inst_over_leg_W60", "PRIMARY: over-leg via spread_z")
    if "X_rawspread_W60" in s.columns:
        report(s, "X_rawspread_W60", "ROBUSTNESS: over-leg via raw spread (collaborator's code, proper alpha)")

    print("\n" + "=" * 88)
    print("VERDICT (honest = within-pair, conservative lag 240 + bootstrap; pre-reg = pooled lag 60):")
    pb = products(s, "X_inst_over_leg_W60", within=True)
    _, _, t240, p240, _ = combined_hac(pb, 240)
    _, _, bp = block_boot(pb)
    pbp = products(s, "X_inst_over_leg_W60", within=False)
    _, _, t60, p60, _ = combined_hac(pbp, 60)
    rho_w = sum(p.sum() for p in pb) / sum(len(p) for p in pb)
    print(f"  HONEST  : rho={rho_w:+.5f}  within-pair HAC lag240 p={p240:.4g}  block-boot p={bp:.4g}  "
          f"-> {'PASS' if (rho_w<0 and max(p240,bp)<0.05) else 'FAIL'}")
    print(f"  PRE-REG : pooled HAC lag60 one-sided p={p60:.4g}  -> "
          f"{'PASS' if p60<0.05 else 'FAIL'} (lag 60 understates SE; treat as optimistic lower bound on p)")
    print("=" * 88)


if __name__ == "__main__":
    main()
