"""
Robustness on the persistence finding (reads scratch/persistence_pairs.csv).

R1. SHUFFLE PLACEBO: within each split, permute the train-metric->OOS link
    (shuffle pair labels on the train metric only). Rank-corr & quintile spread
    must collapse to ~0. Confirms the real result isn't a bug.

R2. Is it just spread-VOLATILITY persistence? Control for OOS spread vol.
    The mechanical floor is matched to OOS increment vol, so 'excess' is already
    vol-neutralized; but check: does train kappa still predict OOS excess AFTER
    partialling out OOS floor magnitude (a proxy for OOS spread vol regime)?
    -> partial Spearman(train_kappa, oos_excess | oos_floor).

R3. Sign of bottom quintile: report how often OOS excess < 0 (worse than a
    random walk) for bottom-quintile (in-sample non-reverting) pairs.

R4. Persistence of the SELECTED set in raw (non-excess) terms too, so the
    reader can see the floor subtraction isn't manufacturing the gap.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(11)
df = pd.read_csv(ROOT / "scratch/persistence_pairs.csv")
splits = sorted(df["split"].unique())


def quintile_spread_per_split(d, metric, asc):
    d = d.dropna(subset=[metric, "oos_excess"])
    if len(d) < 10:
        return None
    nq = max(1, len(d) // 5)
    if asc:
        top = d.nsmallest(nq, metric); bot = d.nlargest(nq, metric)
    else:
        top = d.nlargest(nq, metric); bot = d.nsmallest(nq, metric)
    return np.mean(top["oos_excess"]) - np.mean(bot["oos_excess"])


print("=" * 74)
print("ROBUSTNESS CHECKS ON OOS REVERSION PERSISTENCE")
print("=" * 74)

# ---- R1 shuffle placebo ----
print("\nR1. SHUFFLE PLACEBO (break train->OOS link; expect rho~0, spread~0)")
real_rho = []; shuf_rho = []; real_sp = []; shuf_sp = []
for sp in splits:
    d = df[df["split"] == sp].dropna(subset=["tr_kappa", "oos_excess"]).copy()
    real_rho.append(spearmanr(d["tr_kappa"], d["oos_excess"])[0])
    real_sp.append(quintile_spread_per_split(d, "tr_kappa", asc=False))
    d2 = d.copy(); d2["tr_kappa"] = RNG.permutation(d2["tr_kappa"].to_numpy())
    shuf_rho.append(spearmanr(d2["tr_kappa"], d2["oos_excess"])[0])
    shuf_sp.append(quintile_spread_per_split(d2, "tr_kappa", asc=False))
print(f"  real  : mean rho={np.mean(real_rho):+.3f}   mean quintile-spread={np.mean(real_sp):+.4f}z")
print(f"  shuffled: mean rho={np.mean(shuf_rho):+.3f}   mean quintile-spread={np.mean(shuf_sp):+.4f}z")

# ---- R2 partial spearman controlling for OOS floor (vol regime proxy) ----
print("\nR2. Partial Spearman(train_kappa, OOS excess | OOS floor magnitude)")
print("    (does selection survive after removing OOS spread-vol/floor regime?)")
def partial_spearman(d, a, b, c):
    d = d.dropna(subset=[a, b, c])
    ra, rb, rc = rankdata(d[a]), rankdata(d[b]), rankdata(d[c])
    def resid(y, x):
        x1 = np.c_[np.ones_like(x), x]
        beta = np.linalg.lstsq(x1, y, rcond=None)[0]
        return y - x1 @ beta
    return np.corrcoef(resid(ra, rc), resid(rb, rc))[0, 1]
raw_r = []; par_r = []
for sp in splits:
    d = df[df["split"] == sp]
    raw_r.append(spearmanr(d["tr_kappa"], d["oos_excess"], nan_policy="omit")[0])
    par_r.append(partial_spearman(d, "tr_kappa", "oos_excess", "oos_floor"))
print(f"  raw     mean rho(train_kappa, oos_excess)        = {np.mean(raw_r):+.3f}")
print(f"  partial mean rho(... | oos_floor)               = {np.mean(par_r):+.3f}")

# Also control with TRAIN-side vol (tr spread vol via tr floor not stored) -> use
# train kappa vs train halflife already monotone; instead control OOS excess pred
# by TRAIN excess for the OTHER metrics: does kappa add beyond in-sample excess?
print("\nR2b. Partial Spearman(train_kappa, OOS excess | TRAIN in-sample excess)")
par2 = []
for sp in splits:
    d = df[df["split"] == sp]
    par2.append(partial_spearman(d, "tr_kappa", "oos_excess", "tr_excess"))
print(f"  partial mean rho(train_kappa, oos_excess | tr_excess) = {np.mean(par2):+.3f}")

# ---- R3 sign of bottom quintile ----
print("\nR3. Bottom-quintile (in-sample LEAST reverting) OOS behaviour")
bot_means = []; frac_neg = []
for sp in splits:
    d = df[df["split"] == sp].dropna(subset=["tr_kappa", "oos_excess"])
    nq = max(1, len(d) // 5)
    bot = d.nsmallest(nq, "tr_kappa")
    bot_means.append(np.mean(bot["oos_excess"]))
    frac_neg.append(np.mean(bot["oos_excess"] < 0))
print(f"  bottom-quintile mean OOS excess = {np.mean(bot_means):+.4f}z")
print(f"  fraction of bottom-quintile pairs with OOS excess<0 (worse than RW) = {np.mean(frac_neg):.2%}")
top_means = []
for sp in splits:
    d = df[df["split"] == sp].dropna(subset=["tr_kappa", "oos_excess"])
    nq = max(1, len(d) // 5)
    top_means.append(np.mean(d.nlargest(nq, "tr_kappa")["oos_excess"]))
print(f"  top-quintile    mean OOS excess = {np.mean(top_means):+.4f}z")

# ---- R4 raw (non-excess) OOS reversion for the selected set ----
print("\nR4. RAW (NOT floor-subtracted) OOS reversion, top vs bottom quintile by train kappa")
top_raw=[]; bot_raw=[]; top_fl=[]; bot_fl=[]
for sp in splits:
    d = df[df["split"] == sp].dropna(subset=["tr_kappa", "oos_obs", "oos_floor"])
    nq = max(1, len(d) // 5)
    t = d.nlargest(nq, "tr_kappa"); b = d.nsmallest(nq, "tr_kappa")
    top_raw.append(np.mean(t["oos_obs"])); bot_raw.append(np.mean(b["oos_obs"]))
    top_fl.append(np.mean(t["oos_floor"])); bot_fl.append(np.mean(b["oos_floor"]))
print(f"  top   : raw OOS revert={np.mean(top_raw):+.4f}z   its floor={np.mean(top_fl):+.4f}z")
print(f"  bottom: raw OOS revert={np.mean(bot_raw):+.4f}z   its floor={np.mean(bot_fl):+.4f}z")
print(f"  raw spread (top-bot, NOT floor-subtracted) = {np.mean(top_raw)-np.mean(bot_raw):+.4f}z")

# ---- OOS kappa persistence too (does train kappa predict OOS kappa?) ----
print("\nR5. Does train kappa predict OOS kappa (second outcome measure)?")
kk = []
for sp in splits:
    d = df[df["split"] == sp]
    kk.append(spearmanr(d["tr_kappa"], d["oos_kappa"], nan_policy="omit")[0])
print(f"  mean Spearman(train_kappa, OOS_kappa) = {np.mean(kk):+.3f}  per-split={[round(x,2) for x in kk]}")
print("\nDone.")
