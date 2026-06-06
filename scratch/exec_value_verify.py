"""
Verify the execution-value null (iteration 9). The subagent found the L3 post/cross
rule is *significantly* anti-selective (z~2.6 WORSE than a random placebo). But a
significantly anti-selective signal is INFORMATIVE -- flipping its sign should beat
the placebo. The subagent never tested: (a) the sign-flipped rule, (b) an ORACLE
upper bound on how much ANY post/cross selection rule could save, (c) the raw
correlation of the signal with per-order (passive - aggressive) cost.

Uses the saved per-order costs in scratch/exec_value_orders.csv (no re-simulation).

Decision logic recap from exec_value.py:
  l3 cost = naive_passive_cost  if l3_posted else cost_agg
So:
  l3_flip cost = naive_passive_cost if (NOT l3_posted) else cost_agg
  oracle  cost = min(naive_passive_cost, cost_agg)         (perfect foresight)
  antiora cost = max(naive_passive_cost, cost_agg)         (worst foresight)

Run synchronously:  python scratch/exec_value_verify.py
"""
import os
import numpy as np
import pandas as pd

ROOT = r"C:\Users\jackw\Desktop\math-199-research"
df = pd.read_csv(os.path.join(ROOT, "scratch", "exec_value_orders.csv"))

NAIVE = "naive_H30_back_cost"
AGG = "cost_agg"
L3 = "l3_H30_back_cost"

# signed signal: positive = "favorable for the order's direction"
# book_ofi as stored is in absolute (buy-positive) terms; align to order side so a
# higher value always means "more upward pressure relative to the trade we must do".
df["sig_dir"] = df["book_ofi"] * np.where(df["side"] == "buy", 1.0, -1.0)
df["passive_minus_agg"] = df[NAIVE] - df[AGG]

rng = np.random.default_rng(7)


def placebo_mean_sd(sub, p_post, n=2000):
    naive_c = sub[NAIVE].to_numpy(); agg_c = sub[AGG].to_numpy()
    sims = np.empty(n)
    for k in range(n):
        mask = rng.random(len(sub)) < p_post
        sims[k] = np.where(mask, naive_c, agg_c).mean()
    return float(sims.mean()), float(sims.std())


print("=" * 100)
print("EXECUTION-VALUE VERIFICATION  (H=30s, back-of-queue; cost in bps vs arrival mid, lower=better)")
print("=" * 100)

for size in [10000, 50000, "POOLED"]:
    sub = df if size == "POOLED" else df[df["notional"] == size]
    if len(sub) == 0:
        continue
    agg = sub[AGG].mean()
    naive = sub[NAIVE].mean()
    l3 = sub[L3].mean()
    posted = sub["l3_posted"].astype(bool)
    l3_flip = np.where(~posted, sub[NAIVE], sub[AGG]).mean()
    oracle = np.minimum(sub[NAIVE], sub[AGG]).mean()
    antior = np.maximum(sub[NAIVE], sub[AGG]).mean()
    p_post = posted.mean()
    rmean, rsd = placebo_mean_sd(sub, p_post)
    z_l3 = (l3 - rmean) / rsd if rsd > 0 else np.nan
    z_flip = (l3_flip - rmean) / rsd if rsd > 0 else np.nan
    # correlation of the directional signal with (passive-agg): does the signal carry
    # ANY info about which orders are cheaper to post?
    c = sub[["sig_dir", "passive_minus_agg"]].dropna()
    corr = np.corrcoef(c["sig_dir"], c["passive_minus_agg"])[0, 1] if len(c) > 30 else np.nan

    name = "POOLED" if size == "POOLED" else f"${int(size):,}"
    print(f"\n--- {name}  (n={len(sub)}, L3 post-rate={p_post*100:.0f}%) ---")
    print(f"  AGGRESSIVE (always cross) : {agg:7.3f}   <- the baseline to beat")
    print(f"  NAIVE passive (always)    : {naive:7.3f}")
    print(f"  L3 as-built               : {l3:7.3f}   (z vs random placebo {z_l3:+.2f})")
    print(f"  L3 SIGN-FLIPPED           : {l3_flip:7.3f}   (z vs random placebo {z_flip:+.2f})")
    print(f"  random-same-rate placebo  : {rmean:7.3f}  (sd {rsd:.3f})")
    print(f"  ORACLE  min(passive,agg)  : {oracle:7.3f}   <- best ANY post/cross rule could do")
    print(f"  anti-oracle max(.,.)      : {antior:7.3f}")
    print(f"  >> oracle saves vs AGG    : {agg - oracle:+.3f} bps  "
          f"(if ~0, NO selection rule can beat just crossing)")
    print(f"  >> signal corr(sig_dir, passive-agg) = {corr:+.4f}  "
          f"(|corr|~0 => signal carries no selection info)")

print("\n" + "=" * 100)
print("READING: If ORACLE barely beats AGGRESSIVE, the 'no execution edge' null is robust to")
print("the signal's SIGN -- no post/cross rule (L3, flipped, or perfect) can beat crossing,")
print("because the majors' spread is ~1 tick and the unfilled-tail chase dominates any capture.")
print("=" * 100)

# ---- Is the oracle opportunity capturable by ANY simple L2 feature, or is it the
#      unforecastable future? Correlate candidate features with per-order (passive-agg)
#      advantage and with the naive fill outcome. ----
print("\n" + "=" * 100)
print("CAPTURABILITY: corr of candidate L2 features with per-order execution advantage")
print("  target1 = passive_minus_agg (negative => posting was the cheaper choice)")
print("  target2 = naive fill (1=filled within 30s).  |corr|~0 => feature can't time execution.")
print("=" * 100)
df["naive_fill"] = df["naive_H30_back_fill"].astype(float)
feats = {
    "book_ofi (dir)": "sig_dir",
    "cancel_imb (dir)": None,   # build below
    "spread_bps": "spread_bps",
    "|book_ofi|": None,
}
df["cancel_dir"] = df["cancel_imb"] * np.where(df["side"] == "buy", 1.0, -1.0)
df["abs_ofi"] = df["book_ofi"].abs()
colmap = {"book_ofi (dir)": "sig_dir", "cancel_imb (dir)": "cancel_dir",
          "spread_bps": "spread_bps", "|book_ofi|": "abs_ofi"}
for name, col in colmap.items():
    c = df[[col, "passive_minus_agg", "naive_fill"]].dropna()
    r1 = np.corrcoef(c[col], c["passive_minus_agg"])[0, 1]
    r2 = np.corrcoef(c[col], c["naive_fill"])[0, 1]
    print(f"  {name:<18} corr(.,passive-agg)={r1:+.4f}   corr(.,fill)={r2:+.4f}")
