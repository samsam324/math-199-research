"""Generate the paper's two figures into paper/.
  fig_freq_invariance.pdf -- Kalman screen passes real and placebo alike at every frequency.
  fig_rollingz_noise.pdf  -- a rolling z-score manufactures reversion on a pure random walk.
Run: python scratch/make_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\jackw\Desktop\math-199-research"
OUT = ROOT  # repo root
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight"})

# Figure fig_freq_invariance.pdf is now generated (reshaped) by make_figures_v2.py

# ---- Figure 2: rolling-z reversion on a pure random walk ----
rng = np.random.default_rng(3)
n, lb = 1200, 120
rw = np.cumsum(rng.normal(0, 1, n))
s = pd.Series(rw)
rmean = s.rolling(lb).mean(); rstd = s.rolling(lb).std()
z = ((s - rmean) / rstd).values
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 3.8), sharex=True)
a1.plot(rw, color="#2b6cb0", lw=0.9, label="random walk")
a1.plot(rmean.values, color="#dd6b20", lw=1.0, label="trailing mean (120)")
a1.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5)); a1.set_ylabel("level")
a2.plot(z, color="#2b6cb0", lw=0.8)
for lvl in (2, -2):
    a2.axhline(lvl, color="#999999", lw=0.6, ls="--")
a2.axhline(0, color="#cccccc", lw=0.5)
ent = np.where(np.abs(z) > 2)[0]
a2.scatter(ent, z[ent], s=5, color="#dd6b20", zorder=3, label="$|z|>2$ entries")
a2.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
a2.set_ylabel("rolling $z$"); a2.set_xlabel("time (bars)")
fig.savefig(os.path.join(OUT, "fig_rollingz_noise.pdf")); plt.close(fig)

print("wrote fig_rollingz_noise.pdf")
