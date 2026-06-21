"""Universe-wide order-flow breadth figure for the proposed paper.

Reads scratch/ofi_broad_50sym.csv (the 50-symbol book-OFI run) and plots, per
symbol, contemporaneous incremental book-OFI R^2 against the 1-second-ahead
forecast R^2. Two panels: left on a shared 0-50% axis (forecast power collapses
to the floor), right with the forecast axis magnified so the tiny, flat spread
is visible. Message: across the full top 50, order flow is informative
contemporaneously and almost unpredictive one second out.

Run: python scratch/make_fig_ofi_breadth.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\jackw\Desktop\math-199-research"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "figure.dpi": 130})
BLUE, RED, GRAY = "#2b6cb0", "#c53030", "#a0aec0"

d = pd.read_csv(os.path.join(ROOT, "scratch", "ofi_broad_50sym.csv"))
x = d["contemp_incr_book"].values * 100.0      # 3-48%
y = d["pred_joint_h1"].values * 100.0          # 1s-ahead, all < 0.5%
majors = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"}
is_m = d["sym"].isin(majors).values
print(f"x (contemp) range {x.min():.1f}-{x.max():.1f}%  median {np.median(x):.1f}%")
print(f"y (1s fwd)  range {y.min():.3f}-{y.max():.3f}%  median {np.median(y):.3f}%")
print(f"corr(contemp, 1s-fwd) = {np.corrcoef(x, y)[0,1]:+.3f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
for ax, zoom in ((a1, False), (a2, True)):
    ax.scatter(x[~is_m], y[~is_m], s=24, color=GRAY, edgecolors="none",
               label="top 50 (other 46)", zorder=2)
    ax.scatter(x[is_m], y[is_m], s=46, color=RED, edgecolors="white", linewidths=0.6,
               label="event-level majors", zorder=3)
    ax.set_xlabel(r"contemporaneous incremental book-OFI $R^2$ (%)")
    ax.set_xlim(-1, 50)
    if not zoom:
        ax.set_ylim(-1, 50)
        ax.set_ylabel(r"1-second-ahead forecast $R^2$ (%)")
        ax.set_title("same axis: forecast power sits on the floor", fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    else:
        ax.set_ylim(0, 0.55)
        ax.set_ylabel(r"1-second-ahead forecast $R^2$ (%)")
        ax.set_title(r"forecast axis magnified ($\sim$$100\times$)", fontsize=9, loc="left")
        for xi, yi, s, m in zip(x, y, d["sym"], is_m):
            if m:
                ax.annotate(s.replace("USDT", ""), (xi, yi), fontsize=7, color=RED,
                            xytext=(3, 2), textcoords="offset points")
fig.tight_layout()
out = os.path.join(ROOT, "fig_ofi_breadth.pdf")
fig.savefig(out)
print("wrote", out)
