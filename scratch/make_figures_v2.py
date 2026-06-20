"""Generate the paper v2 figures from the committed CSVs and logs.

Outputs (all into the repo root, vector PDF):
  fig_kalman_innovations_white.pdf   Fig 1  live Kalman run, BTC/ETH + RW placebo
  fig_poscontrol_vs_negcontrol.pdf   Fig 2  scratch/kalman_positive_control.csv
  fig_reversion_persistence_scatter  Fig 3  scratch/persistence_pairs.csv
  fig_circuit_breaker_retention.pdf  Fig 4  scratch/survivorship_adjusted_sharpe.csv
  fig_forced_collapse_perpair.pdf    Fig 5  scratch/forced_collapse.csv
  fig_ofi_decay.pdf                  Fig 6  numbers parsed from scratch/book_ofi_2024.log
  fig_microcap_adv.pdf               Fig 7  scratch/microcap_adv.csv
  fig_hac_inflation.pdf              Fig 8  docs/hac_sharpe_per_split.csv
  fig_cancellation_share.pdf         Fig 9  numbers from scratch/book_ofi_2024.log
  fig_freq_invariance.pdf  RESHAPED  scratch/timeframe_kalman_artifact.csv (placebo-gap view)

Run: python scratch/make_figures_v2.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

ROOT = r"C:\Users\jackw\Desktop\math-199-research"
OUT = ROOT  # repo root
sys.path.insert(0, ROOT)

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "figure.dpi": 130,
})

BLUE, ORANGE, GRAY, GREEN, RED = "#2b6cb0", "#dd6b20", "#a0aec0", "#2f855a", "#c53030"
SYM_C = {"BTCUSDT": "#2b6cb0", "ETHUSDT": "#dd6b20", "SOLUSDT": "#2f855a", "AVAXUSDT": "#c53030"}


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", os.path.relpath(p, ROOT))


# ---------------------------------------------------------------- Fig 1
def fig_kalman_innovations():
    """A Kalman filter whitens its own innovations: a real pair and an independent
    random walk pair give visually indistinguishable innovation series with flat ACFs."""
    from src.kalman_hedge import fit_kalman_mle, kalman_forward_residuals
    NTR, NTE = 1500, 500
    btc = pd.read_parquet(os.path.join(ROOT, "scratch", "cache_hourly", "BTCUSDT.parquet"))["mid"]
    eth = pd.read_parquet(os.path.join(ROOT, "scratch", "cache_hourly", "ETHUSDT.parquet"))["mid"]
    panel = pd.concat({"y": np.log(btc), "x": np.log(eth)}, axis=1).dropna()
    y = panel["y"].to_numpy()[: NTR + NTE]
    x = panel["x"].to_numpy()[: NTR + NTE]
    fit = fit_kalman_mle(y[:NTR], x[:NTR])
    _, _, real_innov = kalman_forward_residuals(y[NTR:NTR + NTE], x[NTR:NTR + NTE], fit)

    rng = np.random.default_rng(11)
    yr = np.cumsum(rng.normal(0, 0.02, NTR + NTE))
    xr = np.cumsum(rng.normal(0, 0.02, NTR + NTE))
    fit_r = fit_kalman_mle(yr[:NTR], xr[:NTR])
    _, _, rw_innov = kalman_forward_residuals(yr[NTR:NTR + NTE], xr[NTR:NTR + NTE], fit_r)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.6),
                             gridspec_kw={"width_ratios": [2.1, 1]})
    rows = [("Real BTC/ETH pair", real_innov, BLUE), ("Independent random walk pair", rw_innov, RED)]
    for i, (label, innov, c) in enumerate(rows):
        innov = np.asarray(innov, float)
        innov = innov[np.isfinite(innov)]
        z = (innov - innov.mean()) / innov.std()
        axes[i, 0].plot(z, lw=0.5, color=c)
        axes[i, 0].set_title(f"{label}: Kalman innovations", fontsize=9, loc="left")
        axes[i, 0].set_ylabel("std. innovation")
        axes[i, 0].axhline(0, color="#cccccc", lw=0.5)
        acf = sm.tsa.acf(innov, nlags=40, fft=True)
        ci = 1.96 / np.sqrt(len(innov))
        axes[i, 1].vlines(range(len(acf)), 0, acf, color=c, lw=1.0)
        axes[i, 1].plot(range(len(acf)), acf, ".", color=c, ms=2.5)
        axes[i, 1].axhline(ci, ls="--", color="gray", lw=0.7)
        axes[i, 1].axhline(-ci, ls="--", color="gray", lw=0.7)
        axes[i, 1].axhline(0, color="#888888", lw=0.6)
        axes[i, 1].set_title("sample ACF", fontsize=9, loc="left")
        axes[i, 1].set_ylim(-0.15, 1.02)
    axes[1, 0].set_xlabel("test-window bar")
    axes[1, 1].set_xlabel("lag")
    fig.tight_layout()
    save(fig, "fig_kalman_innovations_white.pdf")


# ---------------------------------------------------------------- Fig 2
def fig_poscontrol():
    df = pd.read_csv(os.path.join(ROOT, "scratch", "kalman_positive_control.csv"))
    pos = df[df.group.str.startswith("POS")].iloc[0]
    neg = df[df.group.str.startswith("NEG")].iloc[0]
    tests = ["kalman_adf_pass", "clean_eg_pass", "static_adf_pass"]
    labels = ["Kalman\ninnovation ADF", "Engle-Granger\n(clean)", "Static OLS ADF\n(clean)"]
    x = np.arange(len(tests)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ax.bar(x - w / 2, [pos[t] for t in tests], w, label="Cointegrated (positive control)", color=GREEN)
    ax.bar(x + w / 2, [neg[t] for t in tests], w, label="Independent RW (negative control)", color=RED)
    for xi, t in enumerate(tests):
        ax.text(xi - w / 2, pos[t] + 1.5, f"{pos[t]:.0f}", ha="center", fontsize=7.5)
        ax.text(xi + w / 2, neg[t] + 1.5, f"{neg[t]:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("pairs passing (%)"); ax.set_ylim(0, 112)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    save(fig, "fig_poscontrol_vs_negcontrol.pdf")


# ---------------------------------------------------------------- Fig 3
def fig_reversion_persistence():
    df = pd.read_csv(os.path.join(ROOT, "scratch", "persistence_pairs.csv"))
    d = df[["tr_kappa", "oos_excess"]].replace([np.inf, -np.inf], np.nan).dropna()
    xk, yo = d["tr_kappa"].to_numpy(), d["oos_excess"].to_numpy()
    rho_real = pd.Series(xk).corr(pd.Series(yo), method="spearman")
    rng = np.random.default_rng(0)
    yo_shuf = rng.permutation(yo)
    rho_shuf = pd.Series(xk).corr(pd.Series(yo_shuf), method="spearman")

    def panel(ax, yv, rho, title):
        xlo, xhi = np.percentile(xk, [1, 99])
        ylo, yhi = np.percentile(yv, [2, 98])
        ax.scatter(xk, yv, s=4, alpha=0.10, color=BLUE, edgecolors="none")
        # quantile-binned median trend
        qs = np.quantile(xk, np.linspace(0, 1, 13))
        cx, cy = [], []
        for a, b in zip(qs[:-1], qs[1:]):
            m = (xk >= a) & (xk <= b)
            if m.sum() > 5:
                cx.append(xk[m].mean()); cy.append(np.median(yv[m]))
        ax.plot(cx, cy, "-o", color=ORANGE, lw=1.6, ms=3.5, label="binned median")
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel(r"in-sample reversion speed $\hat\kappa$ (train)")
        ax.text(0.97, 0.06, rf"Spearman $\rho={rho:.2f}$", transform=ax.transAxes,
                ha="right", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
    panel(a1, yo, rho_real, "Real pairs")
    panel(a2, yo_shuf, rho_shuf, "Rank-shuffled placebo")
    a1.set_ylabel("out-of-sample excess reversion")
    a1.legend(fontsize=8, loc="upper left", frameon=True, facecolor="white",
              edgecolor="0.85", framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig_reversion_persistence_scatter.pdf")
    print(f"    [persistence] rho_real={rho_real:.3f}  rho_shuf={rho_shuf:.3f}  n={len(d)}")


# ---------------------------------------------------------------- Fig 4
def fig_circuit_breaker():
    df = pd.read_csv(os.path.join(ROOT, "scratch", "survivorship_adjusted_sharpe.csv"))
    xr = df["approx_annual"].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.plot(xr, df["monthlyS_nobreaker"], "-o", color=RED, lw=1.6, ms=4, label="no circuit breaker")
    ax.plot(xr, df["monthlyS_breaker"], "-o", color=BLUE, lw=1.6, ms=4, label="with circuit breaker")
    ax.axhline(0, color="#888888", lw=0.7)
    ax.axvline(8, color="#bbbbbb", lw=0.8, ls="--")
    ax.text(8.4, ax.get_ylim()[1] * 0.92, "8%/yr\n(liquid-major attrition)", fontsize=7.2, color="#555555")
    ax.set_xlabel("injected delisting break rate (%/year)")
    ax.set_ylabel("monthly Sharpe")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_circuit_breaker_retention.pdf")


# ---------------------------------------------------------------- Fig 5
def fig_forced_collapse():
    df = pd.read_csv(os.path.join(ROOT, "scratch", "forced_collapse.csv"))
    coins = [c.replace("USDT", "") for c in df["coin"]]
    x = np.arange(len(coins)); w = 0.36
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.7))
    a1.bar(x - w / 2, df["nostop_pnl_pct"], w, color=RED, label="no breaker")
    a1.bar(x + w / 2, df["cb_pnl_pct"], w, color=BLUE, label="with breaker")
    a1.set_title("forced-hold P&L per pair", fontsize=9, loc="left")
    a1.set_ylabel("P&L (%)"); a1.axhline(0, color="#888888", lw=0.7)
    a2.bar(x - w / 2, df["nostop_maxdd_pct"], w, color=RED)
    a2.bar(x + w / 2, df["cb_maxdd_pct"], w, color=BLUE)
    a2.set_title("worst drawdown per pair", fontsize=9, loc="left")
    a2.set_ylabel("max drawdown (%)"); a2.axhline(0, color="#888888", lw=0.7)
    for ax in (a1, a2):
        ax.set_xticks(x); ax.set_xticklabels(coins)
    # shared legend along the bottom, clear of the (downward) bars in both panels
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, "fig_forced_collapse_perpair.pdf")


# ---------------------------------------------------------------- Fig 6
def fig_ofi_decay():
    # incremental book-OFI R^2 over trade-OFI; extended horizons re-run with book_ofi_2024.py
    # (run_ofi_extended.py); horizons 1/5/10/30 reproduce book_ofi_2024.log exactly.
    df = pd.read_csv(os.path.join(ROOT, "scratch", "ofi_decay_extended.csv")).set_index("sym")
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]
    hs = [1, 2, 5, 10, 30, 60, 120, 300]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    xfull = [0] + hs
    for s in syms:
        yv = [df.loc[s, "contemp_incr_book"] * 100] + [df.loc[s, f"h{h}"] * 100 for h in hs]
        a1.plot(xfull, yv, "-o", color=SYM_C[s], lw=1.5, ms=4, label=s.replace("USDT", ""))
    a1.set_title("contemporaneous vs forecast horizon", fontsize=9, loc="left")
    a1.set_xlabel("forecast horizon (s); 0 = contemporaneous")
    a1.set_ylabel(r"incremental book-OFI $R^2$ (%)")
    a1.legend(frameon=False, fontsize=8, loc="upper right")
    for s in syms:
        yv = [df.loc[s, f"h{h}"] * 100 for h in hs]
        a2.plot(hs, yv, "-o", color=SYM_C[s], lw=1.5, ms=4, label=s.replace("USDT", ""))
    a2.set_xscale("log")
    a2.set_xticks(hs); a2.set_xticklabels([str(h) for h in hs], fontsize=7)
    a2.xaxis.set_minor_locator(plt.NullLocator())
    a2.set_title("forecast horizons only (log scale)", fontsize=9, loc="left")
    a2.set_xlabel("forecast horizon (s)")
    a2.set_ylabel(r"incremental book-OFI $R^2$ (%)")
    a2.axhline(0, color="#aaaaaa", lw=0.6)
    fig.tight_layout()
    save(fig, "fig_ofi_decay.pdf")


# ---------------------------------------------------------------- Fig 7
def fig_microcap_adv():
    df = pd.read_csv(os.path.join(ROOT, "scratch", "microcap_adv.csv"))
    majors = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    df["is_major"] = df["symbol"].isin(majors)
    df = df.sort_values("median_daily_usd_vol", ascending=False).reset_index(drop=True)
    colors = [BLUE if m else RED for m in df["is_major"]]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = np.arange(len(df))
    ax.bar(x, df["median_daily_usd_vol"], color=colors)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([s.replace("USDT", "") for s in df["symbol"]], rotation=0)
    ax.set_ylabel("median daily USD volume")
    for xi, v in zip(x, df["median_daily_usd_vol"]):
        ax.text(xi, v * 1.3, f"${v:,.0f}", ha="center", fontsize=6.8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=RED)]
    ax.legend(handles, ["majors", "held-out microcaps"], frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(10, df["median_daily_usd_vol"].max() * 4)
    fig.tight_layout()
    save(fig, "fig_microcap_adv.pdf")


# ---------------------------------------------------------------- Fig 8
def fig_hac_inflation():
    df = pd.read_csv(os.path.join(ROOT, "docs", "hac_sharpe_per_split.csv"))
    d = df[df.model == "zscore_rule"].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["inflation_factor",
                  "iid_pnl_mean_to_std", "hac_pnl_mean_to_std_lag24"])
    inf = d["inflation_factor"].to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.4))
    a1.hist(inf, bins=12, color=BLUE, edgecolor="white")
    a1.axvline(np.median(inf), color=ORANGE, lw=1.5, ls="--",
               label=f"median {np.median(inf):.2f}x")
    a1.set_xlabel("IID / HAC Sharpe inflation factor")
    a1.set_ylabel(f"walk-forward splits (n={len(d)})")
    a1.legend(frameon=False, fontsize=8, loc="upper right")
    a1.set_title("inflation across splits", fontsize=9, loc="left")
    iid, hac = d["iid_pnl_mean_to_std"].to_numpy(), d["hac_pnl_mean_to_std_lag24"].to_numpy()
    a2.scatter(iid, hac, s=16, color=BLUE, alpha=0.8, edgecolors="none")
    lim = max(iid.max(), hac.max()) * 1.08
    xs = np.linspace(0, lim, 50)
    a2.plot(xs, xs, color="#bbbbbb", lw=0.8, label="HAC = IID")
    a2.plot(xs, 0.5 * xs, color=ORANGE, lw=1.0, ls="--", label="HAC = 0.5 IID")
    a2.set_xlim(0, lim); a2.set_ylim(0, lim)
    a2.set_xlabel("IID per-bar Sharpe"); a2.set_ylabel("Newey-West HAC Sharpe")
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    a2.set_title("per-split IID vs HAC", fontsize=9, loc="left")
    fig.tight_layout()
    save(fig, "fig_hac_inflation.pdf")
    lo, hi = np.percentile(inf, [25, 75])
    print(f"    [hac] n={len(d)} splits, median inflation {np.median(inf):.2f}x, IQR [{lo:.2f}, {hi:.2f}]")


# ---------------------------------------------------------------- Fig 9
def fig_cancellation_share():
    # cancel share of best-level size reductions, ALL-2024, from book_ofi_2024.log table [3].
    share = {"BTCUSDT": 0.802, "ETHUSDT": 0.803, "SOLUSDT": 0.768, "AVAXUSDT": 0.898}
    syms = list(share)
    x = np.arange(len(syms))
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.bar(x, [share[s] * 100 for s in syms], color=[SYM_C[s] for s in syms], width=0.6)
    for xi, s in enumerate(syms):
        ax.text(xi, share[s] * 100 + 1, f"{share[s]*100:.0f}%", ha="center", fontsize=8)
    ax.axhline(50, color="#bbbbbb", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([s.replace("USDT", "") for s in syms])
    ax.set_ylabel("cancellations as % of best-level\nsize reductions")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    save(fig, "fig_cancellation_share.pdf")


# ---------------------------------------------------------------- reshaped freq
def fig_freq_invariance_reshaped():
    df = pd.read_csv(os.path.join(ROOT, "scratch", "timeframe_kalman_artifact.csv"))
    order = ["1H", "4H", "1D"]
    df = df.set_index("freq").loc[order].reset_index()
    labels = ["hourly", "4 hour", "daily"]
    kalman_gap = (df["real_k05"] - df["plac_k05"]).to_numpy()          # ~0 everywhere
    eg_gap = (df["eg05"] - 5.0).to_numpy()                            # clean test vs 5% null size
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.bar(x, eg_gap, 0.5, color=BLUE, label="clean Engle-Granger")
    ax.plot(x, kalman_gap, "o-", color="#4a5568", lw=1.6, ms=8, zorder=5,
            label=r"Kalman screen (gap $=0$)")
    for xi in range(len(labels)):
        ax.text(xi, eg_gap[xi] + 0.6, f"{eg_gap[xi]:.0f}", ha="center", fontsize=8)
        ax.text(xi, -1.4, "0", ha="center", fontsize=7.5, color="#4a5568")
    ax.axhline(0, color="#888888", lw=0.7)
    ax.set_ylim(-2.5, 22)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("real $-$ placebo pass-rate gap\n(percentage points, $p<0.05$)")
    ax.set_xlabel("bar frequency")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_freq_invariance.pdf")


if __name__ == "__main__":
    print("generating paper v2 figures ->", OUT)
    fig_kalman_innovations()
    fig_poscontrol()
    fig_reversion_persistence()
    fig_circuit_breaker()
    fig_forced_collapse()
    fig_ofi_decay()
    fig_microcap_adv()
    fig_hac_inflation()
    fig_cancellation_share()
    fig_freq_invariance_reshaped()
    print("done.")
