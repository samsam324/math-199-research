"""
Iteration-10 characterization: WHAT is the no-stop +2.51 Sharpe?

The classic hidden confound in any "market-neutral" stat-arb backtest: the PnL is
pos*(r_A - r_B) -- dollar-neutral (long $1 A, short $1 B) but NOT necessarily
BETA-neutral. If A and B have different market betas, or the book drifts net-long
the underperformers in a bull market, the apparent alpha can be disguised crypto
beta. Over 2021-2026 (a big net-up crypto regime) this matters a lot.

Test: regress the no-stop (and stop) portfolio hourly returns on (i) BTC return
and (ii) an equal-weight crypto-market return. Report beta (block-bootstrap CI),
raw annualized Sharpe, and the MARKET-NEUTRALIZED residual Sharpe (port - beta*mkt).
Also PCA: how much of the portfolio return variance is the market factor (PC1)?

If beta ~ 0 and residual Sharpe ~ raw Sharpe, the no-stop edge is genuinely
market-neutral reversion. If beta is large and residual Sharpe << raw, it was beta.

Run synchronously:  python scratch/wf_nostop_factor.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import wf_robustness as wr

np.seterr(all="ignore")
HPY = wb.HOURS_PER_YEAR


def portfolio_series(px, cfg, cost_cfg):
    """Concatenated timestamped hourly GROSS and NET portfolio return series
    (equal-weight across the OU-selected pairs in each disjoint test window)."""
    logpx = np.log(px); cols = list(px.columns); splits = wb.make_splits(px.index)
    fee = cost_cfg["fee_bps"]; slip = cost_cfg["slip_bps"]
    gross_parts = []; net_parts = []
    for (tr_s, tr_e, te_e) in splits:
        tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(tr) < 1000 or len(te) < 200:
            continue
        picks = wb.select_pairs_ou(tr, cols, wb.N_PAIRS)
        if not picks:
            continue
        gmat = []; nmat = []
        for p in picks:
            r = wr.simulate_pair_v(te, p["a"], p["b"], p["alpha"], p["beta"],
                                   p["mu"], p["sd"], fee, slip, cfg)
            if r is not None:
                gmat.append(r["gross"]); nmat.append(r["net"])
        if not gmat:
            continue
        L = min(len(v) for v in gmat)
        idx = te.index[:L]
        gross_parts.append(pd.Series(np.mean([v[:L] for v in gmat], axis=0), index=idx))
        net_parts.append(pd.Series(np.mean([v[:L] for v in nmat], axis=0), index=idx))
    g = pd.concat(gross_parts).sort_index()
    n = pd.concat(net_parts).sort_index()
    g = g[~g.index.duplicated(keep="first")]
    n = n[~n.index.duplicated(keep="first")]
    return g, n


def ann_sharpe(x):
    s, _, _ = wb.sharpe_hac(np.asarray(x, float))
    return s


def beta_ci(port, mkt, block=168, nboot=1000, seed=0):
    """OLS beta of port on mkt + block-bootstrap 95% CI."""
    x = mkt.values; y = port.values
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    b = np.cov(x, y)[0, 1] / np.var(x)
    a = y.mean() - b * x.mean()
    n = len(x); rng = np.random.default_rng(seed)
    nbl = int(np.ceil(n / block)); starts = np.arange(0, n - block + 1)
    bs = np.empty(nboot)
    for i in range(nboot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, nbl)])[:n]
        xb = x[idx]; yb = y[idx]
        bs[i] = np.cov(xb, yb)[0, 1] / np.var(xb)
    return a, b, (np.percentile(bs, 2.5), np.percentile(bs, 97.5))


def analyze(label, port, mkt_eq, mkt_btc):
    common = port.index.intersection(mkt_eq.index)
    p = port.reindex(common); me = mkt_eq.reindex(common); mb = mkt_btc.reindex(common)
    raw = ann_sharpe(p)
    print(f"\n##### {label}  (n={len(p)} hrs) #####")
    print(f"  raw annualized Sharpe (gross) = {raw:+.2f}")
    for mk_name, mk in [("equal-wt market", me), ("BTC", mb)]:
        a, b, ci = beta_ci(p, mk)
        resid = p - b * mk
        resid_sh = ann_sharpe(resid)
        corr = np.corrcoef(p.values, mk.values)[0, 1]
        ann_alpha = a * HPY
        print(f"  vs {mk_name:<16}: beta={b:+.4f} (95% CI [{ci[0]:+.4f},{ci[1]:+.4f}]) "
              f"corr={corr:+.3f}  ann_alpha={ann_alpha*100:+.1f}%  "
              f"market-neutralized Sharpe={resid_sh:+.2f}")


def main():
    px = wb.load_universe()
    cc = wb.COST_LEVELS["realistic_30bps_rt"]
    cfg_stop   = dict(stop=4.0,    exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")
    cfg_nostop = dict(stop=np.inf, exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")
    print("=" * 96)
    print("ITER-10: is the no-stop edge market-neutral reversion, or disguised crypto beta?")
    print("=" * 96)
    logret = np.log(px).diff()
    mkt_eq = logret.mean(axis=1).dropna()           # equal-weight crypto market
    mkt_btc = logret["BTCUSDT"].dropna() if "BTCUSDT" in px.columns else mkt_eq

    g_ns, n_ns = portfolio_series(px, cfg_nostop, cc)
    g_st, n_st = portfolio_series(px, cfg_stop, cc)

    analyze("NO-STOP (gross)", g_ns, mkt_eq, mkt_btc)
    analyze("|z|=4 STOP (gross)", g_st, mkt_eq, mkt_btc)

    # average net market exposure proxy: mean of the portfolio's contemporaneous
    # correlation sign already captured by beta; also report the fraction of bars
    # the no-stop portfolio return co-moves with the market.
    common = g_ns.index.intersection(mkt_eq.index)
    p = g_ns.reindex(common); m = mkt_eq.reindex(common)
    print(f"\n  no-stop: % bars same-sign as market = {np.mean(np.sign(p)==np.sign(m))*100:.1f}% "
          f"(50% = independent)")

    # DAILY-frequency beta: a multi-week-hold strategy could accumulate slow market
    # exposure invisible at hourly scale. Resample PnL (sum) & market (sum of logret).
    print("\n  --- DAILY-frequency check (slow-beta accumulation) ---")
    pj = g_ns.copy(); pj.index = pd.to_datetime(pj.index)
    pd_day = pj.resample("1D").sum()
    md_day = m.copy(); md_day.index = pd.to_datetime(md_day.index)
    md_day = md_day.resample("1D").sum()
    cmn = pd_day.index.intersection(md_day.index)
    pdd = pd_day.reindex(cmn); mdd = md_day.reindex(cmn)
    msk = (pdd != 0)  # drop empty days (between windows)
    pdd = pdd[msk]; mdd = mdd[msk]
    bb = np.cov(mdd.values, pdd.values)[0, 1] / np.var(mdd.values)
    corr_d = np.corrcoef(pdd.values, mdd.values)[0, 1]
    print(f"  no-stop DAILY: beta_to_eqmkt={bb:+.4f}  corr={corr_d:+.3f}  (n_days={len(pdd)})")
    print("=" * 96)


if __name__ == "__main__":
    main()
