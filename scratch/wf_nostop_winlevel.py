"""
Iteration-8 self-check: is the no-stop +2.51 Sharpe backed by broad cross-window
consistency, or is the hourly Sharpe just flattered by multi-week-hold
autocorrelation (so the 168h-block bootstrap CI is too tight)?

The honest unit of observation for a strategy whose median hold is ~35 days and
where 78% of positions never converge within a 3-month window is the WINDOW, not
the hour. Here we aggregate net P&L to each of the 19 disjoint walk-forward TEST
windows and test the mean window return across windows (t-stat with only 19 d.f.,
fraction-positive sign test), for both the |z|=4-stop and no-stop rules. This is
the conservative significance that the hourly Sharpe overstates.

Run synchronously:  python scratch/wf_nostop_winlevel.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import wf_robustness as wr

np.seterr(all="ignore")


def window_returns(px, cost_cfg, cfg):
    """Return a list of per-window net total returns (sum of hourly net pnl over
    the equal-weight pair portfolio in each TEST window)."""
    logpx = np.log(px)
    cols = list(px.columns)
    splits = wb.make_splits(px.index)
    fee = cost_cfg["fee_bps"]; slip = cost_cfg["slip_bps"]
    wins = []
    for (tr_s, tr_e, te_e) in splits:
        tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(tr) < 1000 or len(te) < 200:
            continue
        picks = wb.select_pairs_ou(tr, cols, wb.N_PAIRS)
        if not picks:
            continue
        nets = []
        for p in picks:
            r = wr.simulate_pair_v(te, p["a"], p["b"], p["alpha"], p["beta"],
                                   p["mu"], p["sd"], fee, slip, cfg)
            if r is not None:
                nets.append(r["net"])
        if not nets:
            continue
        L = min(len(v) for v in nets)
        port = np.mean([v[:L] for v in nets], axis=0)
        wins.append(float(np.nansum(port)))   # net total return for this window
    return np.array(wins)


def summarize(name, w):
    n = len(w)
    mu = w.mean(); sd = w.std(ddof=1)
    t = mu / (sd / np.sqrt(n)) if sd > 0 else np.nan
    fpos = np.mean(w > 0)
    # sign test p (two-sided) vs 0.5
    from math import comb
    k = int(np.sum(w > 0))
    p_sign = 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2**n if k >= n/2 else \
             2 * sum(comb(n, i) for i in range(0, k + 1)) / 2**n
    p_sign = min(1.0, p_sign)
    print(f"  {name:<16} n={n}  mean={mu*100:+.1f}%  median={np.median(w)*100:+.1f}%  "
          f"sd={sd*100:.1f}%  t={t:+.2f}  %pos={fpos*100:.0f}%  sign-p={p_sign:.3f}  "
          f"min={w.min()*100:+.1f}%  max={w.max()*100:+.1f}%")
    return dict(mean=mu, t=t, fpos=fpos, p_sign=p_sign)


def main():
    px = wb.load_universe()
    cc = wb.COST_LEVELS["realistic_30bps_rt"]
    cfg_stop   = dict(stop=4.0,    exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")
    cfg_nostop = dict(stop=np.inf, exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")
    print("=" * 92)
    print("ITER-8 SELF-CHECK: window-level significance of the no-stop result (realistic 30bps)")
    print("  unit of observation = the 19 disjoint 3-mo TEST windows (conservative; hourly Sharpe overstates)")
    print("=" * 92)
    w_stop = window_returns(px, cc, cfg_stop)
    w_nostop = window_returns(px, cc, cfg_nostop)
    print("\nPer-window NET total return:")
    summarize("|z|=4 stop", w_stop)
    summarize("no-stop", w_nostop)
    # paired difference (same windows): does no-stop beat stop window-by-window?
    m = min(len(w_stop), len(w_nostop))
    d = w_nostop[:m] - w_stop[:m]
    print("\nPaired (no-stop minus stop), same windows:")
    summarize("diff", d)
    print("\nno-stop per-window returns (%):",
          np.round(w_nostop * 100, 1).tolist())
    print("=" * 92)


if __name__ == "__main__":
    main()
