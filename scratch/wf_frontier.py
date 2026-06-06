"""
Iteration-12: the risk-rule efficient frontier for capturing the reversion alpha.

Iters 8-11 established a genuine market-neutral diversified mean-reversion alpha that
is (a) destroyed by a tight |z|=4 stop (net Sharpe -2.25) and (b) capturable no-stop
at +2.51 BUT with a brutal -41% portfolio drawdown on 10 pairs. The decision-relevant
question an advisor would ask: is there a PRACTICAL middle — diversify (more pairs),
vol-target, or use a loose stop — that keeps most of the Sharpe at a BEARABLE drawdown?
The -41% DD is partly idiosyncratic across pairs, so more pairs should cut it.

Maps (annualized Sharpe, max drawdown, Calmar = annual-return/|maxDD|) across:
  n_pairs in {10, 20, 40}, stop in {none, |z|=6, |z|=4}, exit in {z, conv},
  sizing in {unit, vol-target}. Realistic 30bps. Reports the frontier.

Run synchronously:  python scratch/wf_frontier.py
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


def run_cfg(px, cost_cfg, cfg, n_pairs):
    """Walk-forward; return concatenated net hourly series + trade/turnover totals."""
    logpx = np.log(px); cols = list(px.columns); splits = wb.make_splits(px.index)
    fee = cost_cfg["fee_bps"]; slip = cost_cfg["slip_bps"]
    parts = []; tot_tr = 0; tot_turn = 0.0
    for (tr_s, tr_e, te_e) in splits:
        tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(tr) < 1000 or len(te) < 200:
            continue
        picks = wb.select_pairs_ou(tr, cols, n_pairs)
        if not picks:
            continue
        nmat = []
        for p in picks:
            hl = p.get("hl", np.nan)
            pcfg = dict(cfg)
            if cfg.get("exit_mode") == "time":
                pcfg["max_hold"] = max(2, int(round(2.0 * hl)) if np.isfinite(hl) and hl > 0 else 48)
            r = wr.simulate_pair_v(te, p["a"], p["b"], p["alpha"], p["beta"],
                                   p["mu"], p["sd"], fee, slip, pcfg)
            if r is not None:
                nmat.append(r["net"]); tot_tr += r["n_trades"]; tot_turn += r["turnover"]
        if not nmat:
            continue
        L = min(len(v) for v in nmat)
        parts.append(np.mean([v[:L] for v in nmat], axis=0))
    if not parts:
        return None
    return np.concatenate(parts), tot_tr, tot_turn


def maxdd(net):
    eq = np.cumsum(np.nan_to_num(net))
    return float((eq - np.maximum.accumulate(eq)).min())


def main():
    px = wb.load_universe()
    cc = wb.COST_LEVELS["realistic_30bps_rt"]
    base = dict(stop=np.inf, exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")

    configs = [
        # (label, n_pairs, cfg-overrides)
        ("10p nostop z (iter8 baseline)", 10, {}),
        ("20p nostop z",                  20, {}),
        ("40p nostop z",                  40, {}),
        ("40p nostop z vol-target",       40, {"sizing": "vol", "vol_win": 720}),
        ("40p nostop conv",               40, {"exit_mode": "conv", "max_hold": 1000}),
        ("40p wide-stop|z|=6 z",          40, {"stop": 6.0}),
        ("40p wide-stop|z|=6 vol",        40, {"stop": 6.0, "sizing": "vol", "vol_win": 720}),
        ("20p nostop z vol-target",       20, {"sizing": "vol", "vol_win": 720}),
        ("40p tight-stop|z|=4 z (ref)",   40, {"stop": 4.0}),
        ("10p tight-stop|z|=4 (loser)",   10, {"stop": 4.0}),
    ]

    print("=" * 104)
    print("ITER-12: risk-rule efficient frontier for the reversion alpha (realistic 30bps, OU-selected)")
    print("  Calmar = annualized return / |max drawdown|.  Higher Sharpe AND smaller |DD| is better.")
    print("=" * 104)
    print(f"{'config':<34}{'netSharpe':>10}{'maxDD%':>9}{'annRet%':>9}{'Calmar':>8}{'netPnL%':>9}{'trades':>8}")
    print("-" * 104)
    rows = []
    for label, npairs, ov in configs:
        cfg = dict(base); cfg.update(ov)
        out = run_cfg(px, cc, cfg, npairs)
        if out is None:
            print(f"{label:<34}  NO RESULT"); continue
        net, ntr, turn = out
        sh, _, _ = wb.sharpe_hac(net)
        dd = maxdd(net)
        annret = float(np.nanmean(net)) * HPY
        calmar = annret / abs(dd) if dd < 0 else np.nan
        netpnl = float(np.nansum(net))
        rows.append((label, sh, dd, annret, calmar, netpnl, ntr))
        print(f"{label:<34}{sh:>10.2f}{dd*100:>9.1f}{annret*100:>9.1f}{calmar:>8.2f}{netpnl*100:>9.0f}{ntr:>8}")
    print("-" * 104)
    # best by Calmar among positive-Sharpe configs
    pos = [r for r in rows if r[1] > 0 and np.isfinite(r[4])]
    if pos:
        best = max(pos, key=lambda r: r[4])
        print(f"  best risk-adjusted (Calmar): {best[0]}  -> Sharpe {best[1]:+.2f}, maxDD {best[2]*100:.1f}%, Calmar {best[4]:.2f}")
        # diversification effect on DD
        d10 = next((r for r in rows if r[0].startswith("10p nostop z")), None)
        d40 = next((r for r in rows if r[0] == "40p nostop z"), None)
        if d10 and d40:
            print(f"  diversification 10p->40p (nostop z): maxDD {d10[2]*100:.1f}% -> {d40[2]*100:.1f}%, "
                  f"Sharpe {d10[1]:+.2f} -> {d40[1]:+.2f}")
    print("=" * 104)


if __name__ == "__main__":
    main()
