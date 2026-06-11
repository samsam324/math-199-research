"""
Pre-registered single-config evaluation of the never-stop reversion book.

Config locked in docs/prereg_reversion.md (committed before this run). One held-out window,
evaluated once, reported whatever it is. No parameter here may differ from the locked values.

Locked config: 40 OU-half-life-selected pairs, static OLS hedge frozen on the selection
window, static train-window z, |z|>=2 entry, |z|<=0.5 convergence exit, NO stop. Also run with
the structural-break circuit breaker (either leg 50% adverse + halt). Realistic costs (15bps/leg).

Selection window: data start through 2025-05-31. Evaluation window: 2025-06-01 .. 2026-05-18.

Run: python scratch/prereg_run.py
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import nostop_breakstop as nbs

SEL_END = pd.Timestamp("2025-06-01", tz="UTC")    # selection uses data strictly before this
EVAL_END = pd.Timestamp("2026-05-19", tz="UTC")   # evaluate [SEL_END, EVAL_END)
N_PAIRS = 40
CC = wb.COST_LEVELS["realistic_30bps_rt"]


def evaluate(px, circuit_breaker):
    logpx = np.log(px)
    cols = list(px.columns)
    tr = logpx[logpx.index < SEL_END]
    te = logpx[(logpx.index >= SEL_END) & (logpx.index < EVAL_END)]
    avail = [c for c in cols if tr[c].notna().mean() >= wb.MIN_OBS_FRAC]
    picks = wb.select_pairs_ou(tr[avail], avail, N_PAIRS)
    kw = dict(adverse_K=0.50, halt=True) if circuit_breaker else {}
    nmat = []
    for p in picks:
        net = nbs.sim_pair(te, p["a"], p["b"], p["alpha"], p["beta"], p["mu"], p["sd"],
                           CC["fee_bps"], CC["slip_bps"], **kw)
        if net is not None:
            nmat.append(pd.Series(net, index=te.index[:len(net)]))
    if not nmat:
        return None
    L = min(len(v) for v in nmat)
    port = pd.Series(np.mean([v.values[:L] for v in nmat], axis=0), index=te.index[:L])
    sh_h, se_h, infl = wb.sharpe_hac(port.values)
    ci = wb.block_bootstrap_sharpe_ci(port.values)
    m = port.resample("ME").sum(); m = m[m != 0]
    sh_m = float(m.mean() / m.std(ddof=1) * np.sqrt(12)) if len(m) > 2 and m.std() > 0 else float("nan")
    eq = np.cumsum(np.nan_to_num(port.values)); dd = float((eq - np.maximum.accumulate(eq)).min())
    idx = np.log(px).diff().mean(axis=1).reindex(port.index).values
    pr = port.values
    mask = np.isfinite(idx) & np.isfinite(pr)
    beta = float(np.polyfit(idx[mask], pr[mask], 1)[0]) if mask.sum() > 30 else float("nan")
    return dict(n_pairs=len(nmat), n_hours=int(L), n_months=int(len(m)),
                sharpe_monthly=sh_m,
                sharpe_hourly_ann_hac=float(sh_h), sharpe_hourly_se=float(se_h),
                hac_inflation=float(infl),
                sharpe_hourly_ci=[float(ci[0]), float(ci[1])],
                total_pnl_pct=float(np.nansum(port.values) * 100),
                maxdd_pct=float(dd * 100), beta_to_index=beta)


def main():
    px = wb.load_universe()
    print(f"universe {px.shape[1]} syms | span {px.index.min().date()}..{px.index.max().date()}")
    print(f"selection window: < {SEL_END.date()}   eval window: [{SEL_END.date()}, {EVAL_END.date()})")
    out = {}
    for cb, name in [(False, "no_stop"), (True, "no_stop_plus_circuit_breaker")]:
        res = evaluate(px, cb)
        out[name] = res
        if res is None:
            print(f"\n[{name}] no pairs / no result"); continue
        print(f"\n[{name}]  pairs={res['n_pairs']}  months={res['n_months']}  hours={res['n_hours']}")
        print(f"  monthly Sharpe              = {res['sharpe_monthly']:.2f}")
        print(f"  hourly-ann Sharpe (HAC)     = {res['sharpe_hourly_ann_hac']:.2f} "
              f"(SE {res['sharpe_hourly_se']:.2f}, infl {res['hac_inflation']:.2f}, "
              f"95% CI [{res['sharpe_hourly_ci'][0]:.2f}, {res['sharpe_hourly_ci'][1]:.2f}])")
        print(f"  total PnL = {res['total_pnl_pct']:.1f}%   maxDD = {res['maxdd_pct']:.1f}%   "
              f"beta to index = {res['beta_to_index']:.3f}")
    with open(os.path.join(wb.ROOT, "scratch", "prereg_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> scratch/prereg_result.json")


if __name__ == "__main__":
    main()
