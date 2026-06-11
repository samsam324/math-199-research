"""Diagnostic: is the high held-out pre-reg Sharpe concentrated in a few pairs/windows?
Runs the rolling held-out no-stop config and reports per-window Sharpe and per-pair PnL."""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import nostop_breakstop as nbs

SEL_END = pd.Timestamp("2025-06-01", tz="UTC")
CC = wb.COST_LEVELS["realistic_30bps_rt"]
N_PAIRS = 40

px = wb.load_universe(); logpx = np.log(px); cols = list(px.columns)
splits = [s for s in wb.make_splits(px.index) if s[1] >= SEL_END]
pair_pnl = {}
parts = []
for (tr_s, tr_e, te_e) in splits:
    tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
    te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
    avail = [c for c in cols if tr[c].notna().mean() >= wb.MIN_OBS_FRAC]
    picks = wb.select_pairs_ou(tr[avail], avail, N_PAIRS)
    nmat = []; win = []
    for p in picks:
        net = nbs.sim_pair(te, p["a"], p["b"], p["alpha"], p["beta"], p["mu"], p["sd"],
                           CC["fee_bps"], CC["slip_bps"])
        if net is not None:
            nmat.append(pd.Series(net, index=te.index[:len(net)]))
            k = f'{p["a"]}/{p["b"]}'; v = float(np.nansum(net))
            pair_pnl[k] = pair_pnl.get(k, 0.0) + v; win.append((k, v))
    L = min(len(v) for v in nmat)
    port = pd.Series(np.mean([v.values[:L] for v in nmat], axis=0), index=te.index[:L]); parts.append(port)
    m = port.resample("ME").sum(); m = m[m != 0]
    shm = float(m.mean() / m.std(ddof=1) * np.sqrt(12)) if len(m) > 1 and m.std() > 0 else float("nan")
    top = sorted(win, key=lambda x: -x[1])[:3]
    print(f"window {tr_e.date()}..{te_e.date()}: monthlyS~{shm:.2f}  PnL={float(np.nansum(port.values))*100:+.0f}%  "
          f"top pairs {[(k, round(v*100)) for k, v in top]}")

print("\ntop-10 pairs by total held-out PnL:")
for k, v in sorted(pair_pnl.items(), key=lambda x: -x[1])[:10]:
    print(f"  {k:<26} {v*100:+.1f}%")
tot = sum(pair_pnl.values())
vals = sorted(pair_pnl.values())
top1 = vals[-1]; top3 = sum(vals[-3:])
print(f"\ntotal pair-PnL={tot*100:.0f}%  top-1={top1*100:.0f}% ({top1/tot*100:.0f}% of total)  "
      f"top-3={top3*100:.0f}% ({top3/tot*100:.0f}% of total)  n_pairs={len(pair_pnl)}")

# Sharpe after dropping the top-3 contributing pairs
drop = set(k for k, _ in sorted(pair_pnl.items(), key=lambda x: -x[1])[:3])
parts2 = []
for (tr_s, tr_e, te_e) in splits:
    tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
    te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
    avail = [c for c in cols if tr[c].notna().mean() >= wb.MIN_OBS_FRAC]
    picks = wb.select_pairs_ou(tr[avail], avail, N_PAIRS)
    nmat = []
    for p in picks:
        if f'{p["a"]}/{p["b"]}' in drop:
            continue
        net = nbs.sim_pair(te, p["a"], p["b"], p["alpha"], p["beta"], p["mu"], p["sd"],
                           CC["fee_bps"], CC["slip_bps"])
        if net is not None:
            nmat.append(pd.Series(net, index=te.index[:len(net)]))
    L = min(len(v) for v in nmat)
    parts2.append(pd.Series(np.mean([v.values[:L] for v in nmat], axis=0), index=te.index[:L]))
port2 = pd.concat(parts2).sort_index(); port2 = port2[~port2.index.duplicated(keep="first")]
m2 = port2.resample("ME").sum(); m2 = m2[m2 != 0]
shm2 = float(m2.mean() / m2.std(ddof=1) * np.sqrt(12)) if len(m2) > 1 and m2.std() > 0 else float("nan")
print(f"\nmonthly Sharpe after dropping top-3 pairs (hindsight): {shm2:.2f}")
