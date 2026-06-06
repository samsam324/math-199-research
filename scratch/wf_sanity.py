"""Sanity: under the static-TRAIN hedge+mean, does the OU-selected TEST spread
actually revert after |z|>=2 (gross, no costs, no stop)? Measures mean forward
spread-return conditional on entry, sign-corrected, pooled over all splits.
If this is <=0 the negative backtest is REAL (no mean reversion OOS), not a bug."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import wf_backtest as W

px = W.load_universe(); logpx = np.log(px); cols=list(px.columns)
splits = W.make_splits(px.index)
H = 48  # forward horizon hours
fwd_ret_entry=[]; n_entries=0
for (tr_s,tr_e,te_e) in splits:
    tr=logpx[(logpx.index>=tr_s)&(logpx.index<tr_e)]
    te=logpx[(logpx.index>=tr_e)&(logpx.index<te_e)]
    if len(tr)<1000 or len(te)<200: continue
    for p in W.select_pairs_ou(tr, cols, W.N_PAIRS):
        la=te[p["a"]].values; lb=te[p["b"]].values
        sp = la - p["alpha"] - p["beta"]*lb
        z=(sp-p["mu"])/p["sd"]
        spread_ret = np.diff(sp, prepend=sp[0])
        spread_ret=np.where(np.isfinite(spread_ret),spread_ret,0.0)
        n=len(z)
        for t in range(n-H):
            if not (np.isfinite(z[t]) and np.isfinite(sp[t]) and np.isfinite(sp[t+H])): continue
            if z[t]>=2:   # short spread: profit if spread FALLS -> -fwd
                fwd = -(sp[t+H]-sp[t]); fwd_ret_entry.append(fwd/p["sd"]); n_entries+=1
            elif z[t]<=-2: # long spread: profit if spread RISES
                fwd = (sp[t+H]-sp[t]); fwd_ret_entry.append(fwd/p["sd"]); n_entries+=1
arr=np.array(fwd_ret_entry)
print(f"OU-selected, static TRAIN hedge/mean, |z|>=2 entries: N={n_entries}")
print(f"mean reversion-direction forward {H}h move (in z units): {arr.mean():+.4f}")
print(f"  (>0 = spread reverts toward mean = profitable; <0 = diverges)")
print(f"  median={np.median(arr):+.4f}  share profitable={np.mean(arr>0)*100:.1f}%")
# block-ish SE
se=arr.std()/np.sqrt(len(arr)/H)  # deflate for overlap
print(f"  approx t (overlap-deflated) = {arr.mean()/se:+.2f}")
