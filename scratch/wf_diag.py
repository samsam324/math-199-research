"""Diagnostic: inspect OU-selected pairs per split, hedge stability TRAIN->TEST,
and gross spread-reversion realism. Helps decide if negative gross is real."""
import numpy as np, pandas as pd, itertools, importlib.util, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import wf_backtest as W

px = W.load_universe()
logpx = np.log(px); cols = list(px.columns)
splits = W.make_splits(px.index)
print(f"{len(splits)} splits, {len(cols)} syms")

for i,(tr_s,tr_e,te_e) in enumerate(splits):
    tr = logpx[(logpx.index>=tr_s)&(logpx.index<tr_e)]
    te = logpx[(logpx.index>=tr_e)&(logpx.index<te_e)]
    if len(tr)<1000 or len(te)<200: continue
    picks = W.select_pairs_ou(tr, cols, W.N_PAIRS)
    if not picks:
        print(f"split {i}: no picks"); continue
    # check hedge stability + test reversion of z (does spread revert in test?)
    diag=[]
    for p in picks:
        la_tr=tr[p["a"]].values; lb_tr=tr[p["b"]].values
        # refit hedge in test to compare
        m=np.isfinite(te[p["a"]].values)&np.isfinite(te[p["b"]].values)
        a2,b2=W.ols_hedge(te[p["a"]].values[m], te[p["b"]].values[m])
        # test spread under TRAIN hedge
        sp_te = te[p["a"]].values - p["alpha"] - p["beta"]*te[p["b"]].values
        z_te=(sp_te-p["mu"])/p["sd"]
        # half-life in test under train hedge
        hl_te,_=W.ou_half_life(sp_te)
        frac_extreme=np.mean(np.abs(z_te[np.isfinite(z_te)])>2)
        diag.append((p["a"],p["b"],p["hl"],hl_te,p["beta"],b2,np.nanstd(z_te),frac_extreme))
    print(f"\n--- split {i} TRAIN {tr_s.date()}..{tr_e.date()} | TEST ..{te_e.date()} ---")
    print(f"{'pair':<20}{'hlTR':>7}{'hlTE':>7}{'bTR':>7}{'bTE':>7}{'zSD_te':>8}{'frcEx':>7}")
    for d in diag:
        print(f"{d[0]+'/'+d[1]:<20}{d[2]:>7.0f}{(d[3] if np.isfinite(d[3]) else -1):>7.0f}"
              f"{d[4]:>7.2f}{d[5]:>7.2f}{d[6]:>8.2f}{d[7]:>7.2f}")
    if i>=3: break
