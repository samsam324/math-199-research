"""
Core test: conditional on |z|>2 extreme entry, does two-leg signed OFI predict
revert-vs-diverge of the pair spread better than the unconditional base rate?

Design / pitfall controls:
- z-score window = trailing 168h, computed with mean/std SHIFTED by 1 bar (no look-ahead;
  the bar at t is NOT in its own stats). Entry = |z| crosses above 2.0 from below.
- OFI signal measured over the *last* L hours up to & including t (past info only).
- Cooldown = max forward horizon -> events within a pair never overlap.
- Forward outcome = signed toward reversion: rev_ret = -sign(z_t) * (spread_{t+N} - spread_t)
  normalized by spread std (so units = z-units of reversion). Positive => reverted.
- Reversion(binary) = |z_{t+N}| < |z_t|.
- SEs: block bootstrap resampling whole pairs (clusters), 2000 reps. Per-pair also shown.
"""
import pandas as pd, numpy as np, os

CACHE = 'scratch/cache_hourly'
PAIRS = [  # 8 liquid, distinct-ish symbol sets
    ('BTCUSDT','ETHUSDT',0.976922),
    ('SHIBUSDT','DOGEUSDT',1.631940),
    ('ADAUSDT','DOTUSDT',0.852796),
    ('SOLUSDT','AVAXUSDT',1.304799),
    ('XRPUSDT','XLMUSDT',0.874818),
    ('ETHUSDT','LINKUSDT',1.041377),
    ('LINKUSDT','DOTUSDT',0.530780),
    ('MANAUSDT','SANDUSDT',1.023288),
]
ZWIN = 168          # hours, ~1 week
ZTHR = 2.0
FLOW_L = 1          # OFI measured over last 1h (the entry bar's flow)
HORIZONS = [4, 12, 24]
COOLDOWN = 24       # >= max horizon, prevents overlap within a pair

def load(sym):
    return pd.read_parquet(f'{CACHE}/{sym}.parquet')

def build_pair(a, b, beta):
    A, B = load(a), load(b)
    idx = A.index.intersection(B.index)
    A, B = A.loc[idx], B.loc[idx]
    la, lb = np.log(A['mid']), np.log(B['mid'])
    spread = la - beta*lb
    # rolling z, SHIFTED: stats use data strictly before t
    mu = spread.shift(1).rolling(ZWIN).mean()
    sd = spread.shift(1).rolling(ZWIN).std()
    z = (spread - mu) / sd
    # normalized signed flow per leg: signed_vol / trailing rolling abs flow scale (past only)
    def nflow(df):
        sv = df['signed_vol']
        scale = df['vol'].shift(1).rolling(ZWIN).mean()  # avg total vol, past
        return sv / scale.replace(0, np.nan)
    nfa, nfb = nflow(A), nflow(B)
    ofi = nfa - beta*nfb                       # two-leg signed OFI (toward leg_a up)
    # spread_t moves +1 unit when la up / lb down -> ofi positive pushes spread up.
    # intensity & magnitude
    intensity = (A['tcount'] + B['tcount'])
    intensity_z = (intensity - intensity.shift(1).rolling(ZWIN).mean()) / intensity.shift(1).rolling(ZWIN).std()
    flow_mag = nfa.abs() + (beta*nfb).abs()
    out = pd.DataFrame({'spread':spread,'z':z,'sd':sd,'ofi':ofi,
                        'intensity_z':intensity_z,'flow_mag':flow_mag})
    return out

def events_for_pair(a,b,beta):
    df = build_pair(a,b,beta).dropna(subset=['z','sd','ofi'])
    z = df['z'].values
    cross = (np.abs(z) > ZTHR) & (np.abs(np.roll(z,1)) <= ZTHR)
    cross[0] = False
    idxpos = np.where(cross)[0]
    spread = df['spread'].values
    sd = df['sd'].values
    ofi = df['ofi'].values
    n = len(df)
    rows = []
    last_used = -10**9
    for i in idxpos:
        if i - last_used < COOLDOWN:
            continue
        if i + max(HORIZONS) >= n:
            continue
        last_used = i
        zt = z[i]; sgn = np.sign(zt)
        # confirm: ofi pushes spread further from 0 in direction of zt.
        # if z>0 (spread above mean) and ofi>0 (flow pushing spread up) -> still diverging -> CONFIRM
        confirm = np.sign(ofi[i]) == sgn  # flow same direction as deviation
        rec = {'pair':f'{a}_{b}','i':i,'z':zt,'absz':abs(zt),
               'ofi':ofi[i],'confirm':int(confirm),
               'ofi_align': ofi[i]*sgn,  # >0 confirm divergence, <0 reverting flow
               'intensity_z':df['intensity_z'].values[i],
               'flow_mag':df['flow_mag'].values[i]}
        for N in HORIZONS:
            ds = spread[i+N]-spread[i]
            rec[f'rev_ret_{N}'] = (-sgn*ds)/sd[i]      # z-units reverted (+ good)
            rec[f'reverted_{N}'] = int(abs(z[i+N]) < abs(zt))
        rows.append(rec)
    return pd.DataFrame(rows)

def main():
    ev = pd.concat([events_for_pair(*p) for p in PAIRS], ignore_index=True)
    print(f'TOTAL extreme entry events (post-cooldown): {len(ev)}')
    print('events per pair:'); print(ev.groupby('pair').size().to_string())
    print()

    def boot_diff(mask_a, mask_b, col, reps=2000):
        # cluster bootstrap by pair: resample pairs with replacement
        pairs = ev['pair'].unique()
        diffs=[]
        for _ in range(reps):
            samp = np.random.choice(pairs, len(pairs), replace=True)
            parts=[ev[ev.pair==pp] for pp in samp]
            d=pd.concat(parts)
            va=d.loc[mask_a(d),col]; vb=d.loc[mask_b(d),col]
            if len(va)<3 or len(vb)<3: continue
            diffs.append(va.mean()-vb.mean())
        diffs=np.array(diffs)
        return diffs.mean(), diffs.std(), np.percentile(diffs,2.5), np.percentile(diffs,97.5)

    conf = lambda d: d.confirm==1
    opp  = lambda d: d.confirm==0

    print('='*70)
    print('CORE TEST: forward reversion conditional on flow confirming divergence')
    print('confirm=1: flow pushing spread FURTHER from 0 (informed divergence hypothesis)')
    print('confirm=0: flow OPPOSING divergence (noise/reverting)')
    print('rev_ret in z-units (+ = reverted toward 0); reverted = binary |z| shrank')
    print('='*70)
    base = {}
    for N in HORIZONS:
        rr=f'rev_ret_{N}'; rb=f'reverted_{N}'
        uncond_rr = ev[rr].mean(); uncond_rb = ev[rb].mean()
        rr_c = ev.loc[conf(ev),rr].mean(); rr_o = ev.loc[opp(ev),rr].mean()
        rb_c = ev.loc[conf(ev),rb].mean(); rb_o = ev.loc[opp(ev),rb].mean()
        md,sd_,lo,hi = boot_diff(opp,conf,rr)      # opposing - confirming
        mdb,sdb,lob,hib = boot_diff(opp,conf,rb)
        nconf=conf(ev).sum(); nopp=opp(ev).sum()
        print(f'\n--- N={N}h ---  (n_confirm={nconf}, n_oppose={nopp})')
        print(f'  uncond mean rev_ret = {uncond_rr:+.4f} z   uncond reversion rate = {uncond_rb:.3f}')
        print(f'  rev_ret  confirm={rr_c:+.4f}  oppose={rr_o:+.4f}  diff(opp-conf)={rr_o-rr_c:+.4f}'
              f'  [boot SE {sd_:.4f}, 95%CI {lo:+.4f},{hi:+.4f}]')
        print(f'  revrate  confirm={rb_c:.3f}  oppose={rb_o:.3f}  diff(opp-conf)={rb_o-rb_c:+.4f}'
              f'  [boot SE {sdb:.4f}, 95%CI {lob:+.4f},{hib:+.4f}]')
        base[N]=(uncond_rr,uncond_rb)

    # continuous: correlation of ofi_align (signed flow toward divergence) with forward reversion
    print('\n'+'='*70)
    print('Continuous: corr(ofi_align, rev_ret) -- ofi_align>0 means flow confirms divergence.')
    print('Negative corr => more confirming flow -> LESS reversion (supports hypothesis).')
    print('='*70)
    for N in HORIZONS:
        rr=f'rev_ret_{N}'
        c=np.corrcoef(ev['ofi_align'],ev[rr])[0,1]
        # bootstrap CI on corr (cluster by pair)
        pairs=ev.pair.unique(); cs=[]
        for _ in range(2000):
            samp=np.random.choice(pairs,len(pairs),replace=True)
            d=pd.concat([ev[ev.pair==pp] for pp in samp])
            cs.append(np.corrcoef(d['ofi_align'],d[rr])[0,1])
        cs=np.array(cs)
        print(f'  N={N}h: corr={c:+.4f}  boot95%CI [{np.percentile(cs,2.5):+.4f},{np.percentile(cs,97.5):+.4f}]')

    # per-pair breakdown for N=12
    print('\n'+'='*70)
    print('PER-PAIR (N=12h): mean rev_ret confirm vs oppose')
    print('='*70)
    for pp,g in ev.groupby('pair'):
        c=g[g.confirm==1]['rev_ret_12']; o=g[g.confirm==0]['rev_ret_12']
        print(f'  {pp:20s} n={len(g):3d} conf={c.mean():+.3f}(n{len(c)}) opp={o.mean():+.3f}(n{len(o)}) '
              f'diff(opp-conf)={o.mean()-c.mean():+.3f}')

    # ---- Filter sim (N=12): fade ALL extremes vs fade only when flow does NOT confirm ----
    print('\n'+'='*70)
    print('FILTER SIM (fade extreme, hold N=12h). PnL = rev_ret (z-units toward reversion).')
    print('Strategy A: fade ALL extremes. Strategy B: fade only when flow does NOT confirm.')
    print('='*70)
    for N in HORIZONS:
        rr=f'rev_ret_{N}'
        allret=ev[rr]; filt=ev[ev.confirm==0][rr]
        print(f'  N={N}h  ALL: n={len(allret)} mean={allret.mean():+.4f} '
              f'win%={ (allret>0).mean()*100:.1f} sharpe~={allret.mean()/allret.std():+.3f}')
        print(f'        FILT: n={len(filt)} mean={filt.mean():+.4f} '
              f'win%={ (filt>0).mean()*100:.1f} sharpe~={filt.mean()/filt.std():+.3f}')
    ev.to_parquet('scratch/extreme_events.parquet')

if __name__=='__main__':
    np.random.seed(7)
    main()
