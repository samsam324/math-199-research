"""Robustness: unconditional reversion significance, and stability of the flow-conditioning
edge across flow-window length and z-threshold. Cluster-by-pair bootstrap throughout."""
import pandas as pd, numpy as np
import importlib.util
spec=importlib.util.spec_from_file_location('efa','scratch/extreme_flow_analyze.py')
m=importlib.util.module_from_spec(spec)
# we re-implement minimal build to vary params without import side effects
exec(open('scratch/extreme_flow_analyze.py').read().split('def main()')[0])

def run(zthr, flow_l, cooldown=24):
    rows=[]
    for a,b,beta in PAIRS:
        df=build_pair(a,b,beta).dropna(subset=['z','sd','ofi'])
        # recompute ofi over last flow_l hours (rolling sum of signed/scale already 1h; aggregate)
        if flow_l>1:
            df=df.copy()
            df['ofi']=df['ofi'].rolling(flow_l).sum()
            df=df.dropna(subset=['ofi'])
        z=df['z'].values; spread=df['spread'].values; sd=df['sd'].values; ofi=df['ofi'].values
        cross=(np.abs(z)>zthr)&(np.abs(np.roll(z,1))<=zthr); cross[0]=False
        n=len(df); last=-10**9
        for i in np.where(cross)[0]:
            if i-last<cooldown or i+24>=n: continue
            last=i; sgn=np.sign(z[i])
            rows.append({'pair':f'{a}_{b}','confirm':int(np.sign(ofi[i])==sgn),
                         'ofi_align':ofi[i]*sgn,
                         'rev_ret_12':(-sgn*(spread[i+12]-spread[i]))/sd[i]})
    return pd.DataFrame(rows)

def boot_mean(v, pair, reps=3000):
    ps=pair.unique(); ms=[]
    d=pd.DataFrame({'v':v.values,'p':pair.values})
    for _ in range(reps):
        s=np.random.choice(ps,len(ps),replace=True)
        ms.append(pd.concat([d[d.p==pp] for pp in s])['v'].mean())
    ms=np.array(ms); return v.mean(), np.percentile(ms,2.5), np.percentile(ms,97.5)

np.random.seed(11)
ev=pd.read_parquet('scratch/extreme_events.parquet')
print('UNCONDITIONAL reversion edge (is fading extremes even profitable gross?), N=12h')
mu,lo,hi=boot_mean(ev['rev_ret_12'],ev['pair'])
print(f'  mean rev_ret_12 = {mu:+.4f} z   cluster95%CI [{lo:+.4f},{hi:+.4f}]  '
      f'-> {"SIG>0" if lo>0 else "NOT sig"}')
print(f'  reversion rate  = {ev.reverted_12.mean():.3f}\n')

print('STABILITY of flow edge (oppose-minus-confirm mean rev_ret_12) across specs:')
print(f"{'zthr':>5}{'flowL':>6}{'nEv':>6}{'nConf':>6}{'nOpp':>5}{'diff(opp-conf)':>16}{'95%CI':>22}")
for zthr in [1.5,2.0,2.5]:
    for fl in [1,3,6]:
        e=run(zthr,fl)
        if len(e)<20:
            print(f'{zthr:5}{fl:6}{len(e):6}  too few'); continue
        c=e[e.confirm==1]['rev_ret_12']; o=e[e.confirm==0]['rev_ret_12']
        # cluster bootstrap on the difference
        ps=e.pair.unique(); ds=[]
        for _ in range(2000):
            s=np.random.choice(ps,len(ps),replace=True)
            d=pd.concat([e[e.pair==pp] for pp in s])
            ca=d[d.confirm==1]['rev_ret_12']; oa=d[d.confirm==0]['rev_ret_12']
            if len(ca)<3 or len(oa)<3: continue
            ds.append(oa.mean()-ca.mean())
        ds=np.array(ds); diff=o.mean()-c.mean()
        sig='*' if (np.percentile(ds,2.5)>0 or np.percentile(ds,97.5)<0) else ''
        print(f'{zthr:5}{fl:6}{len(e):6}{(e.confirm==1).sum():6}{(e.confirm==0).sum():5}'
              f'{diff:+16.4f}  [{np.percentile(ds,2.5):+.3f},{np.percentile(ds,97.5):+.3f}]{sig}')
