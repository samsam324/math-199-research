import pandas as pd, numpy as np, os
import statsmodels.api as sm
ROOT=r'C:\Users\jackw\Desktop\math-199-research'
DATES=[f'2024-01-{d:02d}' for d in range(2,12)]
HOR=[1,5,10,30,60,300]
def load(sym):
    P=[]
    for d in DATES:
        bp=os.path.join(ROOT,'data','l2',sym,d+'.parquet'); tp=os.path.join(ROOT,'data','trades',sym,d+'.parquet')
        if not(os.path.exists(bp) and os.path.exists(tp)):continue
        b=pd.read_parquet(bp,columns=['timestamp','bid_px_1','ask_px_1']); t=pd.read_parquet(tp,columns=['timestamp','signed_volume_base'])
        b['mid']=(b.bid_px_1+b.ask_px_1)/2
        m=pd.merge(b[['timestamp','mid']],t[['timestamp','signed_volume_base']],on='timestamp',how='left')
        m['signed_volume_base']=m.signed_volume_base.fillna(0.0); P.append(m)
    g=pd.concat(P).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    g['ret1']=np.log(g.mid).diff().shift(-1)   # return over [t,t+1]
    return g
def hac(x,y,L):
    msk=np.isfinite(x)&np.isfinite(y); x,y=x[msk],y[msk]
    X=sm.add_constant(x); r=sm.OLS(y,X).fit(cov_type='HAC',cov_kwds={'maxlags':max(1,L)})
    return r.params[1], r.tvalues[1], r.rsquared, int(r.nobs)
for sym in ['BTCUSDT','ETHUSDT']:
    g=load(sym); lr=np.log(g.mid)
    trail=g.signed_volume_base.rolling(10,min_periods=1).sum().to_numpy()
    print(f"\n=== {sym}  (n~{len(g):,}, trailing-10s flow, Newey-West HAC) ===")
    print(f"{'h(s)':>5} {'beta':>12} {'HAC-t':>9} {'R2':>9}")
    for h in HOR:
        fwd=(lr.shift(-h)-lr).to_numpy()   # future return [t,t+h]
        b,t,r2,n=hac(trail,fwd,h)
        print(f"{h:>5} {b:>12.3e} {t:>9.1f} {r2:>9.5f}")
