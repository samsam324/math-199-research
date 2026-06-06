import pandas as pd, numpy as np, os

ROOT = r'C:\Users\jackw\Desktop\math-199-research'
DATE = '2024-01-02'
SYM = 'BTCUSDT'

raw = pd.read_csv(os.path.join(ROOT,'data','l2_raw','binance','trades',SYM,DATE+'.csv.gz'),
                  usecols=['timestamp','side','price','amount'])
raw['usd'] = raw.price * raw.amount
raw['signed'] = np.where(raw.side=='buy', raw.amount, -raw.amount)  # signed base
raw['signed_usd'] = np.where(raw.side=='buy', raw.usd, -raw.usd)

u = raw.usd.values
tot = u.sum()
print(f"=== {SYM} {DATE} raw trades: n={len(raw):,}  total USD vol=${tot/1e9:.3f}B ===")
for p in [50,90,99,99.9]:
    print(f"  p{p}: ${np.percentile(u,p):,.2f}")
print(f"  mean ${u.mean():,.2f}  max ${u.max():,.2f}")
print("Fraction of TOTAL USD volume by bucket:")
for thr in [10000,50000,100000]:
    f = u[u>thr].sum()/tot
    cnt = (u>thr).mean()
    print(f"  > ${thr:>7,}: {f*100:6.2f}% of vol,  {cnt*100:6.3f}% of trades")
print(f"  < $1,000   : {u[u<1000].sum()/tot*100:6.2f}% of vol,  {(u<1000).mean()*100:6.2f}% of trades")

# ---- Part 4: large vs small signed flow vs mid moves, on 1s grid ----
raw['ts'] = pd.to_datetime(raw.timestamp, unit='us', utc=True).dt.floor('1s')
large = raw[raw.usd>10000].groupby('ts').signed.sum()
small = raw[raw.usd<=10000].groupby('ts').signed.sum()

b = pd.read_parquet(os.path.join(ROOT,'data','l2',SYM,DATE+'.parquet'),
                    columns=['timestamp','bid_px_1','ask_px_1'])
b['mid'] = (b.bid_px_1+b.ask_px_1)/2
g = pd.date_range(DATE+' 00:00:00+00:00', periods=86400, freq='1s')
mid = b.set_index('timestamp')['mid'].reindex(g).ffill()
logmid = np.log(mid.values)

df = pd.DataFrame(index=g)
df['large'] = large.reindex(g).fillna(0.0).values
df['small'] = small.reindex(g).fillna(0.0).values
df['ret1'] = np.r_[np.diff(logmid), np.nan]              # contemporaneous [t,t+1]
fut10 = np.empty(len(logmid)); fut10[:]=np.nan
fut10[:-10] = logmid[10:]-logmid[:-10]
df['ret10'] = fut10

def corr(a,b):
    m=np.isfinite(a)&np.isfinite(b); return np.corrcoef(a[m],b[m])[0,1]

print("\n=== Large (>$10k) vs Small (<=$10k) signed base flow, 1s grid ===")
print(f"large flow share of |signed| base vol: {df.large.abs().sum()/(df.large.abs().sum()+df.small.abs().sum())*100:.1f}%")
for lbl,col in [('contemporaneous ret[t,t+1]','ret1'),('next-10s ret[t,t+10]','ret10')]:
    print(f"  {lbl}:  corr(large)={corr(df.large.values,df[col].values):.4f}   corr(small)={corr(df.small.values,df[col].values):.4f}")
