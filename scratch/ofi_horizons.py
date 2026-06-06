import pandas as pd, numpy as np, glob, os

DATES = ['2024-01-02','2024-01-03','2024-01-04','2024-01-05','2024-01-06',
         '2024-01-07','2024-01-08','2024-01-09','2024-01-10','2024-01-11']
HORIZONS = [1,5,10,30,60,300]
TRAIL = [10,60]
ROOT = r'C:\Users\jackw\Desktop\math-199-research'

def ols(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 50: return (np.nan,)*4
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = (resid @ resid) / (n - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))[1]
    t = beta[1] / se if se > 0 else np.nan
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return beta[1], t, r2, n

def load_sym(sym):
    parts = []
    for d in DATES:
        bp = os.path.join(ROOT, 'data', 'l2', sym, d + '.parquet')
        tp = os.path.join(ROOT, 'data', 'trades', sym, d + '.parquet')
        if not (os.path.exists(bp) and os.path.exists(tp)): continue
        b = pd.read_parquet(bp, columns=['timestamp','bid_px_1','ask_px_1'])
        t = pd.read_parquet(tp, columns=['timestamp','signed_volume_base'])
        b['mid'] = (b.bid_px_1 + b.ask_px_1)/2
        # reindex onto full 1s grid for the day
        g = pd.date_range(d+' 00:00:00+00:00', periods=86400, freq='1s')
        b = b.set_index('timestamp').reindex(g)
        t = t.set_index('timestamp').reindex(g)
        df = pd.DataFrame(index=g)
        df['mid'] = b['mid'].ffill()
        df['sv'] = t['signed_volume_base'].fillna(0.0)
        df['date'] = d
        parts.append(df)
    return parts  # list of per-day frames (don't cross day boundary)

def analyze(sym):
    days = load_sym(sym)
    print(f"\n===== {sym}  ({len(days)} days) =====")
    # precompute per-day
    rows = []
    for trail in TRAIL:
        # build OFI trailing-sum and contemporaneous + future returns per day, then pool
        allx = {h: [] for h in HORIZONS}
        ally = {h: [] for h in HORIZONS}
        contemp_x, contemp_y = [], []   # contemporaneous: return over [t,t+1] vs sv at t
        for df in days:
            logmid = np.log(df['mid'].values)
            sv = df['sv'].values
            ofi_trail = pd.Series(sv).rolling(trail, min_periods=1).sum().values
            for h in HORIZONS:
                fut = np.empty(len(logmid)); fut[:] = np.nan
                fut[:-h] = logmid[h:] - logmid[:-h]
                allx[h].append(ofi_trail)
                ally[h].append(fut)
            if trail == TRAIL[0]:
                c = np.empty(len(logmid)); c[:] = np.nan
                c[:-1] = logmid[1:] - logmid[:-1]
                contemp_x.append(sv); contemp_y.append(c)
        for h in HORIZONS:
            x = np.concatenate(allx[h]); y = np.concatenate(ally[h])
            b,t,r2,n = ols(x,y)
            rows.append((trail,h,b,t,r2,n))
        if trail == TRAIL[0]:
            cx = np.concatenate(contemp_x); cy = np.concatenate(contemp_y)
            cb,ct,cr2,cn = ols(cx,cy)
            print(f"CONTEMPORANEOUS (price impact, sv[t] vs ret[t,t+1]): slope={cb:.3e} t={ct:.1f} R2={cr2:.5f} n={cn}")
    print(f"{'trail':>5} {'h(s)':>5} {'slope':>12} {'t-stat':>8} {'R2':>9} {'n':>9}")
    for trail,h,b,t,r2,n in rows:
        print(f"{trail:>5} {h:>5} {b:>12.3e} {t:>8.1f} {r2:>9.5f} {n:>9}")

for s in ['BTCUSDT','ETHUSDT']:
    analyze(s)
