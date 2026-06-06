import warnings, glob, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\jackw\Desktop\math-199-research'
DATES = [f'2024-01-{d:02d}' for d in range(2, 12)]   # 02..11
HORIZONS = [1, 5, 10, 30, 60, 300]
OFI_WIN = 10  # trailing seconds

PAIRS = [
    ('BTCUSDT', 'ETHUSDT', 0.976922),
    ('SOLUSDT', 'AVAXUSDT', 1.304799),
    ('ADAUSDT', 'DOTUSDT', 0.852796),
    ('ETHUSDT', 'DOTUSDT', 1.138692),
    ('ETHUSDT', 'ADAUSDT', 1.276046),
]

def load_l2(sym, date):
    p = os.path.join(ROOT, 'data', 'l2', sym, date + '.parquet')
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p, columns=['timestamp','bid_px_1','ask_px_1','bid_sz_1','ask_sz_1'])
    df = df.set_index('timestamp')
    den = df['bid_sz_1'] + df['ask_sz_1']
    df['micro'] = np.where(den>0, (df['bid_sz_1']*df['ask_px_1'] + df['ask_sz_1']*df['bid_px_1'])/den,
                           (df['bid_px_1']+df['ask_px_1'])/2)
    df['mid'] = (df['bid_px_1']+df['ask_px_1'])/2
    return df[['micro','mid']]

def load_tr(sym, date):
    p = os.path.join(ROOT, 'data', 'trades', sym, date + '.parquet')
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p, columns=['timestamp','signed_volume_base','volume_base'])
    return df.set_index('timestamp')

def build_day(a, b, beta, date):
    la, lb = load_l2(a,date), load_l2(b,date)
    ta, tb = load_tr(a,date), load_tr(b,date)
    if any(x is None for x in (la,lb,ta,tb)):
        return None
    # common 1s grid for the day
    grid = pd.date_range(date+' 00:00:00+00:00', date+' 23:59:59+00:00', freq='1s')
    def reidx(df):
        return df[~df.index.duplicated(keep='last')].reindex(grid).ffill(limit=5)
    la, lb = reidx(la), reidx(lb)
    # trades: reindex, missing second = no trade -> 0 flow
    def reidx_tr(df):
        df = df[~df.index.duplicated(keep='last')].reindex(grid)
        df['signed_volume_base'] = df['signed_volume_base'].fillna(0.0)
        df['volume_base'] = df['volume_base'].fillna(0.0)
        return df
    ta, tb = reidx_tr(ta), reidx_tr(tb)

    out = pd.DataFrame(index=grid)
    out['spread_micro'] = np.log(la['micro']) - beta*np.log(lb['micro'])
    out['spread_mid']   = np.log(la['mid'])   - beta*np.log(lb['mid'])
    # normalized signed flow per leg (signed/abs volume), 0 when no volume
    sa = np.where(ta['volume_base']>0, ta['signed_volume_base']/ta['volume_base'], 0.0)
    sb = np.where(tb['volume_base']>0, tb['signed_volume_base']/tb['volume_base'], 0.0)
    out['ofi_norm_inst'] = sa - beta*sb
    out['ofi_raw_inst']  = ta['signed_volume_base'].values - beta*tb['signed_volume_base'].values
    out['valid_px'] = la['micro'].notna().values & lb['micro'].notna().values
    return out

def hac_reg(y, x, maxlags):
    X = sm.add_constant(x)
    m = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags':maxlags})
    return m.params.iloc[1], m.tvalues.iloc[1], m.rsquared, int(m.nobs)

def analyze_pair(a, b, beta):
    days = []
    for d in DATES:
        day = build_day(a,b,beta,d)
        if day is not None:
            days.append(day)
    if not days:
        return None
    df = pd.concat(days)
    # trailing-10s OFI (within concatenated series; small day-boundary contamination negligible)
    df['OFI'] = df['ofi_norm_inst'].rolling(OFI_WIN, min_periods=1).sum()
    df['OFI_raw'] = df['ofi_raw_inst'].rolling(OFI_WIN, min_periods=1).sum()
    sp = df['spread_micro'].where(df['valid_px'])
    res = {}
    # contemporaneous: spread return over [t-? ] use same-second? define contemp as [t-1,t]
    ret_c = (sp - sp.shift(1)) * 1e4  # bps, 1s contemporaneous move
    b0,t0,r0,n0 = hac_reg(ret_c, df['OFI'], 5)
    res['contemp'] = (b0,t0,r0,n0)
    for h in HORIZONS:
        fut = (sp.shift(-h) - sp) * 1e4  # bps future spread return [t,t+h]
        bb,tt,rr,nn = hac_reg(fut, df['OFI'], max(h,1))
        res[h] = (bb,tt,rr,nn)
    # raw-OFI robustness at a few horizons
    raw = {}
    for h in [1,10,60]:
        fut = (sp.shift(-h) - sp) * 1e4
        raw[h] = hac_reg(fut, df['OFI_raw'], max(h,1))
    # mid-spread robustness
    spm = df['spread_mid'].where(df['valid_px'])
    midr = {}
    for h in [1,10,60]:
        fut = (spm.shift(-h) - spm) * 1e4
        midr[h] = hac_reg(fut, df['OFI'], max(h,1))
    return res, raw, midr, len(df)

print(f"Window: {DATES[0]}..{DATES[-1]}  OFI=trailing{OFI_WIN}s normalized signed flow, spread in bps\n")
for a,b,beta in PAIRS:
    r = analyze_pair(a,b,beta)
    if r is None:
        print(f"=== {a}/{b}: SKIP (missing data) ==="); continue
    res, raw, midr, n = r
    print(f"=== {a}/{b}  (beta={beta:.4f}, N~{n} sec-obs) ===")
    print(f"{'horizon':>9} {'beta':>12} {'HAC-t':>9} {'R^2':>10} {'N':>9}")
    b0,t0,r0,n0 = res['contemp']
    print(f"{'contemp':>9} {b0:12.4e} {t0:9.2f} {r0:10.5f} {n0:9d}")
    for h in HORIZONS:
        bb,tt,rr,nn = res[h]
        star = '*' if abs(tt)>2.58 else (' ' if abs(tt)<1.96 else '.')
        print(f"{str(h)+'s':>9} {bb:12.4e} {tt:9.2f} {rr:10.5f} {nn:9d} {star}")
    print("  raw-OFI robustness [beta,t,R2]:", {h:(f'{v[0]:.2e}',round(v[1],2),round(v[2],5)) for h,v in raw.items()})
    print("  mid-spread robustness [beta,t,R2]:", {h:(f'{v[0]:.2e}',round(v[1],2),round(v[2],5)) for h,v in midr.items()})
    print()
