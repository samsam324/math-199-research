"""
Build hourly panel: RV (microstructure-noise-robust) + microstructure features.
Saves scratch/har_panel.parquet for the regression step.
RV estimator: realized vol from 1-minute log returns of mid, summed over the hour
(sqrt of sum of squared 1-min returns). 1-min sampling mitigates 1s microstructure noise.
Cross-check: 5-min subsampled estimator also computed.
"""
import pandas as pd, numpy as np, glob, os, sys

SYMS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','LINKUSDT','AVAXUSDT']
START = '2024-01-02'
NDAYS = 90  # keep runtime bounded

def daydates(sym):
    fl = set(os.path.basename(f)[:-8] for f in glob.glob(f'data/l2/{sym}/*.parquet'))
    ft = set(os.path.basename(f)[:-8] for f in glob.glob(f'data/trades/{sym}/*.parquet'))
    both = sorted(fl & ft)
    both = [d for d in both if d >= START]
    return both[:NDAYS]

def build_sym(sym):
    dates = daydates(sym)
    rows = []
    for d in dates:
        try:
            l2 = pd.read_parquet(f'data/l2/{sym}/{d}.parquet',
                columns=['timestamp','bid_px_1','ask_px_1','bid_sz_1','ask_sz_1'])
            tr = pd.read_parquet(f'data/trades/{sym}/{d}.parquet')
        except Exception as e:
            continue
        l2 = l2.set_index('timestamp').sort_index()
        l2 = l2[~l2.index.duplicated(keep='last')]
        mid = (l2['bid_px_1'] + l2['ask_px_1'])/2.0
        spread = (l2['ask_px_1'] - l2['bid_px_1'])/mid          # relative spread
        imb = (l2['bid_sz_1'] - l2['ask_sz_1'])/(l2['bid_sz_1']+l2['ask_sz_1']+1e-12)
        # full-day 1s grid
        grid = pd.date_range(d+' 00:00:00+00:00', periods=86400, freq='1s')
        mid = mid.reindex(grid).ffill()
        spread = spread.reindex(grid).ffill()
        imb = imb.reindex(grid).ffill()
        logmid = np.log(mid)
        # 1-min and 5-min mid (last obs in bucket)
        m1 = logmid.resample('1min').last()
        m5 = logmid.resample('5min').last()
        r1 = m1.diff()
        r5 = m5.diff()
        # quote-update flag: mid changed within the 1s bar
        midchg = (mid.diff().abs() > 0).astype(float)

        tr = tr.set_index('timestamp').sort_index()
        tr = tr[~tr.index.duplicated(keep='last')]
        tr = tr.reindex(grid).fillna(0.0)

        # hourly aggregation
        hr = pd.Grouper(freq='1h')
        # RV from 1-min returns (sum of squared 1-min log returns per hour), as vol (sqrt)
        rv1_var = (r1**2).groupby(pd.Grouper(freq='1h')).sum()
        rv5_var = (r5**2).groupby(pd.Grouper(freq='1h')).sum()
        # bipower variation (jump-robust) from 1-min returns
        ar = r1.abs()
        bpv = (np.pi/2) * (ar * ar.shift(1)).groupby(pd.Grouper(freq='1h')).sum()
        # jump proxy = max(RV-BPV,0)
        jump = (rv1_var - bpv).clip(lower=0)

        g_mid = midchg.groupby(pd.Grouper(freq='1h')).sum()          # quote-update intensity
        g_sprm = spread.groupby(pd.Grouper(freq='1h')).mean()
        g_sprs = spread.groupby(pd.Grouper(freq='1h')).std()
        g_imbm = imb.groupby(pd.Grouper(freq='1h')).mean()
        g_imbs = imb.groupby(pd.Grouper(freq='1h')).std()

        g_cnt = tr['trade_count'].groupby(pd.Grouper(freq='1h')).sum()
        g_vol = tr['volume_base'].groupby(pd.Grouper(freq='1h')).sum()
        g_sgn = tr['signed_volume_base'].groupby(pd.Grouper(freq='1h')).sum()
        g_buy = tr['buy_volume_base'].groupby(pd.Grouper(freq='1h')).sum()
        g_sell = tr['sell_volume_base'].groupby(pd.Grouper(freq='1h')).sum()
        # VPIN-like toxicity: |buy-sell|/(buy+sell) aggregated over hour
        tot = (g_buy + g_sell).replace(0, np.nan)
        vpin = (g_buy - g_sell).abs()/tot
        absflow = g_sgn.abs()

        h = pd.DataFrame({
            'rv1_var': rv1_var, 'rv5_var': rv5_var, 'bpv': bpv, 'jump': jump,
            'qupd': g_mid, 'spr_m': g_sprm, 'spr_s': g_sprs,
            'imb_m': g_imbm, 'imb_s': g_imbs,
            'tcnt': g_cnt, 'tvol': g_vol, 'absflow': absflow,
            'vpin': vpin,
        })
        rows.append(h)
    if not rows:
        return None
    out = pd.concat(rows).sort_index()
    out['sym'] = sym
    out['rv'] = np.sqrt(out['rv1_var'].clip(lower=0))     # hourly realized vol (1-min based)
    out['rv5'] = np.sqrt(out['rv5_var'].clip(lower=0))
    return out

allp = []
for s in SYMS:
    p = build_sym(s)
    if p is not None:
        allp.append(p)
        print(s, 'hours', len(p), 'rv mean', round(p['rv'].mean(),6))
panel = pd.concat(allp)
panel.to_parquet('scratch/har_panel.parquet')
print('TOTAL hours', len(panel), 'saved scratch/har_panel.parquet')
print('corr rv(1min) vs rv(5min):', round(panel['rv'].corr(panel['rv5']),4))
