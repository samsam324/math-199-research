"""Build hourly mid + signed-flow series per symbol, cache to scratch/cache_hourly."""
import pandas as pd, numpy as np, glob, os

OUT = 'scratch/cache_hourly'
os.makedirs(OUT, exist_ok=True)

def sym_dates(sym):
    dl = set(os.path.basename(x).replace('.parquet','') for x in glob.glob(f'data/l2/{sym}/*.parquet'))
    dt = set(os.path.basename(x).replace('.parquet','') for x in glob.glob(f'data/trades/{sym}/*.parquet'))
    return sorted(dl & dt)

def build_symbol(sym):
    out = f'{OUT}/{sym}.parquet'
    if os.path.exists(out):
        return pd.read_parquet(out)
    rows = []
    for d in sym_dates(sym):
        l2 = pd.read_parquet(f'data/l2/{sym}/{d}.parquet', columns=['timestamp','bid_px_1','ask_px_1'])
        tr = pd.read_parquet(f'data/trades/{sym}/{d}.parquet',
                             columns=['timestamp','signed_volume_base','volume_base','trade_count'])
        l2 = l2.set_index('timestamp').sort_index()
        tr = tr.set_index('timestamp').sort_index()
        mid = (l2['bid_px_1'] + l2['ask_px_1']) / 2.0
        # hourly mid = last 1s mid in the hour (last observation)
        h_mid = mid.resample('1h').last()
        # hourly aggregated flow
        h_sv = tr['signed_volume_base'].resample('1h').sum()
        h_v  = tr['volume_base'].resample('1h').sum()
        h_tc = tr['trade_count'].resample('1h').sum()
        df = pd.DataFrame({'mid': h_mid, 'signed_vol': h_sv, 'vol': h_v, 'tcount': h_tc})
        rows.append(df)
    full = pd.concat(rows).sort_index()
    # collapse duplicate hour boundaries across day files
    full = full.groupby(level=0).agg({'mid':'last','signed_vol':'sum','vol':'sum','tcount':'sum'})
    full.to_parquet(out)
    return full

if __name__ == '__main__':
    syms = sorted(set(s for p in [
        'BTCUSDT','ETHUSDT','SHIBUSDT','DOGEUSDT','ADAUSDT','DOTUSDT','SOLUSDT','AVAXUSDT',
        'XRPUSDT','XLMUSDT','LINKUSDT','MANAUSDT','SANDUSDT'] for s in [p]))
    for s in syms:
        df = build_symbol(s)
        print(f'{s}: {len(df)} hourly bars, {df.index.min()} -> {df.index.max()}, '
              f'mid_na={df.mid.isna().sum()}')
