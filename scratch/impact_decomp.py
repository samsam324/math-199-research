"""
Permanent vs transient price-impact decomposition via per-trade (event-time)
response functions, testing whether trade SIZE proxies for INFORMATION.

Event unit: a marketable order = all fills sharing (timestamp, side) [one
aggressor sweeping levels]. Classified by USD notional into retail/mid/institutional.

For each order at second t with aggressor sign s (+1 buy / -1 sell):
    R(h) = s * (mid(t+h) - mid(t)) / mid(t)            (signed return, in bps)
mid(t) = forward-filled 1s mid at the order's second (at/just-before).
Averaged within class -> R_c(h). Also normalized per $1M notional.

Decompose: PEAK = max_h R_c(h); PERMANENT = R_c(300); TRANSIENT = peak - permanent.
SEs: block bootstrap over (day) clusters at key horizons.
"""
import pandas as pd, numpy as np, gzip, os, sys

SYMS = ['BTCUSDT', 'ETHUSDT']
DATES = [f'2024-01-{d:02d}' for d in range(2, 12)]  # 2024-01-02 .. 2024-01-11
HORIZONS = [0, 1, 2, 5, 10, 30, 60, 120, 300]
THRESH = (1000.0, 10000.0)  # retail <1k, mid 1k-10k, inst >10k
CLASSES = ['retail', 'mid', 'institutional']
ROOT = r'C:\Users\jackw\Desktop\math-199-research'

def classify(notional):
    c = np.full(len(notional), 'mid', dtype=object)
    c[notional < THRESH[0]] = 'retail'
    c[notional >= THRESH[1]] = 'institutional'
    return c

def load_mid_grid(sym, date):
    """1s forward-filled mid on a UTC-second integer grid."""
    b = pd.read_parquet(os.path.join(ROOT, 'data', 'l2', sym, f'{date}.parquet'),
                        columns=['timestamp', 'bid_px_1', 'ask_px_1'])
    b = b.dropna(subset=['bid_px_1', 'ask_px_1'])
    mid = (b['bid_px_1'].values + b['ask_px_1'].values) / 2.0
    # second-of-day epoch (UTC). timestamp is tz-aware ns.
    sec = (b['timestamp'].view('int64').values // 1_000_000_000)
    # collapse to last obs per second, then ffill onto contiguous grid
    s = pd.Series(mid, index=sec)
    s = s.groupby(level=0).last()
    full = np.arange(sec.min(), sec.max() + 1)
    grid = s.reindex(full).ffill()
    return grid  # index = epoch second, value = mid

def load_orders(sym, date):
    """Aggregate fills into marketable orders by (timestamp, side)."""
    t = pd.read_csv(os.path.join(ROOT, 'data', 'l2_raw', 'binance', 'trades', sym, f'{date}.csv.gz'),
                    usecols=['timestamp', 'side', 'price', 'amount'])
    t['notional'] = t['price'] * t['amount']
    t['qty'] = t['amount']
    g = t.groupby(['timestamp', 'side'], sort=True).agg(
        notional=('notional', 'sum'), qty=('qty', 'sum'),
        px=('price', 'last')).reset_index()
    g['sec'] = g['timestamp'] // 1_000_000  # us -> s
    g['sign'] = np.where(g['side'].values == 'buy', 1.0, -1.0)
    g['cls'] = classify(g['notional'].values)
    return g

def compute_responses(sym):
    rows = []  # per-order: day, cls, notional, sign, R(h)... in return units
    share_rows = []
    for date in DATES:
        grid = load_mid_grid(sym, date)
        o = load_orders(sym, date)
        gidx = grid.index.values
        gval = grid.values
        lo, hi = gidx[0], gidx[-1]
        sec = o['sec'].values
        # need t and t+max(h) inside grid
        valid = (sec >= lo) & (sec + HORIZONS[-1] <= hi)
        o = o[valid].reset_index(drop=True)
        sec = o['sec'].values
        pos = sec - lo  # integer offset into contiguous grid
        mid_t = gval[pos]
        R = {}
        for h in HORIZONS:
            R[h] = o['sign'].values * (gval[pos + h] - mid_t) / mid_t
        df = pd.DataFrame({'day': date, 'cls': o['cls'].values,
                           'notional': o['notional'].values, 'sign': o['sign'].values})
        for h in HORIZONS:
            df[f'R{h}'] = R[h]
        rows.append(df)
        # shares
        n = o['notional'].values
        c = o['cls'].values
        for cl in CLASSES:
            m = c == cl
            share_rows.append({'day': date, 'cls': cl, 'cnt': m.sum(),
                               'vol': n[m].sum()})
    allr = pd.concat(rows, ignore_index=True)
    shares = pd.DataFrame(share_rows)
    return allr, shares

def summarize(allr):
    """Per-class mean response (bps) and per-$1M (bps per $1M)."""
    out = {}
    for cl in CLASSES:
        d = allr[allr['cls'] == cl]
        raw = {h: d[f'R{h}'].mean() * 1e4 for h in HORIZONS}  # bps
        # per-$1M: R(h)/notional averaged, scaled to 1e6 notional, in bps
        perM = {h: (d[f'R{h}'] / d['notional']).mean() * 1e6 * 1e4 for h in HORIZONS}
        out[cl] = {'n': len(d), 'mean_notional': d['notional'].mean(),
                   'raw': raw, 'perM': perM}
    return out

def block_bootstrap(allr, key_h=(1, 10, 60, 300), B=500, seed=7):
    """Cluster bootstrap over days for raw and per-$1M responses."""
    rng = np.random.default_rng(seed)
    days = sorted(allr['day'].unique())
    # pre-split by day for speed
    byday = {d: allr[allr['day'] == d] for d in days}
    res = {cl: {'raw': {h: [] for h in key_h}, 'perM': {h: [] for h in key_h}}
           for cl in CLASSES}
    for _ in range(B):
        pick = rng.choice(days, size=len(days), replace=True)
        samp = pd.concat([byday[d] for d in pick], ignore_index=True)
        for cl in CLASSES:
            d = samp[samp['cls'] == cl]
            for h in key_h:
                res[cl]['raw'][h].append(d[f'R{h}'].mean() * 1e4)
                res[cl]['perM'][h].append((d[f'R{h}'] / d['notional']).mean() * 1e6 * 1e4)
    se = {cl: {'raw': {h: np.std(res[cl]['raw'][h], ddof=1) for h in key_h},
               'perM': {h: np.std(res[cl]['perM'][h], ddof=1) for h in key_h}}
          for cl in CLASSES}
    return se

def main():
    for sym in SYMS:
        print('=' * 78)
        print(f'SYMBOL: {sym}   days {DATES[0]}..{DATES[-1]}  (event-time, no freq confound)')
        print('=' * 78)
        allr, shares = compute_responses(sym)
        # Step 1: shares
        sg = shares.groupby('cls').agg(cnt=('cnt', 'sum'), vol=('vol', 'sum'))
        sg = sg.reindex(CLASSES)
        sg['cnt_share'] = sg['cnt'] / sg['cnt'].sum()
        sg['vol_share'] = sg['vol'] / sg['vol'].sum()
        print('\n[1] CLASS SHARES (marketable orders aggregated by ts+side)')
        print(sg[['cnt', 'cnt_share', 'vol', 'vol_share']].to_string(
            float_format=lambda x: f'{x:,.4f}' if x < 1 else f'{x:,.0f}'))
        print(f'  total orders: {sg["cnt"].sum():,.0f}')

        summ = summarize(allr)
        # Step 2/3: response tables
        print('\n[2/3] RESPONSE FUNCTIONS R_c(h)')
        hdr = 'h(s) ' + ''.join(f'{h:>9}' for h in HORIZONS)
        for unit, key in [('RAW (bps)', 'raw'), ('per-$1M (bps/$1M)', 'perM')]:
            print(f'  -- {unit} --')
            print('   ' + hdr)
            for cl in CLASSES:
                vals = summ[cl][key]
                print(f'   {cl:13s}' + ''.join(f'{vals[h]:9.3f}' for h in HORIZONS))
        # Step 4: decomposition
        print('\n[4] PERMANENT vs TRANSIENT (permanent = R(300))')
        for unit, key in [('RAW bps', 'raw'), ('per-$1M', 'perM')]:
            print(f'  -- {unit} --')
            print(f'   {"class":13s}{"n":>9}{"meanNotnl":>12}{"peak":>9}{"peak_h":>7}{"perm":>9}{"transient":>11}{"%perm":>8}')
            for cl in CLASSES:
                v = summ[cl][key]
                peak_h = max(HORIZONS, key=lambda h: v[h])
                peak = v[peak_h]
                perm = v[300]
                trans = peak - perm
                pct = 100 * perm / peak if peak != 0 else float('nan')
                print(f'   {cl:13s}{summ[cl]["n"]:9,d}{summ[cl]["mean_notional"]:12,.0f}'
                      f'{peak:9.3f}{peak_h:7d}{perm:9.3f}{trans:11.3f}{pct:8.1f}')
        # Step 5: bootstrap SEs
        se = block_bootstrap(allr)
        print('\n[5] CLUSTER-BOOTSTRAP SEs (over 10 days, B=500) at key horizons')
        for unit, key in [('RAW bps', 'raw'), ('per-$1M', 'perM')]:
            print(f'  -- {unit}: mean (se) --')
            print('   ' + 'class        ' + ''.join(f'{"h="+str(h):>18}' for h in (1,10,60,300)))
            for cl in CLASSES:
                v = summ[cl][key]
                cells = ''.join(f'{v[h]:9.3f} ({se[cl][key][h]:6.3f})' for h in (1,10,60,300))
                print(f'   {cl:13s}' + cells)
        sys.stdout.flush()

if __name__ == '__main__':
    main()
