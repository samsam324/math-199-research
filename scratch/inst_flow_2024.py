"""
Institutional (large-trade) signed order flow -> tradeable-horizon (15min, 1h) return.

CORE: regress fwd mid log-return over [t,t+H] on institutional signed-flow share over
[t-H,t]; add controls (total signed flow + lagged/momentum return); report incremental R2
of inst flow OVER (total flow + momentum). Informed-vs-noise: inst+mid+retail together.
HAC SEs. Pooled (symbol-demeaned) + per-symbol. Tradeability: bps per +1 SD inst flow.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

SYMS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
# Full-2024 sample spanning all 12 months incl. volatile regimes
# (2024-08-05 yen-carry crash, 2024-03-13 BTC-near-ATH).
DATES = ['2024-01-15', '2024-02-15', '2024-03-13', '2024-04-16', '2024-05-15',
         '2024-06-14', '2024-07-15', '2024-08-05', '2024-08-06', '2024-09-16',
         '2024-10-15', '2024-11-12', '2024-12-16']
CRASH_DAY = '2024-08-05'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADE_DIR = os.path.join(ROOT, 'data', 'l2_raw', 'binance', 'trades')
# Original read mid from data/l2 parquet, but those files only exist for early
# 2024. The raw book_snapshot_25 csv.gz exists for ALL target dates, so we read
# the best bid/ask mid from there instead (identical mid definition).
BOOK_DIR = os.path.join(ROOT, 'data', 'l2_raw', 'binance', 'book_snapshot_25')
INST_USD = 10_000.0
RETAIL_USD = 1_000.0
BPS = 1e4


def list_days(sym):
    """Fixed full-2024 date list; keep only dates with both trade & book files."""
    out = []
    for d in DATES:
        tp = os.path.join(TRADE_DIR, sym, d + '.csv.gz')
        bp = os.path.join(BOOK_DIR, sym, d + '.csv.gz')
        if os.path.exists(tp) and os.path.exists(bp):
            out.append(d)
    return out


def load_trades(sym, day):
    fp = os.path.join(TRADE_DIR, sym, day + '.csv.gz')
    df = pd.read_csv(fp, usecols=['timestamp', 'side', 'price', 'amount'])
    df['usd'] = df['price'] * df['amount']
    df['sign'] = np.where(df['side'] == 'buy', 1.0, -1.0)
    df['ts'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
    return df


def load_mid(sym, day):
    # Best bid/ask mid from the raw book_snapshot_25 csv.gz (timestamp in us).
    b = pd.read_csv(os.path.join(BOOK_DIR, sym, day + '.csv.gz'),
                    usecols=['timestamp', 'bids[0].price', 'asks[0].price'])
    b['mid'] = 0.5 * (b['bids[0].price'] + b['asks[0].price'])
    b['ts'] = pd.to_datetime(b['timestamp'], unit='us', utc=True)
    b = b[['ts', 'mid']].dropna().set_index('ts').sort_index()
    return b['mid']


def build_bars(sym, freq):
    out = []
    for day in list_days(sym):
        tr = load_trades(sym, day)
        mid = load_mid(sym, day)
        tr['cls'] = np.where(tr['usd'] >= INST_USD, 'inst',
                             np.where(tr['usd'] >= RETAIL_USD, 'mid', 'retail'))
        tr['sgn_usd'] = tr['sign'] * tr['usd']
        g = tr.set_index('ts')
        agg = pd.DataFrame({
            'tot_usd': g['usd'].resample(freq).sum(),
            'sgn_all': g['sgn_usd'].resample(freq).sum(),
            'sgn_inst': g.loc[g['cls'] == 'inst', 'sgn_usd'].resample(freq).sum(),
            'sgn_mid': g.loc[g['cls'] == 'mid', 'sgn_usd'].resample(freq).sum(),
            'sgn_retail': g.loc[g['cls'] == 'retail', 'sgn_usd'].resample(freq).sum(),
        })
        agg['mid_end'] = mid.resample(freq).last()
        agg['day'] = day
        out.append(agg)
    df = pd.concat(out).sort_index()
    for c in ['sgn_all', 'sgn_inst', 'sgn_mid', 'sgn_retail']:
        df[c + '_n'] = df[c].fillna(0.0) / df['tot_usd'].replace(0, np.nan)
    df['logmid'] = np.log(df['mid_end'])
    df['r'] = df.groupby('day')['logmid'].diff()      # return during bar t  = ret over [t-H,t]
    df['r_fwd'] = df.groupby('day')['r'].shift(-1)     # return over next bar = ret over [t,t+H]
    df['r_lag'] = df['r']                              # momentum control
    return df


def hac_reg(y, X, maxlags):
    return sm.OLS(y, sm.add_constant(X), missing='drop').fit(
        cov_type='HAC', cov_kwds={'maxlags': maxlags})


def run_for(df, label, maxlags, demean_by=None):
    cols = ['sgn_inst_n', 'sgn_mid_n', 'sgn_retail_n', 'sgn_all_n', 'r_lag', 'r_fwd']
    d = df[cols + ([demean_by] if demean_by else [])].dropna()
    if demean_by:  # symbol fixed effects: within-symbol demean of all regressors+target
        for c in cols:
            d[c] = d[c] - d.groupby(demean_by)[c].transform('mean')
    n = len(d)
    y = d['r_fwd'] * BPS
    m1 = hac_reg(y, d[['sgn_inst_n']], maxlags)
    m2 = hac_reg(y, d[['sgn_inst_n', 'sgn_all_n', 'r_lag']], maxlags)
    m2b = hac_reg(y, d[['sgn_all_n', 'r_lag']], maxlags)
    m3 = hac_reg(y, d[['sgn_inst_n', 'sgn_mid_n', 'sgn_retail_n']], maxlags)
    sd_inst = d['sgn_inst_n'].std()
    return dict(
        label=label, n=n,
        beta1=m1.params['sgn_inst_n'], t1=m1.tvalues['sgn_inst_n'], r2_1=m1.rsquared,
        beta2_inst=m2.params['sgn_inst_n'], t2_inst=m2.tvalues['sgn_inst_n'],
        beta2_all=m2.params['sgn_all_n'], t2_all=m2.tvalues['sgn_all_n'],
        beta2_mom=m2.params['r_lag'], t2_mom=m2.tvalues['r_lag'],
        r2_2=m2.rsquared, r2_2b=m2b.rsquared, incr_r2=m2.rsquared - m2b.rsquared,
        b_inst3=m3.params['sgn_inst_n'], t_inst3=m3.tvalues['sgn_inst_n'],
        b_mid3=m3.params['sgn_mid_n'], t_mid3=m3.tvalues['sgn_mid_n'],
        b_ret3=m3.params['sgn_retail_n'], t_ret3=m3.tvalues['sgn_retail_n'],
        sd_inst=sd_inst, bps_per_std=m1.params['sgn_inst_n'] * sd_inst,
    )


def pr(r):
    print(f"\n[{r['label']}]  n={r['n']}")
    print(f"  (1) inst-only:       beta={r['beta1']:+.2f} bps/share  t={r['t1']:+.2f}  R2={r['r2_1']*100:.3f}%")
    print(f"  (2) +total+mom:      inst b={r['beta2_inst']:+.2f} t={r['t2_inst']:+.2f} | total b={r['beta2_all']:+.2f} t={r['t2_all']:+.2f} | mom b={r['beta2_mom']:+.3f} t={r['t2_mom']:+.2f}")
    print(f"      R2full={r['r2_2']*100:.3f}% R2(no-inst)={r['r2_2b']*100:.3f}%  >> INCR-R2(inst)={r['incr_r2']*100:+.4f}%")
    print(f"  (3) inst-vs-noise:   inst b={r['b_inst3']:+.2f} t={r['t_inst3']:+.2f} | mid b={r['b_mid3']:+.2f} t={r['t_mid3']:+.2f} | retail b={r['b_ret3']:+.2f} t={r['t_ret3']:+.2f}")
    print(f"  (4) tradeability:    sd(inst share)={r['sd_inst']:.3f}  E[fwd ret|+1sd inst]={r['bps_per_std']:+.3f} bps  (cost ~10-20 bps round-trip)")


def main():
    horizons = {'15min': ('15min', 8), '1h': ('1h', 6)}
    bars = {}
    for sym in SYMS:
        for hname, (freq, _) in horizons.items():
            bars[(sym, hname)] = build_bars(sym, freq).assign(symbol=sym)

    print("=" * 92)
    print(f"INSTITUTIONAL FLOW -> FUTURE RETURN | {len(DATES)} days across 2024 | flow=signed-USD share of bar vol")
    print(f"Dates: {DATES}")
    print("=" * 92)
    for hname, (freq, mlag) in horizons.items():
        print(f"\n########## HORIZON = {hname} ##########")
        big = pd.concat([bars[(s, hname)] for s in SYMS])
        pr(run_for(big, f'POOLED (sym-FE) {hname}', mlag, demean_by='symbol'))
        for sym in SYMS:
            pr(run_for(bars[(sym, hname)], f'{sym} {hname}', mlag))

    # ------------------------------------------------------------------
    # CRASH-DAY (2024-08-05) ONLY: does inst flow predict CONTINUATION?
    # ------------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"CRASH-DAY REGIME ({CRASH_DAY}) ONLY  — inst flow -> fwd return")
    print("=" * 92)
    for hname, (freq, mlag) in horizons.items():
        big = pd.concat([bars[(s, hname)] for s in SYMS])
        crash = big[big['day'] == CRASH_DAY]
        if len(crash.dropna(subset=['r_fwd', 'sgn_inst_n'])) > 10:
            print(f"\n########## CRASH {hname} ##########")
            pr(run_for(crash, f'POOLED CRASH {hname}', mlag, demean_by='symbol'))

    # ------------------------------------------------------------------
    # PER-DAY sign sweep: look for ANY regime where inst beta(2) is + & signif
    # (would be CONTINUATION, contradicting the Jan-2024 reversal finding).
    # ------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("PER-DAY INST-FLOW SIGN SWEEP (model 2: inst | +total+mom controls)")
    print("  beta2_inst > 0 & |t| > 2  => CONTINUATION (new); < 0 => reversal (orig)")
    print("=" * 92)
    for hname, (freq, mlag) in horizons.items():
        print(f"\n-- horizon {hname} --")
        big = pd.concat([bars[(s, hname)] for s in SYMS])
        for d in DATES:
            sub = big[big['day'] == d]
            sub = sub[['sgn_inst_n', 'sgn_mid_n', 'sgn_retail_n', 'sgn_all_n',
                       'r_lag', 'r_fwd', 'symbol']].dropna()
            if len(sub) < 15:
                print(f"  {d}: n={len(sub)} (too few, skip)")
                continue
            try:
                r = run_for(sub.assign(day=d), f'{d} {hname}', mlag, demean_by='symbol')
                tag = 'CONTINUATION' if (r['beta2_inst'] > 0 and abs(r['t2_inst']) > 2) else \
                      ('reversal' if (r['beta2_inst'] < 0 and abs(r['t2_inst']) > 2) else 'flat')
                print(f"  {d}: n={r['n']:4d}  inst_b={r['beta2_inst']:+7.2f} t={r['t2_inst']:+6.2f}  incrR2={r['incr_r2']*100:+.4f}%  -> {tag}")
            except Exception as e:
                print(f"  {d}: ERROR {e}")

    print("\n" + "=" * 92)
    print("CLASS SHARE SANITY (volume & order-count share by class, ALL days)")
    for sym in SYMS:
        vols = {'inst': 0., 'mid': 0., 'retail': 0.}
        cnts = {'inst': 0, 'mid': 0, 'retail': 0}
        for day in list_days(sym):
            tr = load_trades(sym, day)
            cls = np.where(tr['usd'] >= INST_USD, 'inst',
                           np.where(tr['usd'] >= RETAIL_USD, 'mid', 'retail'))
            for c in vols:
                m = cls == c
                vols[c] += tr.loc[m, 'usd'].sum()
                cnts[c] += int(m.sum())
        tv, tc = sum(vols.values()), sum(cnts.values())
        print(f"  {sym}: vol% inst={vols['inst']/tv*100:4.1f} mid={vols['mid']/tv*100:4.1f} retail={vols['retail']/tv*100:4.1f} | "
              f"ordr% inst={cnts['inst']/tc*100:4.1f} mid={cnts['mid']/tc*100:4.1f} retail={cnts['retail']/tc*100:4.1f}")


if __name__ == '__main__':
    main()
