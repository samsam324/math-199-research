"""
Institutional (large-trade) net order flow -> tradeable-horizon (15min, 1h) return prediction.

Builds signed class flow (institutional>$10k, mid $1k-$10k, retail<$1k) per H-bar,
normalized by total bar volume. Core test: future return ~ inst flow, then + total flow
+ lagged return; report incremental R2. Informed-vs-noise: inst/mid/retail together.

Pooled + per-symbol, HAC SEs. Run: python scratch/inst_flow_horizon.py
"""
import os, sys, gzip
import numpy as np
import pandas as pd
import statsmodels.api as sm

SYMS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
NDAYS = 30
START = '2024-01-02'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADE_DIR = os.path.join(ROOT, 'data', 'l2_raw', 'binance', 'trades')
BOOK_DIR = os.path.join(ROOT, 'data', 'l2')

INST_USD = 10_000.0
RETAIL_USD = 1_000.0


def list_days(sym):
    bd = set(f[:-8] for f in os.listdir(os.path.join(BOOK_DIR, sym)) if f.endswith('.parquet'))
    td = set(f[:-7] for f in os.listdir(os.path.join(TRADE_DIR, sym)) if f.endswith('.csv.gz'))
    days = sorted(d for d in (bd & td) if d >= START)
    return days[:NDAYS]


def load_trades(sym, day):
    fp = os.path.join(TRADE_DIR, sym, day + '.csv.gz')
    df = pd.read_csv(fp, usecols=['timestamp', 'side', 'price', 'amount'])
    df['usd'] = df['price'] * df['amount']
    df['sign'] = np.where(df['side'] == 'buy', 1.0, -1.0)
    df['ts'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
    return df


def load_mid(sym, day):
    fp = os.path.join(BOOK_DIR, sym, day + '.parquet')
    b = pd.read_parquet(fp, columns=['timestamp', 'bid_px_1', 'ask_px_1'])
    b['mid'] = 0.5 * (b['bid_px_1'] + b['ask_px_1'])
    b = b[['timestamp', 'mid']].dropna()
    b = b.set_index('timestamp').sort_index()
    return b


def build_bars(sym, freq):
    """Return DataFrame indexed by bar-start with signed class flow shares and bar-end logmid."""
    days = list_days(sym)
    out = []
    for day in days:
        tr = load_trades(sym, day)
        mid = load_mid(sym, day)
        # signed usd per class
        tr['cls'] = np.where(tr['usd'] >= INST_USD, 'inst',
                      np.where(tr['usd'] >= RETAIL_USD, 'mid', 'retail'))
        tr['sgn_usd'] = tr['sign'] * tr['usd']
        g = tr.set_index('ts')
        # aggregate signed usd per class and total usd per bar
        agg = pd.DataFrame({
            'tot_usd': g['usd'].resample(freq).sum(),
            'sgn_all': g['sgn_usd'].resample(freq).sum(),
            'sgn_inst': g.loc[g['cls'] == 'inst', 'sgn_usd'].resample(freq).sum(),
            'sgn_mid': g.loc[g['cls'] == 'mid', 'sgn_usd'].resample(freq).sum(),
            'sgn_retail': g.loc[g['cls'] == 'retail', 'sgn_usd'].resample(freq).sum(),
        })
        # bar-end mid (last book obs at/just before bar boundary). Reindex mid to bar grid.
        # Use last mid within each bar as the price stamp at the bar's *end*.
        midbar = mid['mid'].resample(freq).last()
        agg['mid_end'] = midbar
        agg.index.name = 'bar'
        agg['day'] = day
        out.append(agg)
    df = pd.concat(out)
    df = df.sort_index()
    # normalize signed flow by total bar volume (share in [-1,1])
    for c in ['sgn_all', 'sgn_inst', 'sgn_mid', 'sgn_retail']:
        df[c + '_n'] = df[c].fillna(0.0) / df['tot_usd'].replace(0, np.nan)
    df['logmid'] = np.log(df['mid_end'])
    return df


def add_targets(df):
    """future ret over [t, t+H] uses next-bar's flow window convention:
    We regress ret_fwd (this bar's return) on PRIOR bar's flow. Equivalent to
    'flow over [t-H,t] predicts return over [t,t+H]'. Build forward return as
    logmid(next) - logmid(now)... but careful with day boundaries.
    """
    # ret over the bar itself = logmid_end(t) - logmid_end(t-1). Forward ret over [t,t+H]
    # = logmid_end(t) - logmid_end(t-1) shifted: define bar return r_t, then target=r_{t+1}
    df = df.copy()
    df['r'] = df.groupby('day')['logmid'].diff()  # return realized DURING bar t
    df['r_fwd'] = df.groupby('day')['r'].shift(-1)  # return over next bar [t,t+H]
    df['r_lag'] = df['r']  # momentum: return over [t-H,t] (same as current bar return)
    return df


def hac_reg(y, X, maxlags):
    Xc = sm.add_constant(X)
    m = sm.OLS(y, Xc, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return m


def run_for(df, label, maxlags):
    """df has columns sgn_inst_n, sgn_all_n, sgn_mid_n, sgn_retail_n, r_lag, r_fwd."""
    d = df[['sgn_inst_n', 'sgn_mid_n', 'sgn_retail_n', 'sgn_all_n', 'r_lag', 'r_fwd']].dropna()
    n = len(d)
    # scale returns to bps for readability
    BPS = 1e4
    y = d['r_fwd'] * BPS
    res = {}

    # (1) inst only
    m1 = hac_reg(y, d[['sgn_inst_n']], maxlags)
    # (2) inst + total + momentum
    m2 = hac_reg(y, d[['sgn_inst_n', 'sgn_all_n', 'r_lag']], maxlags)
    # (2b) total + momentum WITHOUT inst (for incremental R2)
    m2b = hac_reg(y, d[['sgn_all_n', 'r_lag']], maxlags)
    # (3) informed-vs-noise: inst + mid + retail
    m3 = hac_reg(y, d[['sgn_inst_n', 'sgn_mid_n', 'sgn_retail_n']], maxlags)

    incr_r2 = m2.rsquared - m2b.rsquared

    # std of inst flow share, for tradeability: expected fwd ret per +1 std inst flow
    sd_inst = d['sgn_inst_n'].std()
    beta_inst_uni = m1.params['sgn_inst_n']  # bps per unit share
    bps_per_std = beta_inst_uni * sd_inst

    return dict(
        label=label, n=n,
        beta1=m1.params['sgn_inst_n'], t1=m1.tvalues['sgn_inst_n'], r2_1=m1.rsquared,
        beta2_inst=m2.params['sgn_inst_n'], t2_inst=m2.tvalues['sgn_inst_n'],
        beta2_all=m2.params['sgn_all_n'], t2_all=m2.tvalues['sgn_all_n'],
        beta2_mom=m2.params['r_lag'], t2_mom=m2.tvalues['r_lag'],
        r2_2=m2.rsquared, r2_2b=m2b.rsquared, incr_r2=incr_r2,
        b_inst3=m3.params['sgn_inst_n'], t_inst3=m3.tvalues['sgn_inst_n'],
        b_mid3=m3.params['sgn_mid_n'], t_mid3=m3.tvalues['sgn_mid_n'],
        b_ret3=m3.params['sgn_retail_n'], t_ret3=m3.tvalues['sgn_retail_n'],
        sd_inst=sd_inst, bps_per_std=bps_per_std,
    )


def main():
    horizons = {'15min': ('15min', 8), '1h': ('1h', 6)}  # (pandas freq, HAC maxlags)
    all_results = {}
    pooled = {h: [] for h in horizons}
    # also keep class-share diagnostics
    diag = []
    for sym in SYMS:
        for hname, (freq, mlag) in horizons.items():
            df = build_bars(sym, freq)
            # diagnostics: share of volume & order count by class (first horizon only)
            df = add_targets(df)
            df['sym'] = sym
            r = run_for(df, f'{sym} {hname}', mlag)
            all_results[(sym, hname)] = r
            pooled[hname].append(df.assign(symbol=sym))
    # pooled regressions (stack symbols; demean? keep raw, shares already comparable)
    pooled_res = {}
    for hname, (freq, mlag) in horizons.items():
        big = pd.concat(pooled[hname])
        pooled_res[hname] = run_for(big, f'POOLED {hname}', mlag)

    # ---- print ----
    def pr(r):
        print(f"\n[{r['label']}]  n={r['n']}")
        print(f"  (1) inst-only:        beta={r['beta1']:+.2f} bps/share  t={r['t1']:+.2f}  R2={r['r2_1']*100:.3f}%")
        print(f"  (2) +total+momentum:  inst beta={r['beta2_inst']:+.2f} t={r['t2_inst']:+.2f} | total beta={r['beta2_all']:+.2f} t={r['t2_all']:+.2f} | mom beta={r['beta2_mom']:+.3f} t={r['t2_mom']:+.2f}")
        print(f"      R2(full)={r['r2_2']*100:.3f}%  R2(no-inst)={r['r2_2b']*100:.3f}%  INCREMENTAL R2(inst)={r['incr_r2']*100:+.4f}%")
        print(f"  (3) informed-vs-noise: inst b={r['b_inst3']:+.2f} t={r['t_inst3']:+.2f} | mid b={r['b_mid3']:+.2f} t={r['t_mid3']:+.2f} | retail b={r['b_ret3']:+.2f} t={r['t_ret3']:+.2f}")
        print(f"  (4) tradeability: sd(inst share)={r['sd_inst']:.3f}  E[fwd ret | +1sd inst]={r['bps_per_std']:+.3f} bps  (round-trip cost ~10-20 bps)")

    print("="*90)
    print("INSTITUTIONAL FLOW -> FUTURE RETURN  (BPS units; flow = signed-USD share of bar volume)")
    print("="*90)
    for hname in horizons:
        print(f"\n########## HORIZON = {hname} ##########")
        pr(pooled_res[hname])
        for sym in SYMS:
            pr(all_results[(sym, hname)])

    # class-share sanity (use 1h BTC bars already built? rebuild quick from pooled)
    print("\n" + "="*90)
    print("CLASS SHARE SANITY (volume & order-count share by class, pooled 1h)")
    for sym in SYMS:
        tr_days = list_days(sym)[:3]
        vols = {'inst':0.,'mid':0.,'retail':0.}; cnts={'inst':0,'mid':0,'retail':0}
        for day in tr_days:
            tr = load_trades(sym, day)
            cls = np.where(tr['usd']>=INST_USD,'inst',np.where(tr['usd']>=RETAIL_USD,'mid','retail'))
            for c in ['inst','mid','retail']:
                msk = cls==c
                vols[c]+=tr.loc[msk,'usd'].sum(); cnts[c]+=msk.sum()
        tv=sum(vols.values()); tc=sum(cnts.values())
        print(f"  {sym}: vol% inst={vols['inst']/tv*100:.1f} mid={vols['mid']/tv*100:.1f} retail={vols['retail']/tv*100:.1f} | "
              f"ordr% inst={cnts['inst']/tc*100:.1f} mid={cnts['mid']/tc*100:.1f} retail={cnts['retail']/tc*100:.1f}")


if __name__ == '__main__':
    main()
