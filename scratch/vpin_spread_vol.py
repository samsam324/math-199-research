"""
VPIN as a regime/trade-filter for pair spreads.

Hypothesis: VPIN does NOT predict spread return DIRECTION at tradeable horizons,
but MAY predict next-hour spread VOLATILITY / decoupling. Critical control: does
VPIN add anything INCREMENTAL over the spread's own lagged realized vol?

Design:
  - VPIN per symbol via VOLUME CLOCK (equal-volume buckets), toxicity per bucket
    = |buy_vol - sell_vol| / bucket_vol, VPIN = rolling mean over N buckets.
    Aggregate to hourly (mean of VPIN samples whose buckets close in the hour).
  - Pair VPIN = max and mean of the two legs (report both).
  - Hourly spread = log(mid_a) - beta*log(mid_b).
  - Targets (next hour): realized spread vol (std of 1s spread log-returns),
    |spread change| over the hour, decoupling = top-decile |spread move|.
  - Core test: regress next-hour vol on (a) VPIN, (b) lagged realized vol,
    (c) both. Report HAC t-stats, R2, incremental R2 of VPIN.
"""
import warnings, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\jackw\Desktop\math-199-research'
# ~30 trading days
DATES = pd.date_range('2024-01-02', '2024-02-10', freq='D').strftime('%Y-%m-%d').tolist()

# 8 liquid pairs (top of selected_pairs.parquet, liquid majors)
PAIRS = [
    ('BTCUSDT', 'ETHUSDT', 0.976922),
    ('SOLUSDT', 'AVAXUSDT', 1.304799),
    ('ADAUSDT', 'DOTUSDT', 0.852796),
    ('XRPUSDT', 'XLMUSDT', 0.874818),
    ('ETHUSDT', 'LINKUSDT', 1.041377),
    ('LINKUSDT', 'DOTUSDT', 0.530780),
    ('AVAXUSDT', 'DOTUSDT', 0.956218),
    ('ETHUSDT', 'BNBUSDT', 0.747670),
]

BUCKETS_PER_DAY = 50    # equal-volume buckets per symbol-day (volume clock granularity)
VPIN_WIN = 5            # rolling buckets for VPIN average


def load_l2_mid(sym, date):
    p = os.path.join(ROOT, 'data', 'l2', sym, date + '.parquet')
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p, columns=['timestamp', 'bid_px_1', 'ask_px_1'])
    df = df.set_index('timestamp')
    df = df[~df.index.duplicated(keep='last')]
    return ((df['bid_px_1'] + df['ask_px_1']) / 2.0).rename('mid')


def load_trades(sym, date):
    p = os.path.join(ROOT, 'data', 'trades', sym, date + '.parquet')
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p, columns=['timestamp', 'buy_volume_base',
                                      'sell_volume_base', 'volume_base'])
    return df.set_index('timestamp')


def vpin_hourly(sym, date):
    """Volume-clock VPIN -> hourly series for one symbol-day."""
    tr = load_trades(sym, date)
    if tr is None or tr['volume_base'].sum() <= 0:
        return None
    tr = tr.sort_index()
    tot = tr['volume_base'].sum()
    bucket_size = tot / BUCKETS_PER_DAY
    if bucket_size <= 0:
        return None
    # cumulative volume -> bucket id (equal-volume buckets)
    cumv = tr['volume_base'].cumsum()
    bid = np.minimum((cumv / bucket_size).astype(int), BUCKETS_PER_DAY - 1)
    g = pd.DataFrame({
        'buy': tr['buy_volume_base'].values,
        'sell': tr['sell_volume_base'].values,
        'vol': tr['volume_base'].values,
        'ts': tr.index,
        'bid': bid.values,
    })
    agg = g.groupby('bid').agg(buy=('buy', 'sum'), sell=('sell', 'sum'),
                               vol=('vol', 'sum'),
                               ts_close=('ts', 'max')).reset_index()
    agg = agg[agg['vol'] > 0]
    if len(agg) < VPIN_WIN + 1:
        return None
    # toxicity per bucket
    agg['tox'] = (agg['buy'] - agg['sell']).abs() / agg['vol']
    # VPIN = rolling mean of toxicity over VPIN_WIN buckets
    agg['vpin'] = agg['tox'].rolling(VPIN_WIN, min_periods=VPIN_WIN).mean()
    agg = agg.dropna(subset=['vpin'])
    if agg.empty:
        return None
    # aggregate to hour using bucket close time
    agg['hour'] = pd.to_datetime(agg['ts_close']).dt.floor('h')
    hv = agg.groupby('hour')['vpin'].mean()
    return hv


def spread_hourly_targets(a, b, beta, date):
    """Hourly spread realized vol & |change| from 1s mids."""
    ma, mb = load_l2_mid(a, date), load_l2_mid(b, date)
    if ma is None or mb is None:
        return None
    grid = pd.date_range(date + ' 00:00:00+00:00', date + ' 23:59:59+00:00', freq='1s')
    ma = ma.reindex(grid).ffill(limit=5)
    mb = mb.reindex(grid).ffill(limit=5)
    spread = np.log(ma) - beta * np.log(mb)
    valid = ma.notna() & mb.notna()
    spread = spread.where(valid)
    ret = spread.diff()  # 1s spread log-return
    hour = spread.index.floor('h')
    df = pd.DataFrame({'spread': spread.values, 'ret': ret.values, 'hour': hour})
    def hour_stats(grp):
        s = grp['spread'].dropna()
        r = grp['ret'].dropna()
        if len(s) < 60 or len(r) < 60:
            return pd.Series({'rv': np.nan, 'absmove': np.nan, 'signmove': np.nan})
        rv = r.std() * 1e4  # bps realized vol of 1s returns
        move = (s.iloc[-1] - s.iloc[0]) * 1e4  # bps net move over hour
        return pd.Series({'rv': rv, 'absmove': abs(move), 'signmove': move})
    out = df.groupby('hour').apply(hour_stats)
    return out


def build_pair_panel(a, b, beta):
    rows = []
    for d in DATES:
        tgt = spread_hourly_targets(a, b, beta, d)
        if tgt is None or tgt.empty:
            continue
        va = vpin_hourly(a, d)
        vb = vpin_hourly(b, d)
        if va is None or vb is None:
            continue
        idx = tgt.index
        va = va.reindex(idx)
        vb = vb.reindex(idx)
        panel = tgt.copy()
        panel['vpin_max'] = np.maximum(va.values, vb.values)
        panel['vpin_mean'] = (va.values + vb.values) / 2.0
        rows.append(panel)
    if not rows:
        return None
    P = pd.concat(rows).sort_index()
    P = P.dropna(subset=['rv', 'absmove', 'vpin_max', 'vpin_mean'])
    # current-hour realized vol = lagged predictor for next hour
    P['rv_lag'] = P['rv']
    # next-hour targets
    P['rv_next'] = P['rv'].shift(-1)
    P['absmove_next'] = P['absmove'].shift(-1)
    # only keep consecutive hours (next hour exactly +1h)
    nxt_time = P.index.to_series().shift(-1)
    consec = (nxt_time - P.index.to_series()) == pd.Timedelta(hours=1)
    P = P[consec.values]
    P = P.dropna(subset=['rv_next', 'absmove_next', 'rv_lag', 'vpin_max'])
    return P


def z(s):
    return (s - s.mean()) / s.std()


def hac_ols(y, Xdf, maxlags=4):
    X = sm.add_constant(Xdf)
    m = sm.OLS(y.values, X.values, missing='drop').fit(
        cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return m


def fmt(m, names):
    parts = []
    for i, nm in enumerate(['const'] + names):
        parts.append(f"{nm}={m.params[i]:+.4f}(t={m.tvalues[i]:+.2f})")
    return " ".join(parts) + f"  R2={m.rsquared:.4f} N={int(m.nobs)}"


print(f"VPIN spread-vol filter test | window {DATES[0]}..{DATES[-1]} "
      f"({len(DATES)} days) | {BUCKETS_PER_DAY} vol-buckets/day, VPIN win={VPIN_WIN}")
print("Target: next-hour realized spread vol (bps, std of 1s spread log-rets). "
      "VPIN_pair = max(legs). Standardized regressors.\n")

agg_incr = []
for a, b, beta in PAIRS:
    P = build_pair_panel(a, b, beta)
    if P is None or len(P) < 50:
        print(f"=== {a}/{b}: SKIP (N too small) ===\n")
        continue
    y = P['rv_next']
    vpin = z(P['vpin_max'])
    rvlag = z(P['rv_lag'])

    # (a) VPIN alone
    ma = hac_ols(y, pd.DataFrame({'vpin': vpin}))
    # (b) lagged realized vol alone
    mb = hac_ols(y, pd.DataFrame({'rv_lag': rvlag}))
    # (c) both
    mc = hac_ols(y, pd.DataFrame({'vpin': vpin, 'rv_lag': rvlag}))

    incr_r2 = mc.rsquared - mb.rsquared  # incremental R2 of VPIN over rv_lag
    # vpin t-stat in joint model
    vpin_t_joint = mc.tvalues[1]

    # correlation between vpin and lagged rv (redundancy check)
    rho = np.corrcoef(P['vpin_max'], P['rv_lag'])[0, 1]

    # decoupling logit: next-hour absmove in top decile ~ vpin + rv_lag
    thr = P['absmove_next'].quantile(0.90)
    P['decouple'] = (P['absmove_next'] >= thr).astype(int)
    try:
        Xl = sm.add_constant(pd.DataFrame({'vpin': vpin.values, 'rv_lag': rvlag.values}))
        logit = sm.Logit(P['decouple'].values, Xl.values).fit(disp=0)
        logit_line = (f"vpin={logit.params[1]:+.3f}(z={logit.tvalues[1]:+.2f}) "
                      f"rv_lag={logit.params[2]:+.3f}(z={logit.tvalues[2]:+.2f}) "
                      f"pseudoR2={logit.prsquared:.4f}")
        # vpin-only logit pseudo R2 for incremental
        Xl1 = sm.add_constant(rvlag.values)
        lg1 = sm.Logit(P['decouple'].values, Xl1).fit(disp=0)
        logit_incr = logit.prsquared - lg1.prsquared
    except Exception as e:
        logit_line = f"FAILED({e})"
        logit_incr = np.nan

    print(f"=== {a}/{b}  beta={beta:.3f}  N={len(P)} hrs ===")
    print(f"  corr(VPIN, lagged_RV) = {rho:+.3f}")
    print(f"  (a) VPIN only:    {fmt(ma, ['vpin'])}")
    print(f"  (b) rv_lag only:  {fmt(mb, ['rv_lag'])}")
    print(f"  (c) BOTH:         {fmt(mc, ['vpin', 'rv_lag'])}")
    print(f"  -> incremental R2 of VPIN over rv_lag = {incr_r2:+.4f} "
          f"(VPIN joint t={vpin_t_joint:+.2f})")
    print(f"  decouple logit:   {logit_line}")
    print(f"  -> incremental pseudo-R2 of VPIN = {logit_incr:+.4f}")

    # practical: high vs low VPIN hours -> next-hour vol & |move|
    med = P['vpin_max'].median()
    hi = P[P['vpin_max'] > med]
    lo = P[P['vpin_max'] <= med]
    print(f"  high-VPIN next-hr: RV={hi['rv_next'].mean():.3f}bps "
          f"|move|={hi['absmove_next'].mean():.3f}bps  decouple-rate={hi['decouple'].mean():.3f}")
    print(f"  low -VPIN next-hr: RV={lo['rv_next'].mean():.3f}bps "
          f"|move|={lo['absmove_next'].mean():.3f}bps  decouple-rate={lo['decouple'].mean():.3f}")
    print()

    agg_incr.append((f"{a}/{b}", rho, mb.rsquared, mc.rsquared, incr_r2,
                     vpin_t_joint, logit_incr,
                     hi['rv_next'].mean(), lo['rv_next'].mean(),
                     hi['decouple'].mean(), lo['decouple'].mean()))

# pooled summary
if agg_incr:
    import statistics as st
    print("=" * 70)
    print("POOLED SUMMARY across pairs")
    print(f"{'pair':>18} {'corr':>7} {'R2_b':>7} {'R2_c':>7} {'dR2':>8} {'vpin_t':>8} {'dPseudo':>8}")
    for r in agg_incr:
        print(f"{r[0]:>18} {r[1]:+7.3f} {r[2]:7.4f} {r[3]:7.4f} {r[4]:+8.4f} {r[5]:+8.2f} {r[6]:+8.4f}")
    dr2 = [r[4] for r in agg_incr]
    vts = [r[5] for r in agg_incr]
    rhos = [r[1] for r in agg_incr]
    print("-" * 70)
    print(f"  mean corr(VPIN,rvlag) = {st.mean(rhos):+.3f}")
    print(f"  mean incremental R2 (VPIN over rv_lag) = {st.mean(dr2):+.4f}  "
          f"(min {min(dr2):+.4f}, max {max(dr2):+.4f})")
    print(f"  mean VPIN joint t-stat = {st.mean(vts):+.2f}  "
          f"| #pairs |t|>1.96 = {sum(1 for t in vts if abs(t) > 1.96)}/{len(vts)}")
    hi_rv = st.mean([r[7] for r in agg_incr]); lo_rv = st.mean([r[8] for r in agg_incr])
    hi_dc = st.mean([r[9] for r in agg_incr]); lo_dc = st.mean([r[10] for r in agg_incr])
    print(f"  high-VPIN next-hr RV {hi_rv:.3f} vs low {lo_rv:.3f} bps "
          f"(ratio {hi_rv/lo_rv:.2f}x)")
    print(f"  high-VPIN decouple-rate {hi_dc:.3f} vs low {lo_dc:.3f} "
          f"(lift {hi_dc/lo_dc:.2f}x)")
