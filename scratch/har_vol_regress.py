"""
HAR-RV benchmark vs HAR + microstructure for next-hour vol.
- In-sample R2 (OLS), HAC SEs.
- Walk-forward OOS R2 (expanding window, refit weekly).
- Pooled (with symbol FE for HAR lags computed per-symbol) and per-symbol.
- Incremental OOS R2 and vol-targeting Sharpe proxy.
"""
import pandas as pd, numpy as np
import statsmodels.api as sm

panel = pd.read_parquet('scratch/har_panel.parquet')

def build_features(df):
    df = df.sort_index().copy()
    rv = df['rv']
    # HAR components: past hour (rv lag1), past day (mean of last 24), past week (mean last 168)
    df['har_h'] = rv.shift(1)
    df['har_d'] = rv.shift(1).rolling(24, min_periods=24).mean()
    df['har_w'] = rv.shift(1).rolling(168, min_periods=168).mean()
    df['y'] = rv  # predict current hour from lagged info (target = rv_{t}, regressors known at t-1)
    # microstructure features over PAST hour (shift 1 so strictly predictive)
    micro = ['qupd','spr_m','spr_s','imb_m','imb_s','tcnt','tvol','absflow','vpin','bpv','jump']
    for c in micro:
        v = df[c].shift(1)
        # log-transform heavy-tailed positives
        if c in ('tcnt','tvol','absflow','qupd','bpv','jump'):
            v = np.log1p(v.clip(lower=0))
        df['m_'+c] = v
    return df, ['m_'+c for c in micro]

parts=[]
microcols=None
for s, g in panel.groupby('sym'):
    gg, mc = build_features(g)
    microcols = mc
    parts.append(gg)
D = pd.concat(parts)
HAR=['har_h','har_d','har_w']
need = HAR+microcols+['y','sym']
D = D.dropna(subset=need).copy()
# standardize micro within full sample for coef readability (does not affect R2)
print('Usable rows after lags:', len(D))

def ols_r2(df, cols):
    X = sm.add_constant(df[cols]); y=df['y']
    m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':24})
    return m

def oos_r2(df, cols, refit_every=24*7, min_train=24*14):
    """Expanding-window walk-forward; refit weekly. Returns OOS R2 vs mean benchmark
    and OOS R2 vs HAR (computed by caller). Here returns preds + actuals."""
    df = df.sort_index()
    y = df['y'].values
    X = sm.add_constant(df[cols], has_constant='add').values
    n=len(df); preds=np.full(n, np.nan); beta=None
    for i in range(min_train, n):
        if (i-min_train) % refit_every == 0:
            Xt=X[:i]; yt=y[:i]
            beta = np.linalg.lstsq(Xt, yt, rcond=None)[0]
        preds[i]=X[i]@beta
    mask=~np.isnan(preds)
    return preds, mask

def r2_oos(actual, pred, bench):
    # OOS R2 relative to a benchmark prediction (Campbell-Thompson style)
    sse = np.sum((actual-pred)**2); ssb=np.sum((actual-bench)**2)
    return 1 - sse/ssb

print('\n================ POOLED (all symbols stacked, time-ordered) ================')
# pool: order by timestamp then symbol; walk-forward on the stacked timeline per-symbol concat is messy,
# so do pooled IS, and pooled OOS by concatenating per-symbol walk-forward preds.
Dp = D.sort_index()
m_har = ols_r2(Dp, HAR)
m_all = ols_r2(Dp, HAR+microcols)
print(f'IS  R2  HAR-RV          : {m_har.rsquared:.4f}')
print(f'IS  R2  HAR + micro      : {m_all.rsquared:.4f}')
print(f'IS  incremental R2       : {m_all.rsquared-m_har.rsquared:.4f}')
print(f'IS  adj-R2 HAR / HAR+micro: {m_har.rsquared_adj:.4f} / {m_all.rsquared_adj:.4f}')

# Per-symbol walk-forward, then pool predictions for an aggregate OOS R2
def per_sym_wf(cols):
    acts=[]; prds=[]; means=[]
    for s,g in D.groupby('sym'):
        g=g.sort_index()
        pr,mask=oos_r2(g, cols)
        a=g['y'].values
        # benchmark = expanding mean of y up to t (unconditional)
        em = g['y'].expanding().mean().shift(1).values
        acts.append(a[mask]); prds.append(pr[mask]); means.append(em[mask])
    return np.concatenate(acts), np.concatenate(prds), np.concatenate(means)

a_h,p_h,mn = per_sym_wf(HAR)
a_a,p_a,_  = per_sym_wf(HAR+microcols)
# align (same mask since same dropna) -- recompute on identical index
r2_har_vs_mean = r2_oos(a_h,p_h,mn)
r2_all_vs_mean = r2_oos(a_a,p_a,mn)
r2_all_vs_har  = r2_oos(a_a,p_a,p_h)   # incremental OOS: micro model vs HAR as benchmark
print(f'\nOOS R2 (vs uncond mean) HAR-RV   : {r2_har_vs_mean:.4f}')
print(f'OOS R2 (vs uncond mean) HAR+micro: {r2_all_vs_mean:.4f}')
print(f'OOS incremental R2 (HAR+micro vs HAR as benchmark): {r2_all_vs_har:.4f}')

print('\n--- HAR+micro coefficients (pooled IS, HAC maxlags=24) ---')
co=m_all.params; tv=m_all.tvalues
for name in HAR+microcols:
    print(f'  {name:10s} coef={co[name]:+.5f}  t={tv[name]:+.2f}')

print('\n================ PER-SYMBOL ================')
rows=[]
for s,g in D.groupby('sym'):
    g=g.sort_index()
    mh=ols_r2(g,HAR); ma=ols_r2(g,HAR+microcols)
    ph_,mask=oos_r2(g,HAR); pa_,_=oos_r2(g,HAR+microcols)
    a=g['y'].values; em=g['y'].expanding().mean().shift(1).values
    oh=r2_oos(a[mask],ph_[mask],em[mask])
    oa=r2_oos(a[mask],pa_[mask],em[mask])
    oinc=r2_oos(a[mask],pa_[mask],ph_[mask])
    rows.append((s,mh.rsquared,ma.rsquared,ma.rsquared-mh.rsquared,oh,oa,oinc))
print(f'{"sym":8s} {"IS_HAR":>7s} {"IS_all":>7s} {"IS_inc":>7s} {"OOS_HAR":>8s} {"OOS_all":>8s} {"OOS_inc":>8s}')
for r in rows:
    print(f'{r[0]:8s} {r[1]:7.4f} {r[2]:7.4f} {r[3]:7.4f} {r[4]:8.4f} {r[5]:8.4f} {r[6]:8.4f}')

# ---- Vol-targeting value: does better vol forecast reduce realized vol-of-vol of a scaled position? ----
# position size_t = target/forecast_t ; realized "risk" proxy = std of (size_t * rv_t).
print('\n================ VOL-TARGETING PROXY (pooled OOS preds) ================')
def voltgt(a,p):
    p=np.clip(p, np.percentile(p,1), None)
    size = np.median(p)/p              # scale so position larger when predicted vol low
    realized_risk = size*a            # realized vol experienced given the sizing
    return np.std(realized_risk)/np.mean(realized_risk)  # coef of variation of realized risk (vol-of-vol proxy)
cv_har = voltgt(a_h,p_h); cv_all=voltgt(a_a,p_a)
cv_flat = np.std(a_h)/np.mean(a_h)  # no sizing (flat) realized-risk CV
print(f'Realized-risk CV  flat(no model): {cv_flat:.4f}')
print(f'Realized-risk CV  HAR sizing    : {cv_har:.4f}')
print(f'Realized-risk CV  HAR+micro     : {cv_all:.4f}')
print(f'Reduction micro vs HAR          : {(cv_har-cv_all)/cv_har*100:+.2f}%')
