"""Cross-asset lead-lag in crypto: BTC -> alts, with staleness control.
Days 2024-01-02..2024-01-11. Brutally honest verdict on tradeability."""
import numpy as np, pandas as pd, warnings, sys
from pathlib import Path
warnings.filterwarnings("ignore")
import statsmodels.api as sm

ROOT = Path(r"C:\Users\jackw\Desktop\math-199-research")
L2 = ROOT/"data"/"l2"; TR = ROOT/"data"/"trades"
DATES = [f"2024-01-{d:02d}" for d in range(2,12)]
LEADER = "BTCUSDT"
ALTS = ["ETHUSDT","SOLUSDT","ADAUSDT","LINKUSDT","AVAXUSDT"]
ALL = [LEADER]+ALTS

def load_mid(sym):
    frames=[]
    for d in DATES:
        f=L2/sym/f"{d}.parquet"
        if not f.exists(): continue
        df=pd.read_parquet(f, columns=["timestamp","bid_px_1","ask_px_1"])
        df["mid"]=(df["bid_px_1"]+df["ask_px_1"])/2.0
        df=df[["timestamp","mid"]].dropna()
        # resample to strict 1s grid (last obs in each second), ffill within day
        df=df.set_index("timestamp").sort_index()
        df=df[~df.index.duplicated(keep="last")]
        idx=pd.date_range(df.index[0].floor("s"), df.index[-1].ceil("s"), freq="1s", tz="UTC")
        s=df["mid"].reindex(idx, method="ffill")
        s.name=d
        frames.append(s)
    return pd.concat(frames)

def load_flow(sym):
    frames=[]
    for d in DATES:
        f=TR/sym/f"{d}.parquet"
        if not f.exists(): continue
        df=pd.read_parquet(f, columns=["timestamp","signed_volume_base","volume_base"])
        df=df.set_index("timestamp").sort_index()
        df=df[~df.index.duplicated(keep="last")]
        idx=pd.date_range(df.index[0].floor("s"), df.index[-1].ceil("s"), freq="1s", tz="UTC")
        sv=df["signed_volume_base"].reindex(idx, fill_value=0.0)
        vb=df["volume_base"].reindex(idx, fill_value=0.0)
        frames.append(pd.DataFrame({"sv":sv,"vb":vb}))
    return pd.concat(frames)

print("Loading mids...", flush=True)
mids={s:load_mid(s) for s in ALL}
print("Loading flows...", flush=True)
flows={s:load_flow(s) for s in ALL}

# ---------- STALENESS: quote update frequency on 1s grid ----------
print("\n=== (3a) QUOTE UPDATE FREQUENCY (frac of 1s bars with mid change) ===")
upd_freq={}
for s in ALL:
    m=mids[s]
    chg=(m.diff()!=0)&m.diff().notna()
    upd_freq[s]=chg.mean()
    print(f"  {s:9s}: {upd_freq[s]*100:6.2f}% of 1s bars have a mid change")

def aligned_returns(symlist, step):
    """Return DataFrame of log-returns sampled every `step` seconds on a shared
    wall-clock grid. Returns spanning a day-boundary gap (>step s) are dropped so
    cross-day jumps don't contaminate. +lag => first sym leads."""
    cols={}
    for s in symlist:
        m=mids[s]  # 1s grid, ffilled within each day, concatenated across days
        # sample on a wall-clock grid aligned to step-second boundaries
        ms=m[(m.index.astype("int64")//10**9) % step == 0]
        lr=np.log(ms).diff()
        # null out returns that span a gap larger than `step` (day boundaries)
        dt=ms.index.to_series().diff().dt.total_seconds()
        lr[dt>step]=np.nan
        cols[s]=lr
    df=pd.DataFrame(cols).dropna()
    return df

def xcorr(x, y, maxlag):
    """corr(x_t, y_{t+lag}) for lag in -maxlag..maxlag. Positive lag => x leads y."""
    out={}
    for lag in range(-maxlag, maxlag+1):
        if lag>=0:
            a=x.iloc[:len(x)-lag].values; b=y.iloc[lag:].values
        else:
            a=x.iloc[-lag:].values; b=y.iloc[:len(y)+lag].values
        if len(a)<30: out[lag]=np.nan; continue
        out[lag]=np.corrcoef(a,b)[0,1]
    return pd.Series(out)

# ---------- (2) LEAD-LAG cross-correlation, 1s sampling, lags -60..60 ----------
print("\n=== (2) CROSS-CORRELATION  corr(BTC_ret(t), ALT_ret(t+lag)) ; +lag => BTC leads ===")
print("    [1s sampling, lags -60..+60s]")
r1=aligned_returns(ALL, 1)
peaks_1s={}
for a in ALTS:
    cc=xcorr(r1[LEADER], r1[a], 60)
    pk=int(cc.idxmax()); contemp=cc.get(0,np.nan)
    peaks_1s[a]=(pk,cc[pk],contemp)
    lead_neg = cc.loc[1:20].max()  # BTC leads
    lag_neg  = cc.loc[-20:-1].max() # ALT leads
    print(f"  {a:9s}: peak lag={pk:+3d}s r={cc[pk]:.3f} | r(0)={contemp:.3f} | max BTC-leads(1..20s)={lead_neg:.3f} max ALT-leads={lag_neg:.3f}")

# ---------- (3b) COARSE SAMPLING SURVIVAL TEST: 30s & 60s ----------
print("\n=== (3b) COARSE-SAMPLING SURVIVAL TEST (both assets ~always update) ===")
for step,maxlag in [(30,4),(60,3)]:
    rr=aligned_returns(ALL, step)
    print(f"  --- {step}s sampling (lags in units of {step}s) ---")
    for a in ALTS:
        cc=xcorr(rr[LEADER], rr[a], maxlag)
        pk=int(cc.idxmax())
        contemp=cc.get(0,np.nan)
        btc_leads=cc.loc[1:maxlag].max() if maxlag>=1 else np.nan
        alt_leads=cc.loc[-maxlag:-1].max() if maxlag>=1 else np.nan
        verdict = "BTC-LEADS survives" if (btc_leads>0.04 and btc_leads>alt_leads and pk>0) else "no lead (contemp/artifact)"
        print(f"    {a:9s}: r(0)={contemp:.3f} peak={pk*step:+4d}s r={cc[pk]:.3f} | BTCleads={btc_leads:.3f} ALTleads={alt_leads:.3f} -> {verdict}")

# ---------- (4) PREDICTIVE REGRESSION with HAC SEs ----------
print("\n=== (4) PREDICTIVE: alt_fut_ret[t,t+h] ~ BTC_ret[t-1s..t] + BTC_signed_flow[t-w,t] ===")
print("    HAC(Newey-West) SEs. w=lookback for flow. Cost ref ~10bps taker.")
# build 1s log-return for BTC and alts on common index, plus BTC flow series
base_idx=None
ret1={}
for s in ALL:
    ret1[s]=np.log(mids[s]).diff()
btc_flow=flows[LEADER]["sv"]  # signed vol base per 1s
# normalize flow to z-score for interpretable units
def hac_reg(y,X,L):
    X=sm.add_constant(X)
    m=sm.OLS(y,X,missing="drop").fit(cov_type="HAC",cov_kwds={"maxlags":L})
    return m

HS=[5,30,60,300]
W=30  # flow lookback seconds
for a in ALTS:
    print(f"  --- {a} ---")
    ra=ret1[a]; rb=ret1[LEADER]
    # predictors at time t: BTC return over last 1s, BTC cumulative signed flow over [t-W,t]
    btc_ret_1s=rb
    btc_flow_w=btc_flow.rolling(W).sum()
    df=pd.DataFrame({"btc_ret":btc_ret_1s,"btc_flow":btc_flow_w}).dropna()
    # align alt mid for forward returns
    am=mids[a].reindex(df.index)
    for h in HS:
        fwd=np.log(mids[a].shift(-h)/mids[a]).reindex(df.index)
        d2=pd.concat([fwd.rename("y"),df],axis=1).dropna()
        if len(d2)<500:
            print(f"    h={h:4d}s: insufficient"); continue
        # standardize predictors
        Xz=(d2[["btc_ret","btc_flow"]]-d2[["btc_ret","btc_flow"]].mean())/d2[["btc_ret","btc_flow"]].std()
        m=hac_reg(d2["y"].values, Xz.values, L=h)
        r2=m.rsquared
        b_ret,b_flow=m.params[1],m.params[2]
        t_ret,t_flow=m.tvalues[1],m.tvalues[2]
        # implied bps move per +1 std of each signal
        print(f"    h={h:4d}s: R2={r2:.4f} | per+1sd BTCret={b_ret*1e4:+6.2f}bps(t={t_ret:+.1f}) "
              f"BTCflow={b_flow*1e4:+6.2f}bps(t={t_flow:+.1f})")

# ---------- (5) SPREAD: does BTC predict pair-spread returns? ----------
print("\n=== (5) SPREAD lead: spread_ret ~ BTC_ret + BTC_flow,  pairs at h=30,60,300s ===")
PAIRS=[("SOLUSDT","ETHUSDT"),("AVAXUSDT","ETHUSDT")]
for x,y in PAIRS:
    print(f"  --- spread {x} - {y} (log-mid diff) ---")
    lm_x=np.log(mids[x]); lm_y=np.log(mids[y])
    spread=(lm_x-lm_y)
    btc_flow_w=btc_flow.rolling(W).sum()
    base=pd.DataFrame({"btc_ret":ret1[LEADER],"btc_flow":btc_flow_w}).dropna()
    for h in [30,60,300]:
        fwd=(spread.shift(-h)-spread).reindex(base.index)
        d2=pd.concat([fwd.rename("y"),base],axis=1).dropna()
        if len(d2)<500: print(f"    h={h}s: insufficient"); continue
        Xz=(d2[["btc_ret","btc_flow"]]-d2[["btc_ret","btc_flow"]].mean())/d2[["btc_ret","btc_flow"]].std()
        m=hac_reg(d2["y"].values,Xz.values,L=h)
        print(f"    h={h:4d}s: R2={m.rsquared:.4f} | per+1sd BTCret={m.params[1]*1e4:+6.2f}bps(t={m.tvalues[1]:+.1f}) "
              f"BTCflow={m.params[2]*1e4:+6.2f}bps(t={m.tvalues[2]:+.1f})")

print("\nDONE.")
