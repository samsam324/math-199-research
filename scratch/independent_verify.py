"""
CLEAN-ROOM independent re-derivation ("make absolutely sure, from zero").
Imports NOTHING from wf_backtest / wf_robustness / simulate_pair_v. Re-implements
the whole pipeline from the raw parquets to check the central claims aren't an
artifact of my own code:
  - Binance no-stop monthly Sharpe ~2-2.5 ?
  - Coinbase no-stop monthly Sharpe ~1 (positive but ~half) ?
  - stop |z|>=4 loses on both ?
  - ratio Coinbase/Binance ~0.4-0.5 (the "~50-60% weaker") ?
Independent choices: own OLS hedge, own AR(1) kappa selection, own walk-forward,
own z-score trading loop, own cost model (30 bps round-trip). Qualitative + rough
magnitude agreement is the test, not exact decimals.

Run:  python scratch/independent_verify.py
"""
import os, glob
import numpy as np
import pandas as pd

ROOT = r"C:\Users\jackw\Desktop\math-199-research"
TR, TE = 4380, 2190          # 6-month train, 3-month test (hours), roll by test
ENTER, EXIT = 2.0, 0.5
COST_RT = 0.0015             # 15 bps per |Δpos| unit -> 30 bps round trip per pair


def load(venue_dir, symbols):
    closes = {}
    for s in symbols:
        f = os.path.join(ROOT, "data", venue_dir, f"{s}.parquet")
        if not os.path.exists(f):
            continue
        c = pd.read_parquet(f, columns=["close"])["close"]
        c.index = pd.to_datetime(c.index, utc=True)
        closes[s] = c[~c.index.duplicated(keep="last")]
    px = pd.DataFrame(closes).sort_index()
    full = pd.date_range(px.index.min().floor("h"), px.index.max().ceil("h"), freq="h", tz="UTC")
    px = px.reindex(full)
    px = px[px.index >= pd.Timestamp("2021-01-01", tz="UTC")].ffill(limit=3)
    return px


def overlap():
    cb = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(ROOT, "data", "coinbase_1h", "*.parquet"))}
    bn = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(ROOT, "data", "spot_1h", "*.parquet"))}
    return sorted(cb & bn)


def ar1_kappa(x):
    x = x[np.isfinite(x)]
    if len(x) < 200:
        return -1.0
    x0, x1 = x[:-1], x[1:]
    x0c, x1c = x0 - x0.mean(), x1 - x1.mean()
    denom = np.dot(x0c, x0c)
    if denom <= 0:
        return -1.0
    phi = np.dot(x0c, x1c) / denom
    if phi <= 0 or phi >= 1:
        return -1.0
    return -np.log(phi)


def backtest(px, npairs, stop=None):
    logpx = np.log(px); cols = list(px.columns); idx = px.index
    n = len(idx)
    port_parts = []
    s0 = 0
    while s0 + TR + TE <= n:
        tr = logpx.iloc[s0:s0 + TR]; te = logpx.iloc[s0 + TR:s0 + TR + TE]
        avail = [c for c in cols if tr[c].notna().mean() >= 0.90]
        cand = []
        for i in range(len(avail)):
            ai = tr[avail[i]].values
            for j in range(i + 1, len(avail)):
                bj = tr[avail[j]].values
                m = np.isfinite(ai) & np.isfinite(bj)
                if m.sum() < TR * 0.9:
                    continue
                beta, alpha = np.polyfit(bj[m], ai[m], 1)
                spr = ai - (alpha + beta * bj)
                k = ar1_kappa(spr[m])
                if k > 0:
                    cand.append((k, avail[i], avail[j], beta, alpha,
                                 np.nanmean(spr[m]), np.nanstd(spr[m])))
        cand.sort(key=lambda r: r[0], reverse=True)
        legs = []
        for (k, a, b, beta, alpha, mu, sd) in cand[:npairs]:
            if sd <= 0 or not np.isfinite(sd):
                continue
            la = te[a].values; lb = te[b].values
            spr = la - (alpha + beta * lb); z = (spr - mu) / sd
            pos = np.zeros(len(z)); cur = 0.0
            for t in range(len(z)):
                if not np.isfinite(z[t]):
                    cur = 0.0
                elif cur == 0.0:
                    if z[t] >= ENTER:
                        cur = -1.0
                    elif z[t] <= -ENTER:
                        cur = 1.0
                else:
                    if abs(z[t]) <= EXIT:
                        cur = 0.0
                    elif stop is not None and abs(z[t]) >= stop:
                        cur = 0.0
                pos[t] = cur
            dspr = np.diff(spr, prepend=spr[0]); dspr = np.where(np.isfinite(dspr), dspr, 0.0)
            poslag = np.roll(pos, 1); poslag[0] = 0.0
            gross = poslag * dspr
            cost = np.abs(np.diff(pos, prepend=0.0)) * COST_RT
            legs.append(pd.Series(gross - cost, index=te.index))
        if legs:
            port_parts.append(pd.concat(legs, axis=1).mean(axis=1))
        s0 += TE
    port = pd.concat(port_parts).sort_index()
    port = port[~port.index.duplicated(keep="first")]
    m = port.resample("ME").sum(); m = m[m != 0]
    sh_m = m.mean() / m.std(ddof=1) * np.sqrt(12) if len(m) > 5 and m.std() > 0 else np.nan
    eq = np.cumsum(port.values); dd = (eq - np.maximum.accumulate(eq)).min()
    return float(sh_m), float(dd), len(m), m.values


def hac_sharpe(m, L):
    """Annualized Newey-West HAC monthly Sharpe (accounts for serial correlation)."""
    m = np.asarray(m, float); m = m[np.isfinite(m)]
    T = len(m); mu = m.mean(); dem = m - mu
    g0 = np.dot(dem, dem) / T
    var = g0
    for l in range(1, L + 1):
        gl = np.dot(dem[l:], dem[:-l]) / T
        var += 2.0 * (1.0 - l / (L + 1.0)) * gl
    se = np.sqrt(max(var, 1e-18))
    ac1 = (np.dot(dem[1:], dem[:-1]) / T) / g0 if g0 > 0 else np.nan
    return mu / se * np.sqrt(12), ac1, np.sqrt(var / g0)


def main():
    ov = overlap()
    print(f"CLEAN-ROOM independent verify: {len(ov)} symbols on both venues")
    px_bn = load("spot_1h", ov); px_cb = load("coinbase_1h", ov)
    print(f"{'venue':<10}{'config':<14}{'monthlyS':>10}{'maxDD%':>9}{'n_mo':>6}")
    print("-" * 50)
    res = {}; mser = {}
    for vname, px in [("Binance", px_bn), ("Coinbase", px_cb)]:
        for cfg, stop in [("no-stop", None), ("stop|z|=4", 4.0)]:
            for npairs in (20,):
                sh, dd, nm, mvals = backtest(px, npairs, stop=stop)
                res[(vname, cfg)] = sh; mser[(vname, cfg)] = mvals
                print(f"{vname:<10}{cfg+' '+str(npairs)+'p':<14}{sh:>10.2f}{dd*100:>9.1f}{nm:>6}")
    bn = res.get(("Binance", "no-stop"), np.nan); cb = res.get(("Coinbase", "no-stop"), np.nan)
    print("\nINDEPENDENT VERDICT (clean-room, no wf_* imports):")
    print(f"  Binance no-stop  ~{bn:.2f}  (doc claim: ~2-2.5)")
    print(f"  Coinbase no-stop ~{cb:.2f}  (doc claim: ~1.0-1.2)")
    print(f"  Coinbase/Binance ratio = {cb/bn:.2f}  (doc claim: ~0.4-0.5)")
    print(f"  stop|z|=4 Binance {res.get(('Binance','stop|z|=4')):.2f}, Coinbase {res.get(('Coinbase','stop|z|=4')):.2f}  (doc: both lose)")
    print("\nHAC verify (skeptic's new finding — is the MONTHLY Sharpe itself autocorrelation-inflated?):")
    for vname in ("Binance", "Coinbase"):
        mvals = mser[(vname, "no-stop")]
        naive = res[(vname, "no-stop")]
        h3, ac1, infl3 = hac_sharpe(mvals, 3); h6, _, infl6 = hac_sharpe(mvals, 6)
        print(f"  {vname:<9} no-stop: naive={naive:.2f}  AC(1)={ac1:+.2f}  HAC(lag3)={h3:.2f}  HAC(lag6)={h6:.2f}")


if __name__ == "__main__":
    main()
