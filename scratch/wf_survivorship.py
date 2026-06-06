"""
Iteration-15: the survivorship check, using data already on disk.

The backtests used data/l2_universe_top50.txt (50 current-liquid majors) = a
survivorship-filtered universe. But data/spot_1h holds 204 symbols, incl. ~154
lower-quality / pumped-and-crashed / meme names a "current top-50" filter removes
(1000REKTUSDT, USELESSUSDT, NOBODYUSDT, TROLLUSDT, FARTCOINUSDT, ...). Re-running
the no-stop vs stop reversion backtest on the FULL 204-symbol universe is a much
less survivorship-biased test: short-history coins only become selectable once they
have data (~point-in-time), and the universe now contains names that genuinely
decoupled. If the no-stop alpha survives here, it is not merely a survivor-majors
artifact; if it collapses, survivorship was load-bearing.

Runs the identical pipeline (OU selection, static hedge, z-rule) on BOTH universes
and reports hourly AND monthly (frequency-honest) Sharpe + max drawdown.

Run synchronously:  python scratch/wf_survivorship.py
"""
from __future__ import annotations
import os, sys, glob, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import wf_robustness as wr

np.seterr(all="ignore")
ROOT = wb.ROOT
SPOT = os.path.join(ROOT, "data", "spot_1h")
STABLES = {"USDCUSDT", "DAIUSDT", "USD1USDT", "TUSDUSDT", "USDUCUSDT", "FDUSDUSDT", "PAXGUSDT"}


def load_universe_from(symbols):
    closes = {}
    for s in symbols:
        f = os.path.join(SPOT, f"{s}.parquet")
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f, columns=["close"])
        sc = df["close"].copy(); sc.index = pd.to_datetime(sc.index, utc=True)
        sc = sc[~sc.index.duplicated(keep="last")]
        closes[s] = sc
    px = pd.DataFrame(closes).sort_index()
    full = pd.date_range(px.index.min().floor("h"), px.index.max().ceil("h"), freq="h", tz="UTC")
    px = px.reindex(full)
    px = px[px.index >= pd.Timestamp(wb.START_FLOOR, tz="UTC")]
    px = px.ffill(limit=3)
    for stub in STABLES:
        if stub in px.columns:
            px = px.drop(columns=stub)
    return px


def all_spot_symbols():
    return sorted(os.path.basename(p)[:-len(".parquet")] for p in glob.glob(os.path.join(SPOT, "*.parquet")))


def run_universe(px, label):
    cc = wb.COST_LEVELS["realistic_30bps_rt"]
    logpx = np.log(px); cols = list(px.columns); splits = wb.make_splits(px.index)
    cfgs = {
        "stop|z|=4 (10p)":   (10, dict(stop=4.0,    exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")),
        "no-stop z (10p)":   (10, dict(stop=np.inf, exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")),
        "no-stop z (40p)":   (40, dict(stop=np.inf, exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")),
    }
    series = {k: [] for k in cfgs}
    pair_counts = []
    n_avail = []
    for (tr_s, tr_e, te_e) in splits:
        tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(tr) < 1000 or len(te) < 200:
            continue
        # symbols with >=90% coverage in THIS train window (point-in-time gating)
        avail = [c for c in cols if tr[c].notna().mean() >= wb.MIN_OBS_FRAC]
        n_avail.append(len(avail))
        picks40 = wb.select_pairs_ou(tr[avail] if avail else tr, avail, 40)
        if not picks40:
            continue
        pair_counts.append(len(picks40))
        for k, (npairs, cfg) in cfgs.items():
            picks = picks40[:npairs]
            nmat = []
            for p in picks:
                r = wr.simulate_pair_v(te, p["a"], p["b"], p["alpha"], p["beta"],
                                       p["mu"], p["sd"], cc["fee_bps"], cc["slip_bps"], cfg)
                if r is not None:
                    nmat.append(pd.Series(r["net"], index=te.index[:len(r["net"])]))
            if nmat:
                L = min(len(v) for v in nmat)
                series[k].append(pd.Series(np.mean([v.values[:L] for v in nmat], axis=0), index=te.index[:L]))
    print(f"\n##### UNIVERSE: {label}  (cols={px.shape[1]}, splits={len(splits)}, "
          f"avg usable syms/window={np.mean(n_avail):.0f}, avg ranked pairs={np.mean(pair_counts):.0f}) #####")
    print(f"  {'config':<20}{'hourlyS':>9}{'monthlyS':>10}{'maxDD%':>9}{'netPnL%':>9}")
    for k in cfgs:
        if not series[k]:
            print(f"  {k:<20}  NO RESULT"); continue
        s = pd.concat(series[k]).sort_index(); s = s[~s.index.duplicated(keep="first")]
        sh_h, _, _ = wb.sharpe_hac(s.values)
        m = s.resample("ME").sum(); m = m[m != 0]
        sh_m = float(m.mean() / m.std(ddof=1) * np.sqrt(12)) if len(m) > 5 and m.std() > 0 else np.nan
        eq = np.cumsum(np.nan_to_num(s.values)); dd = float((eq - np.maximum.accumulate(eq)).min())
        print(f"  {k:<20}{sh_h:>9.2f}{sh_m:>10.2f}{dd*100:>9.1f}{float(np.nansum(s.values))*100:>9.0f}")


def main():
    t0 = time.time()
    print("=" * 80)
    print("ITER-15: SURVIVORSHIP CHECK -- top-50 vs full 204-symbol on-disk universe")
    print("=" * 80)
    # top-50 (survivorship-filtered) -- identical to the main backtests
    top50 = [s.strip().upper() for s in open(wb.UNIV) if s.strip()]
    px50 = load_universe_from(top50)
    run_universe(px50, "top-50 (survivorship-filtered)")
    # full on-disk universe (204 syms incl. low-quality / crashed names)
    allsyms = all_spot_symbols()
    pxall = load_universe_from(allsyms)
    run_universe(pxall, "FULL on-disk (204 incl. meme/crashed)")
    print("\n" + "=" * 80)
    print("READING: if no-stop survives (similar Sharpe) on the FULL universe, the alpha is not")
    print("merely a survivor-majors artifact. If it collapses, survivorship was load-bearing.")
    print(f"DONE in {time.time()-t0:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
