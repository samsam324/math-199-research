"""
Literal point-in-time top-N survivorship test (red-team / Task-2-audit follow-up).
The combined test used a FIXED pool (top-50 survivors + 4 delisted, coverage-gated) =
point-in-time *entry* only. The faithful version: a TIME-VARYING top-N-by-liquidity
universe rebuilt every walk-forward window from the FULL on-disk pool (204 symbols +
the 4 fully-delisted), so a coin is in the universe only while it is actually top-N
liquid (LUNA/FTT are IN during 2021-22, drop out as they fade/delist; other
then-liquid-now-faded coins are included too). No survivorship by construction.

For each window: rank symbols by TRAIN-window dollar volume (sum of close*volume),
require >=90% train coverage, take the top-N (=50); OU-select 40 pairs from that
point-in-time top-50; run the no-stop config; monthly Sharpe. Compare to the
combined test (~2.3) and the broad-204 run.

Run:  python scratch/nostop_pit.py
"""
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import wf_robustness as wr
import wf_survivorship as ws

CC = wb.COST_LEVELS["realistic_30bps_rt"]
CFG = dict(stop=np.inf, exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")
DELI_DIR = os.path.join(wb.ROOT, "data", "spot_1h_delisted")
DELISTED = ["LUNAUSDT", "USTUSDT", "FTTUSDT", "LUNCUSDT"]
TOPN, N_PAIRS = 50, 40


def load_all_with_dollarvol():
    """close panel + dollar-volume panel over the full 204+4 universe."""
    syms = ws.all_spot_symbols()
    closes, dvols = {}, {}
    for s in syms + DELISTED:
        f = (os.path.join(DELI_DIR, f"{s}.parquet") if s in DELISTED
             else os.path.join(ws.SPOT, f"{s}.parquet"))
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f, columns=["close", "volume"])
        d.index = pd.to_datetime(d.index, utc=True)
        d = d[~d.index.duplicated(keep="last")]
        closes[s] = d["close"]; dvols[s] = (d["close"] * d["volume"])
    px = pd.DataFrame(closes).sort_index()
    full = pd.date_range(px.index.min().floor("h"), px.index.max().ceil("h"), freq="h", tz="UTC")
    px = px.reindex(full)
    px = px[px.index >= pd.Timestamp(wb.START_FLOOR, tz="UTC")].ffill(limit=3)
    for stub in ws.STABLES:
        px = px.drop(columns=stub, errors="ignore")
    dv = pd.DataFrame(dvols).reindex(px.index)
    return px, dv


def run_pit(px, dv):
    logpx = np.log(px); cols = list(px.columns); splits = wb.make_splits(px.index)
    parts = []; univ_sizes = []; delisted_windows = 0
    for (tr_s, tr_e, te_e) in splits:
        tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(tr) < 1000 or len(te) < 200:
            continue
        cov = tr.notna().mean()
        elig = [c for c in cols if cov[c] >= wb.MIN_OBS_FRAC]
        # point-in-time top-N by TRAIN dollar volume
        dvw = dv.loc[(dv.index >= tr_s) & (dv.index < tr_e), elig].sum().sort_values(ascending=False)
        topn = list(dvw.head(TOPN).index)
        univ_sizes.append(len(topn))
        if any(s in topn for s in DELISTED):
            delisted_windows += 1
        picks = wb.select_pairs_ou(tr[topn], topn, N_PAIRS)
        if not picks:
            continue
        nmat = []
        for p in picks:
            r = wr.simulate_pair_v(te, p["a"], p["b"], p["alpha"], p["beta"], p["mu"], p["sd"],
                                   CC["fee_bps"], CC["slip_bps"], CFG)
            if r is not None:
                nmat.append(pd.Series(r["net"], index=te.index[:len(r["net"])]))
        if nmat:
            L = min(len(v) for v in nmat)
            parts.append(pd.Series(np.mean([v.values[:L] for v in nmat], axis=0), index=te.index[:L]))
    s = pd.concat(parts).sort_index(); s = s[~s.index.duplicated(keep="first")]
    sh_h, _, _ = wb.sharpe_hac(s.values)
    m = s.resample("ME").sum(); m = m[m != 0]
    sh_m = m.mean() / m.std(ddof=1) * np.sqrt(12)
    eq = np.cumsum(np.nan_to_num(s.values)); dd = (eq - np.maximum.accumulate(eq)).min()
    return sh_h, sh_m, dd, np.mean(univ_sizes), delisted_windows, len(parts)


def main():
    print("Building literal point-in-time top-50-by-liquidity universe (204+4 pool)...")
    px, dv = load_all_with_dollarvol()
    sh_h, sh_m, dd, avg_u, dw, nwin = run_pit(px, dv)
    print(f"\nLITERAL POINT-IN-TIME top-{TOPN}, no-stop {N_PAIRS}p (time-varying liquidity-ranked universe):")
    print(f"  hourly Sharpe = {sh_h:+.2f}")
    print(f"  MONTHLY Sharpe (frequency-honest) = {sh_m:+.2f}")
    print(f"  max drawdown = {dd*100:+.1f}%")
    print(f"  avg point-in-time universe size/window = {avg_u:.0f} | windows incl. a delisted coin = {dw}/{nwin}")
    print(f"\n  vs combined test (top-50+delisted fixed pool): monthly ~2.29")
    print(f"  vs broad-204 (iter-15): monthly ~3.76 (hourly 5.29)")
    print("  Reading: if monthly Sharpe stays ~2-2.5 here, the no-stop effect is robust to a")
    print("  fully point-in-time, survivorship-free liquidity-ranked universe.")


if __name__ == "__main__":
    main()
