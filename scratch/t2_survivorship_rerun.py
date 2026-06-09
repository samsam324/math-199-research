"""
Task 2 — re-run the no-stop reversion alpha with the fully-delisted coins added to
the selectable pool, point-in-time (a coin enters a window only when it has >=90%
coverage in that window's train; its data ends at its actual delisting, after which
an open position marks the collapse and then goes flat). The collapse months ARE in
the data (LUNA -> $5e-5, UST depeg, FTT -> delist), so the survivorship-worst tail is
tested honestly. Compares 204-symbol on-disk universe vs 204 + {LUNA,UST,FTT,LUNC}.

Reports monthly (frequency-honest) + hourly Sharpe and max drawdown for each, and
crucially: in which walk-forward windows a delisted coin was actually selected, and
the realized P&L on those pairs (does holding a coin through its collapse blow up the
no-stop book?).

Run:  python scratch/t2_survivorship_rerun.py
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import wf_robustness as wr
import wf_survivorship as ws

DELISTED_DIR = os.path.join(wb.ROOT, "data", "spot_1h_delisted")
DELISTED = ["LUNAUSDT", "USTUSDT", "FTTUSDT", "LUNCUSDT"]
CFG = dict(stop=np.inf, exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")
N_PAIRS = 40


def load_combined(include_delisted):
    syms = ws.all_spot_symbols()
    px = ws.load_universe_from(syms)
    if include_delisted:
        extra = {}
        for s in DELISTED:
            f = os.path.join(DELISTED_DIR, f"{s}.parquet")
            if not os.path.exists(f):
                continue
            c = pd.read_parquet(f, columns=["close"])["close"]
            c.index = pd.to_datetime(c.index, utc=True)
            extra[s] = c[~c.index.duplicated(keep="last")]
        ex = pd.DataFrame(extra).reindex(px.index).ffill(limit=3)
        px = pd.concat([px, ex], axis=1)
    return px


def run(px, label):
    cc = wb.COST_LEVELS["realistic_30bps_rt"]
    logpx = np.log(px); cols = list(px.columns); splits = wb.make_splits(px.index)
    series = []
    delisted_hits = []   # (window, pair, pair_net_pnl)
    for (tr_s, tr_e, te_e) in splits:
        tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(tr) < 1000 or len(te) < 200:
            continue
        avail = [c for c in cols if tr[c].notna().mean() >= wb.MIN_OBS_FRAC]
        picks = wb.select_pairs_ou(tr[avail], avail, N_PAIRS)
        if not picks:
            continue
        nmat = []
        for p in picks:
            r = wr.simulate_pair_v(te, p["a"], p["b"], p["alpha"], p["beta"],
                                   p["mu"], p["sd"], cc["fee_bps"], cc["slip_bps"], CFG)
            if r is None:
                continue
            nmat.append(pd.Series(r["net"], index=te.index[:len(r["net"])]))
            if p["a"] in DELISTED or p["b"] in DELISTED:
                delisted_hits.append((f"{te.index[0].date()}..{te.index[-1].date()}",
                                      f"{p['a']}_{p['b']}", float(np.nansum(r["net"]))))
        if nmat:
            L = min(len(v) for v in nmat)
            series.append(pd.Series(np.mean([v.values[:L] for v in nmat], axis=0), index=te.index[:L]))
    s = pd.concat(series).sort_index(); s = s[~s.index.duplicated(keep="first")]
    sh_h, _, _ = wb.sharpe_hac(s.values)
    m = s.resample("ME").sum(); m = m[m != 0]
    sh_m = float(m.mean() / m.std(ddof=1) * np.sqrt(12)) if len(m) > 5 and m.std() > 0 else np.nan
    eq = np.cumsum(np.nan_to_num(s.values)); dd = float((eq - np.maximum.accumulate(eq)).min())
    print(f"\n##### {label}  (cols={px.shape[1]}) #####")
    print(f"  no-stop 40p:  hourly Sharpe={sh_h:+.2f}  monthly Sharpe={sh_m:+.2f}  "
          f"maxDD={dd*100:+.1f}%  netPnL={float(np.nansum(s.values))*100:+.0f}%")
    if delisted_hits:
        print(f"  DELISTED-coin pairs selected: {len(delisted_hits)}")
        for w, pr, pnl in delisted_hits:
            print(f"    window {w}: {pr:<22} pair net PnL = {pnl*100:+.1f}%")
    else:
        print("  DELISTED-coin pairs selected: NONE (the OU selection never picked a delisted-coin pair)")
    return dict(sh_h=sh_h, sh_m=sh_m, dd=dd, hits=delisted_hits)


def main():
    print("=" * 84)
    print("TASK 2: no-stop reversion alpha WITH vs WITHOUT fully-delisted coins in the pool")
    print("=" * 84)
    base = run(load_combined(False), "204-symbol on-disk universe (baseline)")
    deli = run(load_combined(True), "204 + {LUNA, UST, FTT, LUNC} (survivorship-worst added)")
    print("\n" + "=" * 84)
    print(f"VERDICT: monthly Sharpe {base['sh_m']:+.2f} -> {deli['sh_m']:+.2f} | "
          f"maxDD {base['dd']*100:+.1f}% -> {deli['dd']*100:+.1f}% | "
          f"delisted pairs ever selected: {len(deli['hits'])}")
    print("=" * 84)


if __name__ == "__main__":
    main()
