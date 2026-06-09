"""
The combined "all-hits-at-once" test (red-team B1 fix): the project corrected the
no-stop result for FREQUENCY (monthly), SURVIVORSHIP (delisted coins), and SELECTION
(deflated Sharpe) — but never on the SAME run. The headline ~1.7-2.5 was top-50 +
monthly; the survivorship dent was the 204-universe; the DSR was a third config set.
Grafting their haircuts into one "~1.6-2.4, survives" number is a category error.

This runs ONE test: the no-stop config on the **top-50 survivors PLUS the fully-delisted
coins** (LUNA/UST/FTT/LUNC, point-in-time entry via the >=90%-coverage gate, delisting
exit), at **monthly** frequency, then applies the **deflated Sharpe** using the cross-config
variance from the SAME run. sr_var is reported BOTH ways (all searched configs, and
no-stop-variants-only — red-team I2) so the framing isn't load-bearing.

Run:  python scratch/nostop_combined.py
"""
import os, sys
import numpy as np
import pandas as pd
import scipy.stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import wf_robustness as wr
import wf_survivorship as ws
sys.path.insert(0, os.path.join(wb.ROOT, "phase2_l2"))
from src.statistics import deflated_sharpe  # noqa

CC = wb.COST_LEVELS["realistic_30bps_rt"]
DELI_DIR = os.path.join(wb.ROOT, "data", "spot_1h_delisted")
DELISTED = ["LUNAUSDT", "USTUSDT", "FTTUSDT", "LUNCUSDT"]
base = dict(stop=np.inf, exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")
CONFIGS = [  # the searched risk-rule set (iter-12 frontier); flag which are no-stop variants
    ("10p nostop z", 10, {}, True), ("20p nostop z", 20, {}, True), ("40p nostop z", 40, {}, True),
    ("40p nostop z vol", 40, {"sizing": "vol", "vol_win": 720}, True),
    ("40p nostop conv", 40, {"exit_mode": "conv", "max_hold": 1000}, True),
    ("40p widestop6 z", 40, {"stop": 6.0}, False),
    ("40p widestop6 vol", 40, {"stop": 6.0, "sizing": "vol", "vol_win": 720}, False),
    ("20p nostop z vol", 20, {"sizing": "vol", "vol_win": 720}, True),
    ("40p tightstop4 z", 40, {"stop": 4.0}, False), ("10p tightstop4 z", 10, {"stop": 4.0}, False),
]
SELECTED = "40p nostop z"


def load_top50_plus_delisted():
    top50 = [s.strip().upper() for s in open(wb.UNIV) if s.strip()]
    px = ws.load_universe_from(top50)
    extra = {}
    for s in DELISTED:
        f = os.path.join(DELI_DIR, f"{s}.parquet")
        if os.path.exists(f):
            c = pd.read_parquet(f, columns=["close"])["close"]; c.index = pd.to_datetime(c.index, utc=True)
            extra[s] = c[~c.index.duplicated(keep="last")]
    px = pd.concat([px, pd.DataFrame(extra).reindex(px.index).ffill(limit=3)], axis=1)
    return px


def monthly_series(px, cfg, npairs):
    logpx = np.log(px); cols = list(px.columns); splits = wb.make_splits(px.index)
    parts = []
    for (tr_s, tr_e, te_e) in splits:
        tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(tr) < 1000 or len(te) < 200:
            continue
        avail = [c for c in cols if tr[c].notna().mean() >= wb.MIN_OBS_FRAC]
        picks = wb.select_pairs_ou(tr[avail], avail, npairs)
        if not picks:
            continue
        nmat = []
        for p in picks:
            r = wr.simulate_pair_v(te, p["a"], p["b"], p["alpha"], p["beta"], p["mu"], p["sd"],
                                   CC["fee_bps"], CC["slip_bps"], cfg)
            if r is not None:
                nmat.append(pd.Series(r["net"], index=te.index[:len(r["net"])]))
        if nmat:
            L = min(len(v) for v in nmat)
            parts.append(pd.Series(np.mean([v.values[:L] for v in nmat], axis=0), index=te.index[:L]))
    s = pd.concat(parts).sort_index(); s = s[~s.index.duplicated(keep="first")]
    m = s.resample("ME").sum(); return m[m != 0].values


def main():
    px = load_top50_plus_delisted()
    print(f"universe: top-50 survivors + {len(DELISTED)} delisted = {px.shape[1]} symbols, point-in-time gated")
    sh, sh_nostop, sel = [], [], None
    for label, n, ov, is_ns in CONFIGS:
        cfg = dict(base); cfg.update(ov)
        m = monthly_series(px, cfg, n)
        s = m.mean() / m.std(ddof=1) if m.std() > 0 else np.nan
        sh.append(s)
        if is_ns:
            sh_nostop.append(s)
        if label == SELECTED:
            sel = m
        print(f"  {label:<20} monthly SR={s:+.3f} (ann {s*np.sqrt(12):+.2f})  n_mo={len(m)}")
    sr_obs = sel.mean() / sel.std(ddof=1)
    skew, kurt, T = float(sps.skew(sel)), float(sps.kurtosis(sel, fisher=False)), len(sel)
    print(f"\nSELECTED {SELECTED} on the COMBINED universe: monthly SR={sr_obs:+.3f} "
          f"(ann {sr_obs*np.sqrt(12):+.2f})  skew={skew:+.2f} kurt={kurt:.2f} T={T}")
    for tag, arr in [("ALL configs", sh), ("no-stop variants only", sh_nostop)]:
        v = np.var([x for x in arr if np.isfinite(x)], ddof=1)
        print(f"\n  deflated Sharpe (sr_var from {tag}, var={v:.4f}):")
        for N in [len(CONFIGS), 25]:
            sr0, p = deflated_sharpe(sr_obs, v, N, skew, kurt, T)
            print(f"    N={N}: sr_0(ann)={sr0*np.sqrt(12):+.2f}  DSR={p:.3f} "
                  f"-> {'survives' if p>0.95 else ('marginal' if p>0.5 else 'FAILS')}")
    print("\n>>> This single number (combined monthly SR + its DSR) is the honest all-hits-at-once "
          "result; it replaces the chained top-50/204/config haircuts.")


if __name__ == "__main__":
    main()
