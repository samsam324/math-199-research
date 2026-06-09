"""
Task 2 — worst-case tail: what the no-stop rule DODGED. Force every delisted-leg
pair (LUNA/UST/FTT vs each other symbol) to be held through the MAY-2022 collapse
test window (train 2021-10..2022-03, test 2022-04..06), and report the per-pair
no-stop P&L distribution. Shows the catastrophic tail the reversion-speed selection
+ coverage gate avoided in the actual run (t2_survivorship_rerun.py).

Run:  python scratch/t2_worstcase.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import wf_robustness as wr
import wf_survivorship as ws

DEL = os.path.join(wb.ROOT, "data", "spot_1h_delisted")
LEGS = ["LUNAUSDT", "USTUSDT", "FTTUSDT"]
CFG = dict(stop=np.inf, exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")


def main():
    px = ws.load_universe_from(ws.all_spot_symbols())
    extra = {}
    for s in LEGS:
        c = pd.read_parquet(os.path.join(DEL, f"{s}.parquet"), columns=["close"])["close"]
        c.index = pd.to_datetime(c.index, utc=True)
        extra[s] = c[~c.index.duplicated(keep="last")]
    px = pd.concat([px, pd.DataFrame(extra).reindex(px.index).ffill(limit=3)], axis=1)
    logpx = np.log(px); cols = list(px.columns)
    tr = logpx[(logpx.index >= pd.Timestamp("2021-10-01", tz="UTC")) & (logpx.index < pd.Timestamp("2022-04-01", tz="UTC"))]
    te = logpx[(logpx.index >= pd.Timestamp("2022-04-01", tz="UTC")) & (logpx.index < pd.Timestamp("2022-07-01", tz="UTC"))]

    res = []
    for leg in LEGS:
        cov = tr[leg].notna().mean()
        if cov < 0.9:
            print(f"  {leg}: train coverage {cov:.0%} < 90% -> EXCLUDED by point-in-time gate")
            continue
        for other in cols:
            if other == leg or other in ws.STABLES:
                continue
            m = tr[[leg, other]].notna().all(axis=1)
            if m.mean() < 0.9:
                continue
            ya, yb = tr[leg].values, tr[other].values
            mask = np.isfinite(ya) & np.isfinite(yb)
            X = np.column_stack([np.ones(mask.sum()), yb[mask]])
            coef, *_ = np.linalg.lstsq(X, ya[mask], rcond=None)
            al, be = coef
            if be <= 0:
                continue
            sp = ya - al - be * yb
            sd, mu = np.nanstd(sp), np.nanmean(sp)
            r = wr.simulate_pair_v(te, leg, other, al, be, mu, sd, 10.0, 5.0, CFG)
            if r is not None:
                res.append((f"{leg}_{other}", float(np.nansum(r["net"])) * 100))
    res.sort(key=lambda x: x[1])
    v = [x[1] for x in res]
    print(f"\n  forced no-stop on {len(res)} delisted-leg pairs through the May-2022 collapse:")
    print(f"   worst: {res[0][0]} {res[0][1]:+.0f}%  | median {np.median(v):+.0f}%  | "
          f"best: {res[-1][0]} {res[-1][1]:+.0f}%  | mean {np.mean(v):+.0f}%")
    print("   (a typical non-delisted pair in a normal window earns ~+30-40%)")


if __name__ == "__main__":
    main()
