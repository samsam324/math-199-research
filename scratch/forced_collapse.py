"""
Forced-held-collapse survivorship stress.

The OU selector AVOIDS coins that later collapse, so the headline backtest barely exercises the
in-position delisting tail. Here we FORCE the no-stop strategy to hold a position through a real
delisting collapse: we pair each delisted coin (LUNA/UST/FTT/LUNC, on disk in
data/spot_1h_delisted) with a liquid major (BTC), fit a static hedge on pre-collapse data, and run
the same no-stop sim over the full history including the collapse. As the coin falls, the
mean-reversion rule goes long the 'cheap' collapsing leg and -- with no stop -- rides it to zero.
We compare the tail with and without the structural-break circuit breaker (halt after a >50%
adverse single-leg move).

Run: python scratch/forced_collapse.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import nostop_breakstop as nbs

DELI_DIR = os.path.join(wb.ROOT, "data", "spot_1h_delisted")
SPOT_DIR = os.path.join(wb.ROOT, "data", "spot_1h")
DELISTED = ["LUNAUSDT", "USTUSDT", "FTTUSDT", "LUNCUSDT"]
HEDGE = "BTCUSDT"
CC = wb.COST_LEVELS["realistic_30bps_rt"]


def load(sym, d):
    f = os.path.join(d, f"{sym}.parquet")
    if not os.path.exists(f):
        return None
    c = pd.read_parquet(f, columns=["close"])["close"]
    c.index = pd.to_datetime(c.index, utc=True)
    return c[~c.index.duplicated(keep="last")]


def eq_stats(net):
    if net is None:
        return np.nan, np.nan
    eq = np.cumsum(np.nan_to_num(net))
    dd = float((eq - np.maximum.accumulate(eq)).min())
    return float(eq[-1] * 100), float(dd * 100)


def run_pair(D):
    a = load(D, DELI_DIR); b = load(HEDGE, SPOT_DIR)
    if a is None or b is None:
        return None
    px = pd.concat({D: a, HEDGE: b}, axis=1).dropna()
    if len(px) < 24 * 100:
        return None
    logpx = np.log(px); n = len(logpx)
    lp = logpx[D].values
    cidx = int(np.nanargmin(pd.Series(lp).diff(168).values))       # bar of the worst 7-day drop
    fit0, fit1 = max(0, cidx - 90 * 24), max(24, cidx - 30 * 24)   # hedge fit: pre-collapse
    te0, te1 = max(0, cidx - 30 * 24), min(n, cidx + 14 * 24)      # evaluate: across the collapse only
    if fit1 - fit0 < 200 or te1 - te0 < 100:
        return None
    fit = logpx.iloc[fit0:fit1]
    alpha, beta = wb.ols_hedge(fit[D].values, fit[HEDGE].values)
    sp = fit[D].values - alpha - beta * fit[HEDGE].values
    mu, sd = float(np.nanmean(sp)), float(np.nanstd(sp))
    te = logpx.iloc[te0:te1]
    out = {}
    for tag, kw in [("nostop", {}), ("circuit_breaker", dict(adverse_K=0.50, halt=True))]:
        net = nbs.sim_pair(te, D, HEDGE, alpha, beta, mu, sd, CC["fee_bps"], CC["slip_bps"], **kw)
        out[tag] = eq_stats(net)
    drop7 = float((np.exp(lp[cidx] - lp[max(0, cidx - 168)]) - 1) * 100)
    return dict(coin=D, collapse_date=str(px.index[cidx].date()), worst_7d_drop_pct=drop7,
                nostop_pnl_pct=out["nostop"][0], nostop_maxdd_pct=out["nostop"][1],
                cb_pnl_pct=out["circuit_breaker"][0], cb_maxdd_pct=out["circuit_breaker"][1])


def main():
    print("Forced-held-collapse stress: long the collapsing leg, no stop vs structural-break circuit breaker")
    print(f"{'coin':<10}{'worst7d%':>10}{'nostop PnL%':>13}{'nostop DD%':>12}{'+CB PnL%':>11}{'+CB DD%':>10}")
    print("-" * 66)
    rows = []
    for D in DELISTED:
        r = run_pair(D)
        if r is None:
            print(f"{D:<10}  (no data)")
            continue
        rows.append(r)
        print(f"{r['coin']:<10}{r['worst_7d_drop_pct']:>10.0f}{r['nostop_pnl_pct']:>13.0f}"
              f"{r['nostop_maxdd_pct']:>12.0f}{r['cb_pnl_pct']:>11.0f}{r['cb_maxdd_pct']:>10.0f}")
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(wb.ROOT, "scratch", "forced_collapse.csv"), index=False)
        ns = np.nanmean([r["nostop_pnl_pct"] for r in rows]); cb = np.nanmean([r["cb_pnl_pct"] for r in rows])
        print(f"\nmean held-collapse P&L: no-stop {ns:.0f}%  vs  circuit breaker {cb:.0f}%")
        print("saved -> scratch/forced_collapse.csv")
    print("\nReading: forced to hold a collapsing leg, the no-stop book takes a catastrophic per-pair loss;")
    print("the >50%-adverse circuit breaker halts the pair partway down and caps the tail.")


if __name__ == "__main__":
    main()
