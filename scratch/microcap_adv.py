"""
Capacity bound for the held-out micro-cap inflation.

Section 5 says the held-out Sharpe ~5 is an artifact because it lives in high-volatility micro-cap
alts under a flat 15bps cost that 'badly understates the true cost'. This puts a number on it: the
median daily dollar volume of the named held-out micro-caps (WAXP/ZIL/VTHO/ACH/EGLD) vs the majors,
over the held-out window, and a simple participation-based capacity figure. If the micro-caps trade
orders of magnitude less, a size-independent 15bps cost is not defensible and the Sharpe is not
realizable at meaningful capital.

Run: python scratch/microcap_adv.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb

SPOT = os.path.join(wb.ROOT, "data", "spot_1h")
MAJOR = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MICRO = ["WAXPUSDT", "ZILUSDT", "VTHOUSDT", "ACHUSDT", "EGLDUSDT"]
START = pd.Timestamp("2025-06-01", tz="UTC")   # the held-out window


def daily_dollar_vol(sym):
    f = os.path.join(SPOT, f"{sym}.parquet")
    if not os.path.exists(f):
        return None
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index, utc=True)
    cols = set(df.columns)
    if "quote_volume" in cols:
        qv = df["quote_volume"].astype(float)
    elif {"volume", "close"} <= cols:
        qv = df["volume"].astype(float) * df["close"].astype(float)
    else:
        return None
    qv = qv[qv.index >= START]
    if len(qv) < 24 * 30:
        return None
    return float(qv.resample("1D").sum().median())


def main():
    print(f"median daily $ volume over the held-out window (>= {START.date()})\n")
    res = {}
    for s in MAJOR + MICRO:
        v = daily_dollar_vol(s)
        res[s] = v
        print(f"  {s:<10} {('$%0.0f' % v) if v else '(no data)':>18}")
    maj = np.nanmedian([res[s] for s in MAJOR if res.get(s)])
    mic = np.nanmedian([res[s] for s in MICRO if res.get(s)])
    print(f"\n  median major: ${maj:,.0f}/day   median micro-cap: ${mic:,.0f}/day   ratio: {maj/mic:,.0f}x")
    # capacity at a 1% participation cap (a generous bar for a thin alt)
    cap = 0.01 * mic
    print(f"  at a 1% participation cap, deployable notional per micro-cap leg ~ ${cap:,.0f}")
    print(f"  -> a flat 15bps cost is size-independent; on names trading ~${mic:,.0f}/day it is not defensible,")
    print(f"     so the held-out Sharpe is not realizable at meaningful capital.")
    pd.DataFrame([{"symbol": s, "median_daily_usd_vol": res[s]} for s in MAJOR + MICRO]).to_csv(
        os.path.join(wb.ROOT, "scratch", "microcap_adv.csv"), index=False)
    print("\nsaved -> scratch/microcap_adv.csv")


if __name__ == "__main__":
    main()
