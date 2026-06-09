"""
Cross-exchange validation data pull: hourly OHLCV from COINBASE (a different venue
from Binance — different participants, USD not USDT quote), for the Binance-universe
majors that also list on Coinbase. Lets us re-run the no-stop reversion on an
INDEPENDENT exchange to test whether the ~2.3-2.5 Sharpe replicates (real effect) or
not (Binance-specific selection/overfitting) — the one orthogonal test of the result.

Coinbase candles API: max 300 candles/request, [time(s), low, high, open, close, vol].
Paginate forward in 300-hour windows from 2021-01-01. Rate-limited + retried.
Saves data/coinbase_1h/{SYM}USDT.parquet (USDT-named to match the universe loader),
schema = spot_1h (UTC 'timestamp' index; open/high/low/close/volume).

Run:  python scratch/coinbase_pull.py
"""
import os, sys, time, json, urllib.request, urllib.error
import numpy as np
import pandas as pd

ROOT = r"C:\Users\jackw\Desktop\math-199-research"
OUT = os.path.join(ROOT, "data", "coinbase_1h")
os.makedirs(OUT, exist_ok=True)
START = pd.Timestamp("2021-01-01T00:00:00Z")
END = pd.Timestamp("2026-06-01T00:00:00Z")
STEP_H = 300  # candles per request
UA = {"User-Agent": "research"}


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(1.0 + i)  # rate limited
            else:
                time.sleep(0.4)
        except Exception:
            time.sleep(0.5)
    return None


def coinbase_majors():
    syms = [s.strip().upper().replace("USDT", "") for s in open(os.path.join(ROOT, "data", "l2_universe_top50.txt")) if s.strip()]
    out = []
    for c in syms:
        if c in ("USDC", "DAI"):
            continue
        d = get(f"https://api.exchange.coinbase.com/products/{c}-USD")
        if d and d.get("id"):
            out.append(c)
        time.sleep(0.05)
    return out


def pull_symbol(c):
    prod = f"{c}-USD"
    rows = {}
    t = START
    while t < END:
        e = min(t + pd.Timedelta(hours=STEP_H), END)
        u = (f"https://api.exchange.coinbase.com/products/{prod}/candles?granularity=3600"
             f"&start={t.strftime('%Y-%m-%dT%H:%M:%SZ')}&end={e.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        d = get(u)
        if d:
            for row in d:  # [time, low, high, open, close, vol]
                rows[int(row[0])] = row
        t = e
        time.sleep(0.22)
    if not rows:
        return None
    df = pd.DataFrame([rows[k] for k in sorted(rows)], columns=["t", "low", "high", "open", "close", "volume"])
    df.index = pd.to_datetime(df["t"], unit="s", utc=True); df.index.name = "timestamp"
    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[out["close"] > 0]


def main():
    t0 = time.time()
    majors = coinbase_majors()
    print(f"{len(majors)} Binance-universe majors on Coinbase USD: {majors}", flush=True)
    for i, c in enumerate(majors):
        cache = os.path.join(OUT, f"{c}USDT.parquet")
        if os.path.exists(cache):
            print(f"  [{i+1}/{len(majors)}] {c}: cached", flush=True); continue
        df = pull_symbol(c)
        if df is None or len(df) < 5000:
            print(f"  [{i+1}/{len(majors)}] {c}: insufficient ({0 if df is None else len(df)} bars)", flush=True); continue
        df.to_parquet(cache)
        print(f"  [{i+1}/{len(majors)}] {c}: {len(df)} hourly bars {df.index.min().date()}..{df.index.max().date()} "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)
    print(f"DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
