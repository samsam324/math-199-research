"""
Task 2 — pull hourly OHLCV for the fully-delisted coins (the survivorship-worst
tail), from Binance public data (data.binance.vision monthly 1h klines), which
includes the collapse months so the no-stop strategy's exposure to a delisting is
tested honestly. Tardis confirmed it has these symbols' L2+trades too (documented
in SURVIVORSHIP_DELISTED.md); klines suffice for the hourly reversion test and are
free/fast.

Saves data/spot_1h_delisted/{SYM}.parquet matching the spot_1h schema
(UTC 'timestamp' index; open/high/low/close/volume).

Run:  python scratch/t2_pull_delisted.py
"""
import os, io, zipfile, urllib.request
import numpy as np
import pandas as pd

ROOT = r"C:\Users\jackw\Desktop\math-199-research"
OUT = os.path.join(ROOT, "data", "spot_1h_delisted")
os.makedirs(OUT, exist_ok=True)

# (symbol, first_month, last_month) — generous windows; missing months are skipped.
TARGETS = [
    ("LUNAUSDT", "2021-01", "2022-06"),   # Terra collapse May 2022
    ("USTUSDT",  "2021-12", "2022-06"),   # UST depeg May 2022 (Tardis data ends 2022-05-14)
    ("FTTUSDT",  "2020-01", "2023-06"),   # FTX collapse Nov 2022
    ("LUNCUSDT", "2022-09", "2026-05"),   # Luna Classic (post-rebrand)
]
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "qav", "trades", "tbb", "tbq", "ignore"]


def months(a, b):
    return [d.strftime("%Y-%m") for d in pd.date_range(a + "-01", b + "-01", freq="MS")]


def fetch_month(sym, ym):
    u = f"https://data.binance.vision/data/spot/monthly/klines/{sym}/1h/{sym}-1h-{ym}.zip"
    try:
        with urllib.request.urlopen(u, timeout=40) as r:
            blob = r.read()
    except Exception:
        return None
    try:
        z = zipfile.ZipFile(io.BytesIO(blob))
        df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    except Exception:
        return None
    # Binance changed some files to include a header row "open_time,..." — drop it
    if str(df.iloc[0, 0]).lower().startswith("open"):
        df = df.iloc[1:].reset_index(drop=True)
    df = df.iloc[:, :12]
    df.columns = KCOLS
    return df


def main():
    for sym, m0, m1 in TARGETS:
        frames = []
        got = []
        for ym in months(m0, m1):
            d = fetch_month(sym, ym)
            if d is not None and len(d):
                frames.append(d); got.append(ym)
        if not frames:
            print(f"{sym}: NO DATA in {m0}..{m1}"); continue
        df = pd.concat(frames, ignore_index=True)
        ot = pd.to_numeric(df["open_time"], errors="coerce")
        # Binance switched kline open_time from ms to microseconds in 2025, so a
        # symbol spanning that boundary has MIXED units -> detect per row.
        us_mask = ot > 1e14
        ts = pd.to_datetime(np.where(us_mask, ot, ot * 1000.0), unit="us", utc=True)
        out = pd.DataFrame({
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
        })
        out.index = ts; out.index.name = "timestamp"
        out = out[~out.index.duplicated(keep="last")].sort_index().dropna(subset=["close"])
        out = out[out["close"] > 0]
        out.to_parquet(os.path.join(OUT, f"{sym}.parquet"))
        print(f"{sym}: {len(out)} hourly bars  {out.index.min()} -> {out.index.max()}  "
              f"(months {got[0]}..{got[-1]})  last close={out['close'].iloc[-1]:.6g} "
              f"min close={out['close'].min():.6g}")


if __name__ == "__main__":
    main()
