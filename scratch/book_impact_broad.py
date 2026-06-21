"""Broaden the size->impact table (Table 5) from 4 symbols to the top-50 universe.

Reuses impact_decomp_2024.compute_responses + summarize unchanged. The per-class
response means carry no randomness, so the four majors reproduce Table 5 exactly.
Tests whether institutional trades (>$10k) carry larger one second price impact
than retail (<$1k) across the whole liquid universe, i.e. whether trade size
proxies information everywhere or only on the four majors.

Run: python scratch/book_impact_broad.py   (after the execution run frees the disk)
"""
import os
import time
import importlib.util
import numpy as np
import pandas as pd

ROOT = r"C:\Users\jackw\Desktop\math-199-research"
spec = importlib.util.spec_from_file_location("im", os.path.join(ROOT, "scratch", "impact_decomp_2024.py"))
im = importlib.util.module_from_spec(spec)
spec.loader.exec_module(im)

uni = [s.strip() for s in open(os.path.join(ROOT, "data", "l2_universe_top50.txt")) if s.strip()]
MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]
SYMS = MAJORS + [s for s in uni if s not in MAJORS]

rows = []
t0 = time.time()
for sym in SYMS:
    try:
        allr, _ = im.compute_responses(sym)
        summ = im.summarize(allr)
        r = {"sym": sym}
        for cl in im.CLASSES:
            r[f"{cl}_1s"] = summ[cl]["raw"][1]
            r[f"{cl}_n"] = summ[cl]["n"]
            r[f"{cl}_perm"] = summ[cl]["raw"][300]
        rt = r.get("retail_1s", np.nan)
        r["inst_over_retail"] = r["institutional_1s"] / rt if rt and rt > 0 else np.nan
        rows.append(r)
        print(f"  {sym:10s} retail={r['retail_1s']:.3f} mid={r['mid_1s']:.3f} "
              f"inst={r['institutional_1s']:.3f}  inst/retail={r['inst_over_retail']:.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  {sym:10s} ERROR {e}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "scratch", "impact_broad_50sym.csv"), index=False)

ref = {"BTCUSDT": (0.13, 0.23, 0.33), "ETHUSDT": (0.15, 0.24, 0.32),
       "SOLUSDT": (0.15, 0.26, 0.32), "AVAXUSDT": (0.18, 0.38, 0.38)}
print("\nSANITY vs Table 5 (retail/mid/institutional 1s RAW bps):")
for s, (rt, md, it) in ref.items():
    row = out[out.sym == s].iloc[0]
    ok = ("OK" if abs(row.retail_1s - rt) < 0.01 and abs(row.mid_1s - md) < 0.01
          and abs(row.institutional_1s - it) < 0.01 else "CHECK")
    print(f"  {s:8s} ref={rt}/{md}/{it}  got={row.retail_1s:.2f}/{row.mid_1s:.2f}/{row.institutional_1s:.2f}  {ok}")

print("\nSIZE -> IMPACT across the 50:")
mono = out[out.institutional_1s > out.retail_1s]
print(f"  institutional 1s impact > retail in {len(mono)}/{len(out)} symbols")
print(f"  inst/retail ratio: median {out.inst_over_retail.median():.2f}  "
      f"min {out.inst_over_retail.min():.2f}  max {out.inst_over_retail.max():.2f}")
print(out[["sym", "retail_1s", "mid_1s", "institutional_1s", "inst_over_retail"]]
      .sort_values("inst_over_retail").to_string(index=False))
print("DONE.")
