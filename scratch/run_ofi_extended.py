"""Re-run the EXACT book-OFI 2024 analysis with extended horizons for Fig 6.

Imports scratch/book_ofi_2024.py and only overrides HORIZONS so the logic is
byte-for-byte identical to what produced book_ofi_2024.log; the overlapping
horizons (1,5,10,30) must reproduce the committed log values, which validates
the new ones (2,60,120,300).

Writes scratch/ofi_decay_extended.csv with the ALL-regime contemporaneous and
predictive incremental book-OFI R^2 per symbol/horizon.
Run: python scratch/run_ofi_extended.py
"""
import os, sys
import pandas as pd

ROOT = r"C:\Users\jackw\Desktop\math-199-research"
sys.path.insert(0, os.path.join(ROOT, "scratch"))
import book_ofi_2024 as bo

bo.HORIZONS = [1, 2, 5, 10, 30, 60, 120, 300]
print("HORIZONS =", bo.HORIZONS)
print("DATES    =", bo.DATES)
print("SYMS     =", bo.SYMS)

all_summ = {}
for s in bo.SYMS:
    res = bo.analyze_symbol(s, trail=10)
    if res is not None:
        all_summ[s] = res
bo.print_master_tables(all_summ)

rows = []
for sym, s in all_summ.items():
    a = s["ALL"]
    row = {"sym": sym, "contemp_incr_book": a["contemp_incr_book"]}
    for h in bo.HORIZONS:
        row[f"h{h}"] = a["pred"].get(h, {}).get("incr_book", float("nan"))
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "scratch", "ofi_decay_extended.csv"), index=False)
print("\n=== ofi_decay_extended.csv ===")
print(out.to_string(index=False))
print("\nDONE.")
