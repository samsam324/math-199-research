"""Broaden the execution-cost null from 4 symbols to the full top-50 universe.

Reuses exec_value_2024.py's simulate_day + block_metrics unchanged. The four event
level majors are iterated FIRST with the same rng(12345) and same DATES, so their
order-by-order draws match exec_value_2024 exactly and the four-symbol numbers
reproduce; the other 46 symbols then extend the test on the same 13 days.

Question: does "aggressive crossing is cheapest; the L2 signal timed rule is no
better than crossing; the oracle saves only ~1.4 bps" hold across the whole liquid
core, or do thinner names (wider spreads) give the L3-aware signal something to time?

Run: python scratch/book_exec_broad.py
"""
import os
import time
import importlib.util
import numpy as np
import pandas as pd

ROOT = r"C:\Users\jackw\Desktop\math-199-research"
spec = importlib.util.spec_from_file_location("ev", os.path.join(ROOT, "scratch", "exec_value_2024.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

uni = [s.strip() for s in open(os.path.join(ROOT, "data", "l2_universe_top50.txt")) if s.strip()]
MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]
SYMS = MAJORS + [s for s in uni if s not in MAJORS]   # majors first => exact reproduction

orders_csv = os.path.join(ROOT, "scratch", "exec_broad_orders.csv")
if os.path.exists(orders_csv):
    print("loading cached order-rows from", orders_csv)
    df = pd.read_csv(orders_csv)
else:
    rng = np.random.default_rng(ev.RNG_SEED)
    rows = []
    t0 = time.time()
    for sym in SYMS:
        for date in ev.DATES:
            rows += ev.simulate_day(sym, date, rng)
        print(f"  {sym:10s} done  rows={len(rows):>7,}  (elapsed {time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(orders_csv, index=False)
print(f"\ntotal order-rows {len(df):,} across {df['sym'].nunique()} symbols, {df['date'].nunique()} dates\n")

# per-symbol null metrics (block_metrics is deterministic for agg/naive/l3/oracle/corr;
# the placebo bootstrap uses the passed rng)
vrng = np.random.default_rng(7)
summ = []
for sym in SYMS:
    sub = df[df.sym == sym]
    if len(sub) < 30:
        continue
    m = ev.block_metrics(sub, vrng, sym)
    if m:
        summ.append(m)
out = pd.DataFrame(summ)
out.to_csv(os.path.join(ROOT, "scratch", "exec_broad_50sym.csv"), index=False)

print("\n" + "=" * 80)
print("POOLED across all 50 symbols")
ev.block_metrics(df, np.random.default_rng(7), "POOLED ALL 50")

# ---- faithfulness: 4 majors vs committed exec_value_2024_summary.csv ----
ref = pd.read_csv(os.path.join(ROOT, "scratch", "exec_value_2024_summary.csv"))
print("\nSANITY vs exec_value_2024_summary.csv (pooled both sizes):")
for s in MAJORS:
    r = ref[ref["sym"] == s]
    ref_agg = float((r["agg"] * r["n"]).sum() / r["n"].sum())
    ref_l3 = float((r["l3_30"] * r["n"]).sum() / r["n"].sum())
    mine = out[out["label"] == s].iloc[0]
    ok = "OK" if abs(ref_agg - mine["agg"]) < 0.02 and abs(ref_l3 - mine["l3"]) < 0.02 else "CHECK"
    print(f"  {s:8s} AGG ref={ref_agg:7.3f} got={mine['agg']:7.3f} | "
          f"L3 ref={ref_l3:7.3f} got={mine['l3']:7.3f}  {ok}")

# ---- headline: does the signal ever beat crossing? (bracket access: 'agg' collides
#      with DataFrame.agg / Series.agg under attribute access) ----
print("\nEXECUTION across 50 (bps; lower=cheaper). L3 = signal-timed; AGG = always cross.")
print(f"{'sym':>10} | {'AGG':>7} {'NAIVE':>7} {'L3':>7} | {'oracle_save':>11} {'corr_sig':>9}")
for _, r in out.sort_values("oracle_save_vs_agg", ascending=False).iterrows():
    print(f"{r['label']:>10} | {r['agg']:7.3f} {r['naive']:7.3f} {r['l3']:7.3f} | "
          f"{r['oracle_save_vs_agg']:>+11.3f} {r['corr_sig']:>+9.4f}")
helps = list(out[out["l3"] < out["agg"]]["label"])
print(f"\nsymbols where L3 (signal) is cheaper than just crossing: {helps if helps else 'NONE'}")
print(f"max oracle_saves_vs_agg over 50: {out['oracle_save_vs_agg'].max():.3f} bps "
      f"({out.loc[out['oracle_save_vs_agg'].idxmax(), 'label']})")
print("DONE.")
