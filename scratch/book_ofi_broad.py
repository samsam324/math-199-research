"""
Broaden the book-OFI decay null from 4 symbols to the FULL top-50 universe.

Reuses book_ofi_2024.py's EXACT CKS event formula, build, HAC regressions, and
run_block, so the 4 majors reproduce book_ofi_2024.log's ALL-regime numbers
byte-for-byte (sanity check printed at the end). Only change: loop over all 50
symbols from data/l2_universe_top50.txt, pooled "ALL" regime over the same 7
regime-spanning 2024 days, and write a compact per-symbol CSV of the decay curve.

Question this answers: does "order flow is priced within seconds, no tradable
forecast horizon survives" hold across the whole liquid core, or do thinner /
less efficient names leave a longer-lived (tradable) predictive horizon?

Run: python scratch/book_ofi_broad.py
"""
import os
import numpy as np
import pandas as pd
import book_ofi_2024 as bo   # same dir; reuse everything

ROOT = bo.ROOT
UNI = os.path.join(ROOT, 'data', 'l2_universe_top50.txt')
SYMS = [s.strip() for s in open(UNI) if s.strip()]
HZ = bo.HORIZONS   # [1, 5, 10, 30]

rows = []
for i, sym in enumerate(SYMS, 1):
    days = [bo.build_day(sym, d) for d in bo.DATES]
    keep = [(d, dt) for d, dt in zip(bo.DATES, days) if dt is not None]
    if not keep:
        print(f"[{i:2d}/50] {sym:10s} NO DATA")
        continue
    df = pd.concat([dt for _, dt in keep])
    df = bo._prep(df, 10)
    res = bo.run_block(df.copy(), f"{sym} ALL ({len(keep)} days)", 10)
    r = {'sym': sym, 'n_days': len(keep), 'n_sec': res['n'],
         'contemp_incr_book': res.get('contemp_incr_book', np.nan),
         'contemp_joint_r2': res.get('contemp_joint_r2', np.nan),
         'cancel_share': res.get('cancel_share', np.nan)}
    for h in HZ:
        p = res.get('pred', {}).get(h, {})
        r[f'pred_incr_book_h{h}'] = p.get('incr_book', np.nan)
        r[f'pred_joint_h{h}'] = p.get('joint_r2', np.nan)
        r[f't_book_h{h}'] = p.get('t_book', np.nan)
    rows.append(r)
    print(f"[{i:2d}/50] {sym:10s} done  n={res['n']:>7,}  "
          f"contemp_incr_book={r['contemp_incr_book']:.4f}  "
          f"pred_joint_h1={r['pred_joint_h1']:.5f}  pred_joint_h30={r['pred_joint_h30']:.5f}")

out = pd.DataFrame(rows)
csv = os.path.join(ROOT, 'scratch', 'ofi_broad_50sym.csv')
out.to_csv(csv, index=False)
print(f"\nwrote {csv}  ({len(out)} symbols)")

# ---- faithfulness check: 4 majors must match book_ofi_2024.log ALL regime ----
ref = {'BTCUSDT': 0.1548, 'ETHUSDT': 0.1137, 'SOLUSDT': 0.1993, 'AVAXUSDT': 0.4759}
print("\nSANITY vs book_ofi_2024.log (contemporaneous incr book-OFI, ALL):")
for s, v in ref.items():
    got = out.loc[out.sym == s, 'contemp_incr_book']
    got = float(got.iloc[0]) if len(got) else np.nan
    ok = "OK" if abs(got - v) < 5e-4 else "MISMATCH"
    print(f"  {s:10s} ref={v:.4f}  got={got:.4f}  {ok}")

# ---- headline: does any symbol keep meaningful predictive power past 10s? ----
print("\nPREDICTIVE joint R^2 decay (total forecast power, trade+book):")
print(f"{'sym':>10} | {'h=1':>8} {'h=5':>8} {'h=10':>8} {'h=30':>8} | {'max>10s':>8}")
for _, r in out.sort_values('pred_joint_h1', ascending=False).iterrows():
    late = max(r['pred_joint_h10'], r['pred_joint_h30'])
    print(f"{r['sym']:>10} | {r['pred_joint_h1']:8.5f} {r['pred_joint_h5']:8.5f} "
          f"{r['pred_joint_h10']:8.5f} {r['pred_joint_h30']:8.5f} | {late:8.5f}")
print("\nDONE.")
