"""
Sensitivity sweep on the structural-break circuit-breaker threshold K.

Paper claims "halt a pair when one leg moves more than 50% adversely" preserves ~96% of
monthly Sharpe at 8%/year breakage rate. A careful reviewer asks "why 50%?" — this script
runs the same machinery as scratch/nostop_breakstop.py across a wider grid of K and reports
monthly Sharpe + max DD at each threshold so we can state sensitivity honestly in the paper.

Output: scratch/threshold_sweep.csv
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Override the Windows-hardcoded ROOT before importing modules that bake it in
import wf_backtest as wb
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wb.ROOT = REPO
wb.UNIV = os.path.join(REPO, "data", "l2_universe_top50.txt")
import nostop_breakstop as nbs
nbs.DELI_DIR = os.path.join(REPO, "data", "spot_1h_delisted")

THRESHOLDS = [0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def main():
    px = nbs.load_top50_plus_delisted()
    print(f"universe = {px.shape[1]} symbols (top-50 + 4 delisted)")
    print(f"\n{'K (%)':>8}{'monthlyS':>12}{'maxDD%':>10}{'worst pw%':>14}{'retain':>10}")
    print("-" * 60)
    # baseline (no breaker)
    sh0, dd0, w0 = nbs.run(px)
    print(f"{'none':>8}{sh0:>12.3f}{dd0*100:>10.1f}{w0*100:>14.0f}{1.00:>10.3f}")
    rows = [{"K": None, "monthlyS": sh0, "maxDD_pct": dd0*100, "worst_pw_pct": w0*100, "retain": 1.0}]
    for K in THRESHOLDS:
        sh, dd, w = nbs.run(px, adverse_K=K, halt=True)
        retain = sh / sh0 if sh0 not in (0, np.nan) else np.nan
        print(f"{K*100:>8.0f}{sh:>12.3f}{dd*100:>10.1f}{w*100:>14.0f}{retain:>10.3f}")
        rows.append({"K": K, "monthlyS": sh, "maxDD_pct": dd*100, "worst_pw_pct": w*100, "retain": retain})
    out = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold_sweep.csv")
    out.to_csv(out_path, index=False)
    print(f"\nsaved -> {out_path}")
    return out


if __name__ == "__main__":
    main()
