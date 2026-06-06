"""
Iteration-11 characterization (cont'd): NAME the no-stop exposure.

Iter-10 showed the no-stop edge is market-neutral (beta~0 to PC1/market). This asks
the finer question: is the +2.5 a CONCENTRATED bet on one slow statistical factor
(an eigenportfolio), or DIVERSIFIED idiosyncratic pair reversion (many ~independent
bets)? The latter is far more robust and is the better scientific characterization.

Method: PCA (eigendecomposition) of the universe's standardized hourly returns over
the well-covered core symbols -> eigenportfolio return series (PC scores). Regress the
no-stop portfolio's gross hourly return on PC1..PC10. Low total R2 => the no-stop alpha
is idiosyncratic/diversified (good). High R2 concentrated on a couple PCs => a factor bet.

Run synchronously:  python scratch/wf_nostop_pca.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import wf_robustness as wr
import wf_nostop_factor as fac

np.seterr(all="ignore")
N_PC = 10


def main():
    px = wb.load_universe()
    cc = wb.COST_LEVELS["realistic_30bps_rt"]
    cfg_nostop = dict(stop=np.inf, exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")

    # ---- universe hourly log-returns; keep well-covered core symbols ----
    logret = np.log(px).diff()
    cov_frac = logret.notna().mean()
    core = cov_frac[cov_frac > 0.95].index.tolist()
    R = logret[core].dropna(how="any")
    print("=" * 96)
    print("ITER-11 PCA: is the no-stop edge a concentrated factor bet or diversified reversion?")
    print(f"  core symbols={len(core)}  clean hourly rows={len(R)}  "
          f"span={R.index.min().date()}..{R.index.max().date()}")
    print("=" * 96)

    # standardize columns, eigendecompose covariance of standardized returns (=correlation)
    Z = (R - R.mean()) / R.std()
    C = np.cov(Z.values, rowvar=False)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]; evecs = evecs[:, order]
    var_expl = evals / evals.sum()
    print("\nEigenvalue spectrum (variance share):")
    print("  " + "  ".join(f"PC{k+1}={var_expl[k]*100:4.1f}%" for k in range(N_PC)))
    print(f"  PC1 (market) share = {var_expl[0]*100:.1f}%   top-10 cumulative = {var_expl[:N_PC].sum()*100:.1f}%")

    # PC score (eigenportfolio return) time series
    scores = pd.DataFrame(Z.values @ evecs[:, :N_PC], index=Z.index,
                          columns=[f"PC{k+1}" for k in range(N_PC)])

    # ---- no-stop gross portfolio return series ----
    g_ns, _ = fac.portfolio_series(px, cfg_nostop, cc)
    common = g_ns.index.intersection(scores.index)
    y = g_ns.reindex(common).values
    X = scores.reindex(common).values
    y = y - y.mean()
    Xc = X - X.mean(axis=0)

    # OLS regression of no-stop return on PC1..PC10
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(y), Xc]), y, rcond=None)
    yhat = np.column_stack([np.ones_like(y), Xc]) @ beta
    ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum(y ** 2)
    r2_all = 1 - ss_res / ss_tot

    # per-PC univariate R2 (how much each single factor explains)
    print(f"\nNo-stop return regressed on PC1..PC{N_PC} (n={len(y)}):")
    print(f"  TOTAL R2 (all {N_PC} PCs) = {r2_all*100:.1f}%")
    print("  per-PC univariate R2 (variance of no-stop return explained by each factor alone):")
    uni = []
    for k in range(N_PC):
        xk = Xc[:, k]
        b = np.dot(xk, y) / np.dot(xk, xk)
        r2 = (b ** 2 * np.dot(xk, xk)) / ss_tot
        uni.append(r2)
        print(f"    PC{k+1:<2} R2={r2*100:5.2f}%   (factor var share {var_expl[k]*100:4.1f}%)")
    print(f"  max single-PC R2 = {max(uni)*100:.2f}%  (PC{int(np.argmax(uni))+1})")

    print("\nINTERPRETATION:")
    if r2_all < 0.20:
        print(f"  Total factor R2 {r2_all*100:.0f}% is LOW => the no-stop alpha is largely IDIOSYNCRATIC /")
        print( "  diversified pair reversion (many ~independent bets), not a single slow-factor bet.")
        print( "  This is the robust outcome: the edge is not one crowded basket.")
    else:
        print(f"  Total factor R2 {r2_all*100:.0f}% is non-trivial; the no-stop return loads on")
        print( "  statistical factor(s) -- check which PC dominates above.")
    print("=" * 96)


if __name__ == "__main__":
    main()
