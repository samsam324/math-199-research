"""
Best-practice gap (adversarial self-review): the no-stop result (~+2.5 monthly
Sharpe) was SELECTED from a search over risk-rule configs (the iter-12 frontier:
stop on/off/wide, z/time/conv exits, 10/20/40 pairs, vol-target). Iters 13/15
corrected for FREQUENCY (monthly) and SURVIVORSHIP, but never for the config
SELECTION. This applies the Bailey & Lopez de Prado deflated Sharpe (the same tool
the phase-1 RESULTS.md used) to ask: does the no-stop Sharpe survive a
multiple-configuration / selection correction?

Method: compute each config's MONTHLY net-return series (the frequency-honest unit
where holds ~1 month are ~independent); the selected config = 40-pair no-stop z-exit.
sr_var_across_trials = variance of per-month Sharpe across the configs searched.
deflated_sharpe(sr_observed, sr_var, n_trials, skew, kurt, T) with n_trials swept
over {actual frontier count, 25, 50} for robustness (phase-1 convention).

Run:  python scratch/nostop_dsr.py
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import wf_sharpe_freq as wsf
sys.path.insert(0, os.path.join(wb.ROOT, "phase2_l2"))
from src.statistics import deflated_sharpe  # noqa

CC = wb.COST_LEVELS["realistic_30bps_rt"]
base = dict(stop=np.inf, exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")
# the configs that were searched (the iter-12 frontier) -> (label, n_pairs, overrides)
CONFIGS = [
    ("10p nostop z", 10, {}), ("20p nostop z", 20, {}), ("40p nostop z", 40, {}),
    ("40p nostop z vol", 40, {"sizing": "vol", "vol_win": 720}),
    ("40p nostop conv", 40, {"exit_mode": "conv", "max_hold": 1000}),
    ("40p widestop6 z", 40, {"stop": 6.0}),
    ("40p widestop6 vol", 40, {"stop": 6.0, "sizing": "vol", "vol_win": 720}),
    ("20p nostop z vol", 20, {"sizing": "vol", "vol_win": 720}),
    ("40p tightstop4 z", 40, {"stop": 4.0}), ("10p tightstop4 z", 10, {"stop": 4.0}),
]
SELECTED = "40p nostop z"


def monthly_returns(px, cfg, npairs):
    s = wsf.net_series(px, CC, cfg, npairs)
    m = s.resample("ME").sum(); m = m[m != 0]
    return m.values


def main():
    px = wb.load_universe()
    sh_per_month, sel_m = [], None
    print("config monthly Sharpe (per-month mean/std; annualized in parens):")
    for label, npairs, ov in CONFIGS:
        cfg = dict(base); cfg.update(ov)
        m = monthly_returns(px, cfg, npairs)
        sh = m.mean() / m.std(ddof=1) if m.std() > 0 else np.nan   # per-month Sharpe
        sh_per_month.append(sh)
        print(f"  {label:<22} per-month SR={sh:+.3f}  (ann {sh*np.sqrt(12):+.2f})  n_months={len(m)}")
        if label == SELECTED:
            sel_m = m
    sh_per_month = np.array([x for x in sh_per_month if np.isfinite(x)])
    sr_var = float(np.var(sh_per_month, ddof=1))

    sr_obs = sel_m.mean() / sel_m.std(ddof=1)
    skew = float(sps.skew(sel_m)); kurt = float(sps.kurtosis(sel_m, fisher=False)); T = len(sel_m)
    print(f"\nSELECTED = {SELECTED}: per-month SR={sr_obs:+.3f} (ann {sr_obs*np.sqrt(12):+.2f}) "
          f"skew={skew:+.2f} kurt={kurt:.2f} T={T} months")
    print(f"variance of per-month SR across {len(sh_per_month)} searched configs = {sr_var:.4f}")
    print("\nDeflated Sharpe (Bailey & Lopez de Prado) vs n_trials:")
    for N in [len(CONFIGS), 25, 50]:
        sr0, dsr_p = deflated_sharpe(sr_obs, sr_var, N, skew, kurt, T)
        print(f"  N={N:>3} trials: sr_0(per-month)={sr0:+.3f} (ann {sr0*np.sqrt(12):+.2f})  "
              f"DSR=Pr(SR>0|obs, selection-corrected)={dsr_p:.4f}  "
              f"-> {'SURVIVES (DSR>0.95)' if dsr_p>0.95 else ('marginal' if dsr_p>0.5 else 'FAILS')}")
    print("\nReading: sr_0 = expected best per-month Sharpe under the null across N searched configs;")
    print("DSR>0.95 => the no-stop Sharpe beats selection-corrected chance at that trial count.")


if __name__ == "__main__":
    main()
