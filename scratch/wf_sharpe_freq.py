"""
Iteration-13 self-correction: the iter-12 frontier Sharpes (+3.2/+3.4) are HOURLY
annualized Sharpes on a strategy with ~35-day median holds, so they are inflated by
hold autocorrelation (HAC only partly corrects it). The honest deployable Sharpe must
be computed at a frequency where the autocorrelation has washed out.

Computes the annualized Sharpe of the no-stop portfolio at hourly / daily / weekly /
monthly frequency for the key configs, plus a monthly block-bootstrap CI. The Sharpe
should DECLINE from hourly toward a stable low-frequency value = the honest number.
(Test windows tile the timeline with step==test==3mo, so the series is continuous and
resampling is clean.)

Run synchronously:  python scratch/wf_sharpe_freq.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import wf_robustness as wr

np.seterr(all="ignore")


def net_series(px, cost_cfg, cfg, n_pairs):
    """Timestamped concatenated hourly NET portfolio return series."""
    logpx = np.log(px); cols = list(px.columns); splits = wb.make_splits(px.index)
    fee = cost_cfg["fee_bps"]; slip = cost_cfg["slip_bps"]
    parts = []
    for (tr_s, tr_e, te_e) in splits:
        tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(tr) < 1000 or len(te) < 200:
            continue
        picks = wb.select_pairs_ou(tr, cols, n_pairs)
        if not picks:
            continue
        nmat = []
        for p in picks:
            hl = p.get("hl", np.nan); pcfg = dict(cfg)
            if cfg.get("exit_mode") == "time":
                pcfg["max_hold"] = max(2, int(round(2.0 * hl)) if np.isfinite(hl) and hl > 0 else 48)
            r = wr.simulate_pair_v(te, p["a"], p["b"], p["alpha"], p["beta"],
                                   p["mu"], p["sd"], fee, slip, pcfg)
            if r is not None:
                nmat.append(r["net"])
        if not nmat:
            continue
        L = min(len(v) for v in nmat)
        parts.append(pd.Series(np.mean([v[:L] for v in nmat], axis=0), index=te.index[:L]))
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="first")]


def ann_sharpe_at(series, freq, ppy):
    r = series.resample(freq).sum()
    r = r[r != 0.0]            # drop any inactive period
    if len(r) < 5 or r.std() == 0:
        return np.nan, len(r)
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ppy)), len(r)


def monthly_boot_ci(series, nboot=2000, seed=0):
    r = series.resample("ME").sum(); r = r[r != 0.0].values
    n = len(r); rng = np.random.default_rng(seed)
    if n < 8:
        return (np.nan, np.nan)
    sh = np.empty(nboot)
    for b in range(nboot):
        s = r[rng.integers(0, n, n)]
        sh[b] = s.mean() / s.std(ddof=1) * np.sqrt(12) if s.std() > 0 else 0.0
    return (np.nanpercentile(sh, 2.5), np.nanpercentile(sh, 97.5))


def main():
    px = wb.load_universe()
    cc = wb.COST_LEVELS["realistic_30bps_rt"]
    base = dict(stop=np.inf, exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")
    configs = [
        ("10p no-stop z", 10, {}),
        ("40p no-stop z", 40, {}),
        ("40p no-stop conv", 40, {"exit_mode": "conv", "max_hold": 1000}),
    ]
    print("=" * 92)
    print("ITER-13: honest deployable Sharpe -- annualized Sharpe by sampling frequency")
    print("  (multi-week holds => HOURLY Sharpe is autocorrelation-inflated; low-freq = honest)")
    print("=" * 92)
    print(f"{'config':<20}{'hourly':>9}{'daily':>9}{'weekly':>9}{'monthly':>10}{'monthly 95% CI':>22}")
    print("-" * 92)
    for label, npairs, ov in configs:
        cfg = dict(base); cfg.update(ov)
        s = net_series(px, cc, cfg, npairs)
        sh_h, _ = ann_sharpe_at(s, "h", 24 * 365)
        sh_d, _ = ann_sharpe_at(s, "D", 365)
        sh_w, _ = ann_sharpe_at(s, "W", 52)
        sh_m, nm = ann_sharpe_at(s, "ME", 12)
        ci = monthly_boot_ci(s)
        print(f"{label:<20}{sh_h:>9.2f}{sh_d:>9.2f}{sh_w:>9.2f}{sh_m:>10.2f}"
              f"   [{ci[0]:+.2f},{ci[1]:+.2f}] (n_mo={nm})")
    print("-" * 92)
    print("READING: the honest deployable Sharpe is the low-frequency (weekly/monthly) value;")
    print("the hourly figure overstates it because month-long holds make hourly returns autocorrelated.")
    print("=" * 92)


if __name__ == "__main__":
    main()
