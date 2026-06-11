"""
Survivorship-adjusted monthly Sharpe at the DEPLOYABLE scale.

The referee's strongest objection: the survivor's ~1.0 monthly Sharpe is estimated on a
survivorship-filtered universe (current top-50; delisted coins absent), so is it real-but-marginal
or a survivorship artifact? scratch/wf_nostop_stress.py answered this on the GROSS (~2.5) hourly-HAC
scale. This re-asks it on the honest, deployable figure: the no-stop book WITH the structural-break
circuit breaker (halt after a >50% adverse leg move), aggregated to MONTHLY returns -- the same
~1.0-scale number the paper reports.

For each per-pair per-quarter break probability p, with probability p a selected pair suffers a
permanent delisting-scale structural break in the test window (wf_nostop_stress.inject_break,
~86% leg blowup, never reverts); selection is on the unbroken train. We report the monthly Sharpe
of the deployable no-stop+breaker book and, for contrast, the no-stop book without the breaker.

Run: python scratch/survivorship_adjusted_sharpe.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import wf_nostop_stress as wns   # inject_break
import nostop_breakstop as nbs   # sim_pair with the circuit breaker

np.seterr(all="ignore")

CC = wb.COST_LEVELS["realistic_30bps_rt"]
N_PAIRS = 40
BREAK_PROBS = [0.0, 0.02, 0.05, 0.10, 0.20]   # per selected pair, per 3-mo (quarter) test window
MAG = 2.0                                      # ~86% leg blowup in log-price (delisting-scale)
SEEDS = [0, 1, 2, 3, 4, 5]

_PICKS = {}   # (tr_s, tr_e) -> OU picks; identical across rate/seed/rule (selection is on the unbroken train)


def picks_for(logpx, cols, tr_s, tr_e):
    key = (tr_s, tr_e)
    if key not in _PICKS:
        tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
        avail = [c for c in cols if tr[c].notna().mean() >= wb.MIN_OBS_FRAC]
        _PICKS[key] = wb.select_pairs_ou(tr[avail], avail, N_PAIRS) if len(tr) >= 1000 else []
    return _PICKS[key]


def monthly_sharpe(parts):
    if not parts:
        return np.nan
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    m = s.resample("ME").sum()
    m = m[m != 0]
    return float(m.mean() / m.std(ddof=1) * np.sqrt(12)) if len(m) > 5 and m.std() > 0 else np.nan


def run_rate(logpx, cols, splits, p_break, seed, breaker):
    rng = np.random.default_rng(10_000 * seed + int(10_000 * p_break) + (7 if breaker else 0))
    kw = dict(adverse_K=0.50, halt=True) if breaker else {}
    parts = []
    for (tr_s, tr_e, te_e) in splits:
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(te) < 200:
            continue
        picks = picks_for(logpx, cols, tr_s, tr_e)
        if not picks:
            continue
        nmat = []
        for pk in picks:
            la = te[pk["a"]].values.astype(float)
            lb = te[pk["b"]].values.astype(float)
            if p_break > 0.0 and rng.random() < p_break:
                la, lb = wns.inject_break(la, lb, pk["alpha"], pk["beta"], pk["mu"], pk["sd"], rng, MAG)
            te_pair = pd.DataFrame({pk["a"]: la, pk["b"]: lb}, index=te.index)
            net = nbs.sim_pair(te_pair, pk["a"], pk["b"], pk["alpha"], pk["beta"], pk["mu"], pk["sd"],
                               CC["fee_bps"], CC["slip_bps"], **kw)
            if net is not None:
                nmat.append(pd.Series(net, index=te.index[:len(net)]))
        if nmat:
            L = min(len(v) for v in nmat)
            parts.append(pd.Series(np.mean([v.values[:L] for v in nmat], axis=0), index=te.index[:L]))
    return monthly_sharpe(parts)


def main():
    px = wb.load_universe()
    logpx = np.log(px); cols = list(px.columns); splits = wb.make_splits(px.index)
    print("Survivorship-adjusted monthly Sharpe (deployable no-stop + circuit breaker)")
    print(f"universe={px.shape[1]} syms (current top-50) | {len(splits)} splits | {N_PAIRS} pairs | "
          f"break mag ~86% leg | seeds={len(SEEDS)}")
    print(f"\n{'break p/qtr':>12}{'~annual':>9}{'  monthlyS no-breaker':>22}{'  monthlyS +breaker':>20}")
    print("-" * 64)
    rows = []
    for p in BREAK_PROBS:
        seeds = [0] if p == 0.0 else SEEDS
        nb = np.nanmean([run_rate(logpx, cols, splits, p, sd, breaker=False) for sd in seeds])
        br = np.nanmean([run_rate(logpx, cols, splits, p, sd, breaker=True) for sd in seeds])
        ann = 1.0 - (1.0 - p) ** 4
        rows.append(dict(break_p_per_qtr=p, approx_annual=ann, monthlyS_nobreaker=nb, monthlyS_breaker=br))
        print(f"{p*100:>11.0f}%{ann*100:>8.0f}%{nb:>22.2f}{br:>20.2f}")
    pd.DataFrame(rows).to_csv(os.path.join(wb.ROOT, "scratch", "survivorship_adjusted_sharpe.csv"), index=False)
    print("\nsaved -> scratch/survivorship_adjusted_sharpe.csv")
    print("Reading: the deployable (monthly, breaker) Sharpe degrades with delisting attrition; the rate")
    print("at which it reaches ~0 is the survivorship breakeven, to be compared with realized delisting")
    print("frequencies for liquid top-50 names (low single-digit %/yr).")


if __name__ == "__main__":
    main()
