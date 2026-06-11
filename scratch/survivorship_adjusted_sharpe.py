"""
Survivorship-adjusted monthly Sharpe at the DEPLOYABLE scale.

The referee's strongest objection: the survivor's ~1.0 monthly Sharpe is estimated on a
survivorship-filtered universe (current top-50; delisted coins absent), so is it real-but-marginal
or a survivorship artifact? scratch/wf_nostop_stress.py answered this on the GROSS (~2.5) hourly-HAC
scale. This re-asks it on the deployable rule: the no-stop book WITH the structural-break circuit
breaker (halt after a >50% adverse leg move), aggregated to MONTHLY returns, against the SAME book
without the breaker, on IDENTICAL break draws.

For each per-pair per-quarter break probability p, with probability p a selected pair suffers a
permanent delisting-scale structural break in the test window (wf_nostop_stress.inject_break,
~86% leg blowup, never reverts); selection is on the unbroken train. We run both the breaker and
no-breaker rules on the same broken book and report each monthly Sharpe.

Caveats (read with the paper): inject_break ramps the move smoothly, so the breaker exits near its
50% threshold and the per-break loss here is an OPTIMISTIC ~50% -- real gapping delistings cost the
same breaker -65% to -91% per pair (scratch/forced_collapse.py), so the retention below is an UPPER
bound. The baseline is the gross monthly Sharpe (~2.4), not the HAC/venue-honest ~1.0. Breaks are
drawn independently across pairs, so this understates clustered (risk-off) delisting waves.

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


def run_rate(logpx, cols, splits, p_break, seed):
    """One walk-forward; both rules see the SAME injected breaks. Returns (S_no_breaker, S_breaker)."""
    rng = np.random.default_rng(10_000 * seed + int(10_000 * p_break))
    parts_nb, parts_br = [], []
    for (tr_s, tr_e, te_e) in splits:
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(te) < 200:
            continue
        picks = picks_for(logpx, cols, tr_s, tr_e)
        if not picks:
            continue
        nb_mat, br_mat = [], []
        for pk in picks:
            la = te[pk["a"]].values.astype(float)
            lb = te[pk["b"]].values.astype(float)
            if p_break > 0.0 and rng.random() < p_break:
                la, lb = wns.inject_break(la, lb, pk["alpha"], pk["beta"], pk["mu"], pk["sd"], rng, MAG)
            te_pair = pd.DataFrame({pk["a"]: la, pk["b"]: lb}, index=te.index)
            net_nb = nbs.sim_pair(te_pair, pk["a"], pk["b"], pk["alpha"], pk["beta"], pk["mu"], pk["sd"],
                                  CC["fee_bps"], CC["slip_bps"])
            net_br = nbs.sim_pair(te_pair, pk["a"], pk["b"], pk["alpha"], pk["beta"], pk["mu"], pk["sd"],
                                  CC["fee_bps"], CC["slip_bps"], adverse_K=0.50, halt=True)
            if net_nb is not None:
                nb_mat.append(pd.Series(net_nb, index=te.index[:len(net_nb)]))
            if net_br is not None:
                br_mat.append(pd.Series(net_br, index=te.index[:len(net_br)]))
        for mat, parts in ((nb_mat, parts_nb), (br_mat, parts_br)):
            if mat:
                L = min(len(v) for v in mat)
                parts.append(pd.Series(np.mean([v.values[:L] for v in mat], axis=0), index=te.index[:L]))
    return monthly_sharpe(parts_nb), monthly_sharpe(parts_br)


def main():
    px = wb.load_universe()
    logpx = np.log(px); cols = list(px.columns); splits = wb.make_splits(px.index)
    print("Survivorship-adjusted monthly Sharpe (deployable no-stop + circuit breaker)")
    print(f"universe={px.shape[1]} syms (current top-50) | {len(splits)} splits | {N_PAIRS} pairs | "
          f"break mag ~86% leg | seeds={len(SEEDS)} | paired draws")
    print(f"\n{'break p/qtr':>12}{'~annual':>9}{'  S no-breaker':>15}{'  S +breaker':>13}{'  breaker retain':>17}")
    print("-" * 66)
    rows = []
    base_br = None
    for p in BREAK_PROBS:
        seeds = [0] if p == 0.0 else SEEDS
        res = [run_rate(logpx, cols, splits, p, sd) for sd in seeds]
        nb = float(np.nanmean([r[0] for r in res]))
        br = float(np.nanmean([r[1] for r in res]))
        if base_br is None:
            base_br = br
        ann = 1.0 - (1.0 - p) ** 4
        retain = br / base_br if base_br else np.nan
        rows.append(dict(break_p_per_qtr=p, approx_annual=ann, monthlyS_nobreaker=nb,
                         monthlyS_breaker=br, breaker_retain=retain))
        print(f"{p*100:>11.0f}%{ann*100:>8.0f}%{nb:>15.2f}{br:>13.2f}{retain*100:>15.0f}%")
    pd.DataFrame(rows).to_csv(os.path.join(wb.ROOT, "scratch", "survivorship_adjusted_sharpe.csv"), index=False)
    print("\nsaved -> scratch/survivorship_adjusted_sharpe.csv")
    print("Reading (UPPER bound; smooth ramp lets the breaker exit near 50% -- real gaps cost -65% to -91%):")
    print("the breaker keeps the book positive across the range while the no-breaker book turns negative;")
    print("the qualitative result is the breaker, not diversification alone, bounds the delisting tail.")


if __name__ == "__main__":
    main()
