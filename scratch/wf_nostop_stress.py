"""
Is the no-stop result real, or a SURVIVORSHIP artifact?

Iteration-8 robustness (scratch/wf_robustness.py) found that removing the |z|>=4
stop flips the pair strategy from net Sharpe -2.25 to +2.51 (gross -1.15 -> +2.65).
That overturns the iter-7 "untradeable / loses even gross" headline -- BUT the
no-stop rule means "hold every diverging spread until it reverts", which is only
safe if every divergence DOES eventually revert. The backtest universe is
data/l2_universe_top50.txt == the CURRENT top-50 liquid symbols, so it is
survivorship-filtered: every coin in it survived to 2026. Coins that permanently
decoupled / delisted (LUNA, FTT, ...) are excluded by construction -- and those
are exactly the tail the stop exists to protect against. Result 1 (no genuine
cointegration) says we CANNOT certify ex ante which spreads are truly stationary,
so we cannot tell a temporarily-diverged (will-revert) spread from a permanently
broken one.

This script demonstrates the survivorship dependence two ways:

(1) TAIL DIAGNOSTICS of the no-stop p=0 run: max drawdown, the worst per-pair-
    window PnL, holding-time distribution, and the fraction of positions still
    open (never converged) at window end. A benign tail == the survivorship
    signature (the real-universe tail has been removed).

(2) STRUCTURAL-BREAK STRESS TEST. Re-introduce the tail survivorship removed:
    in each TEST window, with probability p per selected pair, the pair suffers a
    permanent structural break (delisting / depeg / re-rating) -- at a random time
    the spread diverges monotonically AWAY from its mean by BREAK_MAG in log-price
    and never reverts. Selection is on the (unbroken) TRAIN window, exactly as in
    live trading: you pick on history, then a coin breaks in the future. Recompute
    NET Sharpe for the NO-STOP rule and the |z|=4-STOP rule under the SAME break
    draws, for p in {0, 1%, 2%, 5%, 10%}. The stop caps each break loss at ~|z|=4;
    the no-stop rule eats the full BREAK_MAG. If the no-stop edge collapses at a
    realistic crypto attrition rate while the stopped rule is robust, the no-stop
    "edge" is a survivorship artifact and the stop's value is tail protection.

Run synchronously:  python scratch/wf_nostop_stress.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import wf_robustness as wr

np.seterr(all="ignore")

N_PAIRS = wb.N_PAIRS
Z_ENTER = wb.Z_ENTER
COST_LEVELS = wb.COST_LEVELS
HOURS_PER_YEAR = wb.HOURS_PER_YEAR

BREAK_PROBS = [0.0, 0.02, 0.05, 0.10, 0.20]    # per selected pair, per 3-mo test window
BREAK_MAGS  = [0.50, 1.00, 2.00]               # adverse log move: 50%, ~63%, ~86% leg blowup (delisting-scale)
MC_SEEDS    = [0, 1, 2, 3, 4, 5, 6, 7]         # Monte-Carlo over which pairs break & when
RAMP_FRAC   = 1.0                               # break ramps over the remainder of the window


# ---------------------------------------------------------------------------
# Inject a permanent structural break into a single test leg-price pair.
# Push the spread further from its mean (adverse to ANY mean-reversion position)
# starting at a random time, ramping to +BREAK_MAG in log-price by window end.
# Operates on copies; returns broken (la, lb).
# ---------------------------------------------------------------------------
def inject_break(la, lb, alpha, beta, mu, sd, rng, mag):
    la = la.copy(); lb = lb.copy()
    n = len(la)
    if n < 40 or not np.isfinite(sd) or sd <= 0:
        return la, lb
    spread = la - alpha - beta * lb
    # break time: somewhere in the first 70% so the divergence has room to run
    t_b = int(rng.integers(int(0.10 * n), int(0.70 * n) + 1))
    s0 = np.sign(spread[t_b] - mu)
    if s0 == 0:
        s0 = 1.0
    remain = n - t_b
    ramp_len = max(2, int(RAMP_FRAC * remain))
    # linear ramp in log-price added to leg A, in the direction that worsens |z|
    add = np.zeros(n)
    ramp = np.linspace(0.0, mag, ramp_len)
    end = min(n, t_b + ramp_len)
    add[t_b:end] = s0 * ramp[: end - t_b]
    if end < n:
        add[end:] = s0 * mag
    la = la + add
    return la, lb


# ---------------------------------------------------------------------------
# Run S1(stop=4) and S2(no-stop) over the walk-forward, optionally injecting
# breaks into the TEST window for a fraction p of selected pairs. Returns the
# concatenated hourly NET pnl for each rule (at the given cost level).
# Also returns per-pair-window diagnostics when collect_diag=True (p=0 only).
# ---------------------------------------------------------------------------
_PICK_CACHE = {}  # split-key -> OU picks (identical across break/seed/cost)


def _picks_for_split(logpx, cols, tr_s, tr_e):
    key = (tr_s, tr_e)
    if key not in _PICK_CACHE:
        tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
        _PICK_CACHE[key] = wb.select_pairs_ou(tr, cols, N_PAIRS) if len(tr) >= 1000 else []
    return _PICK_CACHE[key]


def run_pair_rules(px, cost_cfg, p_break, mag, seed, collect_diag=False):
    rng = np.random.default_rng(10_000 * seed + int(1000 * mag) + int(10_000 * p_break))
    logpx = np.log(px)
    cols = list(px.columns)
    splits = wb.make_splits(px.index)
    fee = cost_cfg["fee_bps"]; slip = cost_cfg["slip_bps"]
    cfg_stop   = dict(stop=4.0,    exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")
    cfg_nostop = dict(stop=np.inf, exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")
    net_stop = []; net_nostop = []
    diag = []  # per-pair-window dicts for the no-stop rule

    for (tr_s, tr_e, te_e) in splits:
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(te) < 200:
            continue
        picks = _picks_for_split(logpx, cols, tr_s, tr_e)
        if not picks:
            continue
        s_mat = []; n_mat = []
        for p in picks:
            la = te[p["a"]].values.astype(float)
            lb = te[p["b"]].values.astype(float)
            broke = (p_break > 0.0) and (rng.random() < p_break)
            if broke:
                la, lb = inject_break(la, lb, p["alpha"], p["beta"], p["mu"], p["sd"], rng, mag)
            te_pair = pd.DataFrame({p["a"]: la, p["b"]: lb})
            rs = wr.simulate_pair_v(te_pair, p["a"], p["b"], p["alpha"], p["beta"],
                                    p["mu"], p["sd"], fee, slip, cfg_stop)
            rn = wr.simulate_pair_v(te_pair, p["a"], p["b"], p["alpha"], p["beta"],
                                    p["mu"], p["sd"], fee, slip, cfg_nostop)
            if rs is None or rn is None:
                continue
            s_mat.append(rs["net"]); n_mat.append(rn["net"])
            if collect_diag and p_break == 0.0:
                pnl = rn["net"]; pos = rn["pos"]
                # holding segments
                seg_lens = []; seg = 0; open_at_end = False
                for t in range(len(pos)):
                    if pos[t] != 0:
                        seg += 1
                        if t == len(pos) - 1:
                            open_at_end = True; seg_lens.append(seg)
                    elif seg > 0:
                        seg_lens.append(seg); seg = 0
                eq = np.cumsum(np.nan_to_num(pnl))
                dd = float((eq - np.maximum.accumulate(eq)).min())
                diag.append(dict(pair=f"{p['a']}_{p['b']}", pnl=float(np.nansum(pnl)),
                                 mdd=dd, max_hold=(max(seg_lens) if seg_lens else 0),
                                 open_at_end=open_at_end, n_bars=len(pos)))
        if not s_mat:
            continue
        L = min(min(len(v) for v in s_mat), min(len(v) for v in n_mat))
        net_stop.append(np.mean([v[:L] for v in s_mat], axis=0))
        net_nostop.append(np.mean([v[:L] for v in n_mat], axis=0))

    if not net_stop:
        return None
    return (np.concatenate(net_stop), np.concatenate(net_nostop), diag)


def sharpe(x):
    s, _, _ = wb.sharpe_hac(x)
    return s


def main():
    t0 = time.time()
    px = wb.load_universe()
    splits = wb.make_splits(px.index)
    print("=" * 100)
    print("NO-STOP SURVIVORSHIP / STRUCTURAL-BREAK STRESS TEST")
    print(f"universe={px.shape[1]} syms (CURRENT top-50 == survivorship-filtered) | "
          f"span={px.index.min().date()}..{px.index.max().date()} | {len(splits)} splits")
    print("=" * 100)

    # -------- (1) tail diagnostics of the clean (p=0) no-stop run --------
    cc = COST_LEVELS["realistic_30bps_rt"]
    base = run_pair_rules(px, cc, 0.0, 0.0, seed=0, collect_diag=True)
    net_stop0, net_nostop0, diag = base
    print("\n[1] TAIL DIAGNOSTICS  (no-stop, no breaks, realistic 30bps)")
    print(f"    no-stop  net Sharpe = {sharpe(net_nostop0):+.2f}   "
          f"stop     net Sharpe = {sharpe(net_stop0):+.2f}")
    d = pd.DataFrame(diag)
    if not d.empty:
        eq = np.cumsum(np.nan_to_num(net_nostop0))
        port_mdd = float((eq - np.maximum.accumulate(eq)).min())
        print(f"    per-(pair,window) net PnL: n={len(d)}  mean={d.pnl.mean()*100:+.2f}%  "
              f"min={d.pnl.min()*100:+.2f}%  p05={d.pnl.quantile(0.05)*100:+.2f}%  "
              f"max={d.pnl.max()*100:+.2f}%  skew={d.pnl.skew():+.2f}")
        print(f"    worst single pair-window loss = {d.pnl.min()*100:+.2f}%   "
              f"(# losing pair-windows = {(d.pnl<0).sum()}/{len(d)})")
        print(f"    PORTFOLIO max drawdown (no-stop) = {port_mdd*100:+.2f}%")
        print(f"    holding time (bars): median={d.max_hold.median():.0f}  "
              f"p95={d.max_hold.quantile(0.95):.0f}  max={d.max_hold.max():.0f}  "
              f"(window len ~{int(d.n_bars.median())}h)")
        print(f"    positions STILL OPEN at window end (never converged) = "
              f"{d.open_at_end.sum()}/{len(d)} ({100*d.open_at_end.mean():.1f}%)")
        print("    >> A benign worst-case tail here is the SURVIVORSHIP signature: the")
        print("       real-universe non-reverting tail (delistings) is absent by construction.")

    # -------- (2) structural-break stress test --------
    print("\n[2] STRUCTURAL-BREAK STRESS TEST  (NET Sharpe vs per-pair break prob p)")
    print("    Each break = a selected pair permanently diverges by BREAK_MAG in log-price")
    print("    in the TEST window and never reverts (selection is on the unbroken TRAIN).")
    for clabel, cc in COST_LEVELS.items():
        print(f"\n    ----- cost: {clabel} -----")
        for mag in BREAK_MAGS:
            print(f"      BREAK_MAG = {mag*100:.0f}% leg blowup")
            header = "        p_break ->" + "".join(f"{p*100:>8.0f}%" for p in BREAK_PROBS)
            row_ns = "        no-stop net S:"
            row_st = "        stop|z|=4   :"
            for p in BREAK_PROBS:
                ns_list, st_list = [], []
                seeds = [0] if p == 0.0 else MC_SEEDS
                for sd in seeds:
                    out = run_pair_rules(px, cc, p, mag, seed=sd)
                    if out is None:
                        continue
                    st, ns, _ = out
                    st_list.append(sharpe(st)); ns_list.append(sharpe(ns))
                row_ns += f"{np.mean(ns_list):>+9.2f}"
                row_st += f"{np.mean(st_list):>+9.2f}"
            print(header)
            print(row_ns)
            print(row_st)
    print("\n" + "=" * 100)
    print(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
