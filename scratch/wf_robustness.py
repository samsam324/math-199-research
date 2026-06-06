"""
Robustness matrix for the pair mean-reversion strategy.

Reuses the EXACT data loader, OU/random selection, walk-forward splits, cost
conventions, and Sharpe/HAC/bootstrap stats from scratch/wf_backtest.py, then
sweeps a matrix of honest strategy variants over the SAME protocol:

  Axis 1  STOP   : (a) baseline |z|>=4   (b) NO stop (+inf)   (c) wider |z|>=6
  Axis 2  EXIT   : (a) z-exit |z|<=0.5   (b) TIME exit at 2*HL bars
                   (c) hold-to-convergence (z sign flip), cap 1000h, no stop
  Axis 3  HEDGE  : (a) static TRAIN beta (baseline)
                   (b) ROLLING causal OLS hedge (30d=720h and 60d=1440h)
  Axis 4  SIZING : (a) unit position   (b) volatility-targeted (inverse trailing
                   spread vol so pairs contribute equal risk)

For EVERY variant: GROSS Sharpe, NET Sharpe at both cost levels, net total PnL %,
#trades, turnover, win rate, HAC SE + block-bootstrap 95% CI on net Sharpe, and a
RANDOM-PAIR placebo (avg over seeds) so any positive number is checked vs placebo.

The decisive cell: NO stop + rolling hedge + time/convergence exit at LOW (2bps)
cost == most-favorable honest implementation.

CAUTION (known artifact): any variant that recomputes z from a trailing window
mechanically reverts even on noise; the prior work found random pairs earned
+2-3 net Sharpe and phase-randomized noise +2.4 under rolling-z. So a rolling-z
"positive" is only an edge if it clearly EXCEEDS the random-pair placebo by more
than the across-seed SD. Static-hedge variants here keep the static TRAIN z
(no trailing-window z artifact); rolling-hedge variants necessarily use a
trailing z, so their placebo is mandatory.

Run synchronously:  python scratch/wf_robustness.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb  # reuse everything

np.seterr(all="ignore")

# Inherit config from the baseline module so the protocol is identical.
N_PAIRS       = wb.N_PAIRS
Z_ENTER       = wb.Z_ENTER          # 2.0
Z_EXIT_BASE   = wb.Z_EXIT           # 0.5
COST_LEVELS   = wb.COST_LEVELS
HOURS_PER_YEAR = wb.HOURS_PER_YEAR

# Keep runtime reasonable: 3 seeds for placebos (task allows reducing to 3).
RANDOM_SEEDS  = [0, 1, 2]

# Rolling-hedge window settings to test
ROLL_HEDGE_WINS = [720, 1440]       # 30d, 60d (hours)
ROLL_HEDGE_MINP = 200               # min_periods for rolling OLS
CONV_MAX_HOLD   = 1000              # cap for hold-to-convergence (hours)


# ----------------------------------------------------------------------------
# Parameterized single-pair simulator.
#
# cfg keys:
#   stop        : float, |z|>=stop -> flat (use np.inf for "no stop")
#   exit_mode   : "z" | "time" | "conv"
#   z_exit      : float (used when exit_mode=="z")
#   max_hold    : int bars (used for "time" and as cap for "conv")
#   hedge       : "static" | "roll"
#   roll_win    : int hours (rolling OLS + rolling z window) when hedge=="roll"
#   sizing      : "unit" | "vol"
#   vol_win     : int hours, trailing spread-vol window for vol-targeting
#
# Returns dict(gross, net, pos, n_trades, turnover, spread_ret) compatible with
# wf_backtest.run_strategy's expectations, or None if the pair is unusable.
# ----------------------------------------------------------------------------
def simulate_pair_v(logpx_test, sym_a, sym_b, alpha, beta, mu, sd,
                    fee_bps, slip_bps, cfg, hl_bars=None):
    la = logpx_test[sym_a].values.astype(float)
    lb = logpx_test[sym_b].values.astype(float)
    n = len(la)
    if n < 30:
        return None

    hedge = cfg.get("hedge", "static")
    roll_win = int(cfg.get("roll_win", 720))

    if hedge == "static":
        if sd is None or sd <= 0 or not np.isfinite(sd):
            return None
        spread = la - alpha - beta * lb
        z = (spread - mu) / sd
    else:
        # ROLLING causal OLS hedge: at each bar t, fit y=a+b*x on the trailing
        # window [t-roll_win, t-1] (strictly past), then z from a trailing
        # mean/std of the *rolling-hedge* spread over the same window.
        sa = pd.Series(la); sb = pd.Series(lb)
        minp = ROLL_HEDGE_MINP
        # rolling OLS via rolling moments (cov/var) -> beta, alpha
        mx  = sb.rolling(roll_win, min_periods=minp).mean()
        my  = sa.rolling(roll_win, min_periods=minp).mean()
        mxx = (sb * sb).rolling(roll_win, min_periods=minp).mean()
        mxy = (sa * sb).rolling(roll_win, min_periods=minp).mean()
        varx = mxx - mx * mx
        covxy = mxy - mx * my
        beta_r = covxy / varx
        alpha_r = my - beta_r * mx
        # SHIFT by 1 so the hedge at bar t uses only data through t-1 (causal)
        beta_r = beta_r.shift(1)
        alpha_r = alpha_r.shift(1)
        spread = (sa - alpha_r - beta_r * sb)
        # trailing mean/std of the rolling-hedge spread (also causal: shift 1)
        rmu = spread.rolling(roll_win, min_periods=minp).mean().shift(1)
        rsd = spread.rolling(roll_win, min_periods=minp).std().shift(1)
        z = ((spread - rmu) / rsd).values
        spread = spread.values

    if not np.any(np.isfinite(z)):
        return None

    stop     = float(cfg.get("stop", 4.0))
    exit_mode = cfg.get("exit_mode", "z")
    z_exit   = float(cfg.get("z_exit", Z_EXIT_BASE))
    max_hold = int(cfg.get("max_hold", CONV_MAX_HOLD))

    pos = np.zeros(n)
    cur = 0.0
    hold = 0          # bars held in current position
    entry_sign = 0.0  # sign of z at entry (for convergence detection)
    for t in range(n):
        zt = z[t]
        if not np.isfinite(zt):
            cur = 0.0; hold = 0; entry_sign = 0.0
            pos[t] = cur
            continue
        if cur == 0.0:
            # entry
            if abs(zt) >= stop:
                # never enter beyond the stop band
                pass
            elif zt >= Z_ENTER:
                cur = -1.0; hold = 1; entry_sign = +1.0   # spread high -> short
            elif zt <= -Z_ENTER:
                cur = +1.0; hold = 1; entry_sign = -1.0   # spread low  -> long
        else:
            hold += 1
            exited = False
            # stop first (applies in z and conv modes; time mode sets stop=inf)
            if abs(zt) >= stop:
                cur = 0.0; exited = True
            elif exit_mode == "z":
                if abs(zt) <= z_exit:
                    cur = 0.0; exited = True
            elif exit_mode == "time":
                if hold >= max_hold:
                    cur = 0.0; exited = True
            elif exit_mode == "conv":
                # exit on sign flip of z (crossed 0) or hard cap
                if np.sign(zt) != entry_sign or hold >= max_hold:
                    cur = 0.0; exited = True
            if exited:
                hold = 0; entry_sign = 0.0
        pos[t] = cur

    # leg log-returns (NaN-safe), dollar-neutral spread return per unit pos
    ra = np.diff(la, prepend=la[0]); rb = np.diff(lb, prepend=lb[0])
    ra = np.where(np.isfinite(ra), ra, 0.0)
    rb = np.where(np.isfinite(rb), rb, 0.0)
    spread_ret = ra - rb

    # sizing
    if cfg.get("sizing", "unit") == "vol":
        vol_win = int(cfg.get("vol_win", roll_win if hedge == "roll" else 720))
        sret = pd.Series(spread_ret)
        tvol = sret.rolling(vol_win, min_periods=ROLL_HEDGE_MINP).std().shift(1)
        med = np.nanmedian(tvol.values)
        if not np.isfinite(med) or med <= 0:
            med = np.nanstd(spread_ret) or 1.0
        size = (med / tvol).values
        # guard: finite, bounded so a tiny-vol window can't blow up notional
        size = np.where(np.isfinite(size), size, 1.0)
        size = np.clip(size, 0.0, 5.0)
    else:
        size = np.ones(n)

    sized_pos = pos * size
    pos_lag = np.roll(sized_pos, 1); pos_lag[0] = 0.0
    gross_pnl = pos_lag * spread_ret

    dpos = np.abs(np.diff(sized_pos, prepend=0.0))
    cost_per_unit_change = 2.0 * (fee_bps + slip_bps) / 1e4
    cost = dpos * cost_per_unit_change
    net_pnl = gross_pnl - cost

    # trade count uses the discrete position (entries), turnover uses sized dpos
    dpos_disc = np.abs(np.diff(pos, prepend=0.0))
    n_trades = int(np.sum(dpos_disc > 0))
    turnover = float(np.sum(dpos))
    return dict(gross=gross_pnl, net=net_pnl, pos=pos, n_trades=n_trades,
                turnover=turnover, spread_ret=spread_ret)


# ----------------------------------------------------------------------------
# Walk-forward driver for a given cfg. Mirrors wb.run_strategy but routes through
# simulate_pair_v and threads the pair's train OU half-life into time-exit mode.
# ----------------------------------------------------------------------------
def run_variant(px, selector, cost_cfg, cfg, seed_for_random=None):
    logpx = np.log(px)
    cols = list(px.columns)
    splits = wb.make_splits(px.index)
    fee = cost_cfg["fee_bps"]; slip = cost_cfg["slip_bps"]
    all_net = []; all_gross = []
    tot_trades = 0; tot_turn = 0.0
    wins = 0; losses = 0
    n_pair_windows = 0
    for (tr_s, tr_e, te_e) in splits:
        tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(tr) < 1000 or len(te) < 200:
            continue
        if selector == "ou":
            picks = wb.select_pairs_ou(tr, cols, N_PAIRS)
        else:
            picks = wb.select_pairs_random(tr, cols, N_PAIRS, seed_for_random)
        if not picks:
            continue
        net_mat = []; gross_mat = []
        for p in picks:
            # time-exit horizon = 2 * train OU half-life bars (fallback 48h)
            hl = p.get("hl", np.nan)
            if cfg.get("exit_mode") == "time":
                mh = int(round(2.0 * hl)) if np.isfinite(hl) and hl > 0 else 48
                pcfg = dict(cfg); pcfg["max_hold"] = max(2, mh)
            else:
                pcfg = cfg
            r = simulate_pair_v(te, p["a"], p["b"], p["alpha"], p["beta"],
                                p["mu"], p["sd"], fee, slip, pcfg)
            if r is None:
                continue
            net_mat.append(r["net"]); gross_mat.append(r["gross"])
            tot_trades += r["n_trades"]; tot_turn += r["turnover"]
            n_pair_windows += 1
            pnl = r["net"]; pos = r["pos"]
            seg = 0.0; inpos = False
            for t in range(len(pos)):
                if pos[t] != 0:
                    seg += pnl[t]; inpos = True
                elif inpos:
                    (wins, losses) = (wins+1, losses) if seg > 0 else (wins, losses+1)
                    seg = 0.0; inpos = False
            if inpos:
                (wins, losses) = (wins+1, losses) if seg > 0 else (wins, losses+1)
        if not net_mat:
            continue
        L = min(len(v) for v in net_mat)
        net = np.mean([v[:L] for v in net_mat], axis=0)
        gross = np.mean([v[:L] for v in gross_mat], axis=0)
        all_net.append(net); all_gross.append(gross)
    if not all_net:
        return None
    net = np.concatenate(all_net); gross = np.concatenate(all_gross)
    sh_net, se_net, infl = wb.sharpe_hac(net)
    sh_gross, se_gross, _ = wb.sharpe_hac(gross)
    ci_net = wb.block_bootstrap_sharpe_ci(net, seed=(seed_for_random or 0))
    wr = wins/(wins+losses) if (wins+losses) > 0 else np.nan
    return dict(n_hours=len(net),
                sharpe_net=sh_net, se_net=se_net, ci_net=ci_net,
                sharpe_gross=sh_gross, se_gross=se_gross,
                net_pnl=float(np.nansum(net)), gross_pnl=float(np.nansum(gross)),
                n_trades=tot_trades, turnover=tot_turn, win_rate=wr,
                n_pair_windows=n_pair_windows)


def placebo(px, cost_cfg, cfg):
    """Random-pair placebo: avg net/gross Sharpe over seeds + across-seed SD."""
    runs = []
    for sd in RANDOM_SEEDS:
        rr = run_variant(px, "random", cost_cfg, cfg, seed_for_random=sd)
        if rr:
            runs.append(rr)
    if not runs:
        return None
    nets = [r["sharpe_net"] for r in runs]
    grs  = [r["sharpe_gross"] for r in runs]
    return dict(avg_net=float(np.mean(nets)), std_net=float(np.std(nets)),
                avg_gross=float(np.mean(grs)),
                avg_netpnl=float(np.mean([r["net_pnl"] for r in runs])),
                per_seed=nets)


# ----------------------------------------------------------------------------
# Variant matrix. id -> (human label, cfg). We do NOT take the full cross-product;
# we cover the key cells along each axis plus the decisive combined cell.
# ----------------------------------------------------------------------------
def build_variants():
    V = []
    base = dict(stop=4.0, exit_mode="z", z_exit=Z_EXIT_BASE,
                hedge="static", sizing="unit")

    # --- Axis 1: STOP (static hedge, z-exit) ---
    V.append(("S1_baseline_stop4",   {**base}))
    V.append(("S2_nostop",           {**base, "stop": np.inf}))
    V.append(("S3_widestop6",        {**base, "stop": 6.0}))

    # --- Axis 2: EXIT (static hedge, no stop where the rule defines holding) ---
    # time exit: enter |z|>=2, hold 2*HL bars, no z-exit, no stop
    V.append(("E1_time_2xHL",        {**base, "stop": np.inf, "exit_mode": "time"}))
    # hold-to-convergence: exit on z sign flip, cap 1000h, no stop
    V.append(("E2_conv_nostop",      {**base, "stop": np.inf, "exit_mode": "conv",
                                      "max_hold": CONV_MAX_HOLD}))

    # --- Axis 3: ROLLING HEDGE (z-exit, baseline stop) at both windows ---
    for w in ROLL_HEDGE_WINS:
        V.append((f"H_roll{w}_zexit", {**base, "hedge": "roll", "roll_win": w}))
    # rolling hedge + NO stop + z-exit
    for w in ROLL_HEDGE_WINS:
        V.append((f"H_roll{w}_nostop", {**base, "hedge": "roll", "roll_win": w,
                                        "stop": np.inf}))

    # --- Axis 4: SIZING (vol-targeted), static hedge baseline rule ---
    V.append(("Z_voltarget_static", {**base, "sizing": "vol", "vol_win": 720}))

    # --- DECISIVE CELL: no stop + rolling hedge + conv exit ---
    for w in ROLL_HEDGE_WINS:
        V.append((f"D_roll{w}_nostop_conv",
                  {**base, "hedge": "roll", "roll_win": w, "stop": np.inf,
                   "exit_mode": "conv", "max_hold": CONV_MAX_HOLD}))
    # plus a vol-targeted version of the decisive cell (60d)
    V.append(("D_roll1440_nostop_conv_vol",
              {**base, "hedge": "roll", "roll_win": 1440, "stop": np.inf,
               "exit_mode": "conv", "max_hold": CONV_MAX_HOLD,
               "sizing": "vol", "vol_win": 1440}))
    return V


def fmt_full(label, r):
    if r is None:
        return f"  {label:<28} NO RESULT"
    ci = r["ci_net"]
    return (f"  {label:<28} "
            f"net S={r['sharpe_net']:+.2f}(SE{r['se_net']:.2f},CI[{ci[0]:+.2f},{ci[1]:+.2f}]) "
            f"gross S={r['sharpe_gross']:+.2f} "
            f"netPnL={r['net_pnl']*100:+.1f}% trades={r['n_trades']} "
            f"turn={r['turnover']:.0f} WR={r['win_rate']*100:.1f}%")


def main():
    t0 = time.time()
    px = wb.load_universe()
    splits = wb.make_splits(px.index)
    print("="*108)
    print("ROBUSTNESS MATRIX -- pair mean-reversion (stop / exit / hedge / sizing)")
    print(f"universe={px.shape[1]} syms | hours={px.shape[0]} | "
          f"span={px.index.min().date()}..{px.index.max().date()} | "
          f"{len(splits)} walk-forward splits | N_PAIRS={N_PAIRS} | "
          f"z enter={Z_ENTER} | placebo seeds={RANDOM_SEEDS}")
    print("="*108)

    variants = build_variants()
    summary = []  # (label, cost, gross, net, plac_net, plac_sd)

    for clabel, ccfg in COST_LEVELS.items():
        print(f"\n########## COST LEVEL: {clabel} "
              f"(fee {ccfg['fee_bps']}bps + slip {ccfg['slip_bps']}bps/leg) ##########")
        for label, cfg in variants:
            r = run_variant(px, "ou", ccfg, cfg)
            pl = placebo(px, ccfg, cfg)
            print(fmt_full(label, r))
            if pl is not None:
                edge = (r["sharpe_net"] - pl["avg_net"]) if r else np.nan
                beats = ("BEATS placebo" if (r and edge > pl["std_net"] and r["sharpe_net"] > 0)
                         else "<= placebo")
                print(f"    placebo(random): net S={pl['avg_net']:+.2f} "
                      f"(SD {pl['std_net']:.2f}) gross S={pl['avg_gross']:+.2f} "
                      f"netPnL={pl['avg_netpnl']*100:+.1f}% per-seed={['{:+.2f}'.format(x) for x in pl['per_seed']]}  "
                      f"=> edge={edge:+.2f} [{beats}]")
            summary.append(dict(
                label=label, cost=clabel,
                gross=(r["sharpe_gross"] if r else np.nan),
                net=(r["sharpe_net"] if r else np.nan),
                plac_net=(pl["avg_net"] if pl else np.nan),
                plac_sd=(pl["std_net"] if pl else np.nan)))

    # ---------------- compact final table ----------------
    print("\n" + "="*108)
    print("COMPACT SUMMARY  (S=annualized Sharpe; edge=net-placebo; * = net>0 AND edge>placebo_SD)")
    print("="*108)
    # reshape to: per variant, gross + net@30bps + net@2bps + placebo@each
    by_label = {}
    for s in summary:
        by_label.setdefault(s["label"], {})[s["cost"]] = s
    hdr = (f"{'variant':<28} {'grossS':>7} "
           f"{'net30':>7} {'plac30':>7} "
           f"{'net2':>7} {'plac2':>7} {'flag':>6}")
    print(hdr); print("-"*len(hdr))
    order = [l for l, _ in build_variants()]
    for label in order:
        d = by_label.get(label, {})
        a = d.get("realistic_30bps_rt"); b = d.get("low_2bps_rt")
        if a is None or b is None:
            continue
        # gross identical across cost levels (cost-free); take from low
        gross = b["gross"]
        flag = ""
        # flag the most-favorable: net2 > 0 and beats placebo by > SD
        if (np.isfinite(b["net"]) and b["net"] > 0 and np.isfinite(b["plac_net"])
                and (b["net"] - b["plac_net"]) > b["plac_sd"]):
            flag = "*EDGE*"
        print(f"{label:<28} {gross:>+7.2f} "
              f"{a['net']:>+7.2f} {a['plac_net']:>+7.2f} "
              f"{b['net']:>+7.2f} {b['plac_net']:>+7.2f} {flag:>6}")

    print("\n" + "="*108)
    print(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
