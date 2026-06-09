"""
Deployability test (last constructive item): can a STRUCTURAL-BREAK stop cap the
per-pair delisting tail WITHOUT hurting the effect? The project's "stops hurt" finding
is about a SPREAD-z stop (which exits normal reversion). A LEG-CRASH stop is different:
it exits a pair only when the leg you are LONG drops > K% from entry (a delisting/crash
signature), not on normal |z| divergence — so it should leave the normal reverting pairs
untouched while bounding the LUNA-type −100%/leg tail.

Universe = top-50 + delisted (LUNA/UST/FTT/LUNC, point-in-time) so the tail can manifest.
Compares no-stop vs no-stop+leg-crash-stop (K=30/50/70%): monthly Sharpe, maxDD, and the
WORST single per-(pair,window) loss (the tail metric the stop is meant to cap).

Run:  python scratch/nostop_breakstop.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import wf_robustness as wr
import wf_survivorship as ws

CC = wb.COST_LEVELS["realistic_30bps_rt"]
DELI_DIR = os.path.join(wb.ROOT, "data", "spot_1h_delisted")
DELISTED = ["LUNAUSDT", "USTUSDT", "FTTUSDT", "LUNCUSDT"]
N_PAIRS = 40
Z_ENTER, Z_EXIT = 2.0, 0.5


def load_top50_plus_delisted():
    top50 = [s.strip().upper() for s in open(wb.UNIV) if s.strip()]
    px = ws.load_universe_from(top50)
    extra = {}
    for s in DELISTED:
        f = os.path.join(DELI_DIR, f"{s}.parquet")
        if os.path.exists(f):
            c = pd.read_parquet(f, columns=["close"])["close"]; c.index = pd.to_datetime(c.index, utc=True)
            extra[s] = c[~c.index.duplicated(keep="last")]
    return pd.concat([px, pd.DataFrame(extra).reindex(px.index).ffill(limit=3)], axis=1)


def sim_pair(te, a, b, alpha, beta, mu, sd, fee, slip, crash_K=None, adverse_K=None, posloss_X=None, halt=False):
    """No-stop static-hedge sim with optional break-stops:
      crash_K   : exit when the LONG leg drops > K from entry (delisting/crash of held leg)
      adverse_K : exit when EITHER leg moves > K adversely (long-leg crash OR short-leg pump)
      posloss_X : exit when the current position's cumulative net P&L < -X (direct tail cap = a real stop)
      halt      : if True, after a break/loss stop fires, HALT the pair for the rest of the window
                  (a pair-level circuit breaker — prevents re-entry into a sustained crisis)
    """
    la = te[a].values.astype(float); lb = te[b].values.astype(float)
    n = len(la)
    if sd <= 0 or not np.isfinite(sd):
        return None
    z = (la - alpha - beta * lb - mu) / sd
    cthr = -np.log(1.0 - crash_K) if crash_K else None
    athr = -np.log(1.0 - adverse_K) if adverse_K else None
    pos = np.zeros(n); cur = 0.0; la0 = lb0 = 0.0; pos_pnl = 0.0; halted = False
    ra_ = np.diff(la, prepend=la[0]); rb_ = np.diff(lb, prepend=lb[0])
    ra_ = np.where(np.isfinite(ra_), ra_, 0.0); rb_ = np.where(np.isfinite(rb_), rb_, 0.0)
    for t in range(n):
        if cur != 0.0:                                   # mark this bar's P&L for posloss
            pos_pnl += cur * (ra_[t] - rb_[t])
        zt = z[t]
        if not np.isfinite(zt):
            cur = 0.0; pos_pnl = 0.0
        elif cur == 0.0:
            if not halted:
                if zt >= Z_ENTER:
                    cur = -1.0; la0, lb0 = la[t], lb[t]; pos_pnl = 0.0    # short A, LONG B
                elif zt <= -Z_ENTER:
                    cur = +1.0; la0, lb0 = la[t], lb[t]; pos_pnl = 0.0    # LONG A, short B
        else:
            stop = False; brk = False
            if abs(zt) <= Z_EXIT:
                stop = True
            elif cthr is not None and ((cur > 0 and la[t]-la0 < -cthr) or (cur < 0 and lb[t]-lb0 < -cthr)):
                stop = True; brk = True                  # long-leg crash
            elif athr is not None and ((cur > 0 and (la[t]-la0 < -athr or lb[t]-lb0 > athr)) or
                                       (cur < 0 and (lb[t]-lb0 < -athr or la[t]-la0 > athr))):
                stop = True; brk = True                  # either leg adverse (crash or pump)
            elif posloss_X is not None and pos_pnl < -posloss_X:
                stop = True; brk = True                  # direct position-loss cap
            if stop:
                cur = 0.0; pos_pnl = 0.0
                if brk and halt:
                    halted = True                        # circuit breaker: no re-entry this window
        pos[t] = cur
    ra = np.diff(la, prepend=la[0]); rb = np.diff(lb, prepend=lb[0])
    ra = np.where(np.isfinite(ra), ra, 0.0); rb = np.where(np.isfinite(rb), rb, 0.0)
    spread_ret = ra - rb
    pos_lag = np.roll(pos, 1); pos_lag[0] = 0.0
    gross = pos_lag * spread_ret
    dpos = np.abs(np.diff(pos, prepend=0.0))
    cost = dpos * 2.0 * (fee + slip) / 1e4
    net = gross - cost
    return net


def run(px, **kw):
    logpx = np.log(px); cols = list(px.columns); splits = wb.make_splits(px.index)
    parts = []; worst = 0.0
    for (tr_s, tr_e, te_e) in splits:
        tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(tr) < 1000 or len(te) < 200:
            continue
        avail = [c for c in cols if tr[c].notna().mean() >= wb.MIN_OBS_FRAC]
        picks = wb.select_pairs_ou(tr[avail], avail, N_PAIRS)
        if not picks:
            continue
        nmat = []
        for p in picks:
            net = sim_pair(te, p["a"], p["b"], p["alpha"], p["beta"], p["mu"], p["sd"],
                           CC["fee_bps"], CC["slip_bps"], **kw)
            if net is not None:
                nmat.append(pd.Series(net, index=te.index[:len(net)]))
                worst = min(worst, float(np.nansum(net)))
        if nmat:
            L = min(len(v) for v in nmat)
            parts.append(pd.Series(np.mean([v.values[:L] for v in nmat], axis=0), index=te.index[:L]))
    s = pd.concat(parts).sort_index(); s = s[~s.index.duplicated(keep="first")]
    m = s.resample("ME").sum(); m = m[m != 0]
    sh_m = float(m.mean() / m.std(ddof=1) * np.sqrt(12)) if len(m) > 5 and m.std() > 0 else np.nan
    eq = np.cumsum(np.nan_to_num(s.values)); dd = float((eq - np.maximum.accumulate(eq)).min())
    return sh_m, dd, worst


def main():
    px = load_top50_plus_delisted()
    print(f"universe top-50 + delisted = {px.shape[1]} syms")
    print(f"\n{'variant':<30}{'monthlyS':>10}{'maxDD%':>9}{'worst pair-window%':>20}")
    print("-" * 70)
    variants = [
        ("no-stop (baseline)", {}),
        ("+ either-leg adverse 50%", {"adverse_K": 0.50}),
        ("+ position-loss stop 25%", {"posloss_X": 0.25}),
        ("+ either-leg 50% + HALT", {"adverse_K": 0.50, "halt": True}),
        ("+ either-leg 30% + HALT", {"adverse_K": 0.30, "halt": True}),
        ("+ posloss 25% + HALT", {"posloss_X": 0.25, "halt": True}),
    ]
    for label, kw in variants:
        sh, dd, w = run(px, **kw)
        print(f"{label:<30}{sh:>10.2f}{dd*100:>9.1f}{w*100:>20.0f}")
    print("\nReading: a break-stop that CAPS the per-pair tail without cutting monthly Sharpe would be")
    print("deployable; if every tail-capping stop also cuts the Sharpe, the no-stop tail risk is intrinsic.")


if __name__ == "__main__":
    main()
