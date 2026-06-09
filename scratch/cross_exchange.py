"""
Cross-exchange validation (the one ORTHOGONAL test of the no-stop reversion effect):
run the IDENTICAL no-stop pipeline on Binance vs Coinbase hourly data, restricted to
the SAME overlapping major symbols, so the only thing that differs is the EXCHANGE
(different venue/participants, USD vs USDT quote). If the monthly Sharpe replicates on
Coinbase (~2-2.5), the effect is real and not Binance-specific selection/overfitting —
the independent evidence the red-team's "correlated supports" criticism asked for. If it
collapses on Coinbase, the Binance result was venue-specific.

Run (after coinbase_pull.py finishes):  python scratch/cross_exchange.py
"""
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_backtest as wb
import wf_robustness as wr
import wf_survivorship as ws
import nostop_breakstop as nbs   # for the real-time circuit-breaker + ex-ACH robustness passes

CC = wb.COST_LEVELS["realistic_30bps_rt"]
BN_DIR = os.path.join(wb.ROOT, "data", "spot_1h")
CB_DIR = os.path.join(wb.ROOT, "data", "coinbase_1h")
NOSTOP = dict(stop=np.inf, exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")
STOP4 = dict(stop=4.0, exit_mode="z", z_exit=wb.Z_EXIT, hedge="static", sizing="unit")


def load_dir(d, symbols):
    closes = {}
    for s in symbols:
        f = os.path.join(d, f"{s}.parquet")
        if not os.path.exists(f):
            continue
        c = pd.read_parquet(f, columns=["close"])["close"]
        c.index = pd.to_datetime(c.index, utc=True)
        closes[s] = c[~c.index.duplicated(keep="last")]
    px = pd.DataFrame(closes).sort_index()
    full = pd.date_range(px.index.min().floor("h"), px.index.max().ceil("h"), freq="h", tz="UTC")
    px = px.reindex(full)
    px = px[px.index >= pd.Timestamp(wb.START_FLOOR, tz="UTC")].ffill(limit=3)
    for stub in ws.STABLES:
        px = px.drop(columns=stub, errors="ignore")
    return px


def overlap():
    cb = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(CB_DIR, "*.parquet"))}
    bn = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(BN_DIR, "*.parquet"))}
    return sorted(cb & bn)


def run(px, npairs, cfg):
    logpx = np.log(px); cols = list(px.columns); splits = wb.make_splits(px.index)
    parts = []
    for (tr_s, tr_e, te_e) in splits:
        tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(tr) < 1000 or len(te) < 200:
            continue
        avail = [c for c in cols if tr[c].notna().mean() >= wb.MIN_OBS_FRAC]
        picks = wb.select_pairs_ou(tr[avail], avail, npairs)
        if not picks:
            continue
        nmat = []
        for p in picks:
            r = wr.simulate_pair_v(te, p["a"], p["b"], p["alpha"], p["beta"], p["mu"], p["sd"],
                                   CC["fee_bps"], CC["slip_bps"], cfg)
            if r is not None:
                nmat.append(pd.Series(r["net"], index=te.index[:len(r["net"])]))
        if nmat:
            L = min(len(v) for v in nmat)
            parts.append(pd.Series(np.mean([v.values[:L] for v in nmat], axis=0), index=te.index[:L]))
    if not parts:
        return np.nan, np.nan, np.nan
    s = pd.concat(parts).sort_index(); s = s[~s.index.duplicated(keep="first")]
    sh_h, _, _ = wb.sharpe_hac(s.values)
    m = s.resample("ME").sum(); m = m[m != 0]
    sh_m = float(m.mean() / m.std(ddof=1) * np.sqrt(12)) if len(m) > 5 and m.std() > 0 else np.nan
    eq = np.cumsum(np.nan_to_num(s.values)); dd = float((eq - np.maximum.accumulate(eq)).min())
    return sh_h, sh_m, dd


def run_bs(px, npairs, **kw):
    """Same walk-forward as run(), but uses nostop_breakstop.sim_pair so we can apply the
    real-time circuit breaker (adverse_K + halt). Returns (monthlyS, maxDD)."""
    logpx = np.log(px); cols = list(px.columns); splits = wb.make_splits(px.index); parts = []
    for (tr_s, tr_e, te_e) in splits:
        tr = logpx[(logpx.index >= tr_s) & (logpx.index < tr_e)]
        te = logpx[(logpx.index >= tr_e) & (logpx.index < te_e)]
        if len(tr) < 1000 or len(te) < 200:
            continue
        avail = [c for c in cols if tr[c].notna().mean() >= wb.MIN_OBS_FRAC]
        picks = wb.select_pairs_ou(tr[avail], avail, npairs)
        if not picks:
            continue
        nm = []
        for p in picks:
            net = nbs.sim_pair(te, p["a"], p["b"], p["alpha"], p["beta"], p["mu"], p["sd"],
                               CC["fee_bps"], CC["slip_bps"], **kw)
            if net is not None:
                nm.append(pd.Series(net, index=te.index[:len(net)]))
        if nm:
            L = min(len(v) for v in nm)
            parts.append(pd.Series(np.mean([v.values[:L] for v in nm], axis=0), index=te.index[:L]))
    s = pd.concat(parts).sort_index(); s = s[~s.index.duplicated(keep="first")]
    m = s.resample("ME").sum(); m = m[m != 0]
    sh = float(m.mean() / m.std(ddof=1) * np.sqrt(12)) if len(m) > 5 and m.std() > 0 else np.nan
    eq = np.cumsum(np.nan_to_num(s.values)); dd = float((eq - np.maximum.accumulate(eq)).min())
    return sh, dd


def main():
    ov = overlap()
    print("=" * 80)
    print(f"CROSS-EXCHANGE VALIDATION: {len(ov)} symbols on BOTH Binance & Coinbase")
    print(f"  {ov}")
    print("=" * 80)
    px_bn = load_dir(BN_DIR, ov); px_cb = load_dir(CB_DIR, ov)
    print(f"  Binance panel: {px_bn.shape}  span {px_bn.index.min().date()}..{px_bn.index.max().date()}")
    print(f"  Coinbase panel: {px_cb.shape}  span {px_cb.index.min().date()}..{px_cb.index.max().date()}")
    print(f"\n[A] RAW no-stop & spread-stop, identical pipeline:")
    print(f"{'config':<22}{'venue':<10}{'hourlyS':>9}{'monthlyS':>10}{'maxDD%':>9}")
    print("-" * 60)
    for npairs in (10, 20):
        for cfgname, cfg in [("no-stop", NOSTOP), ("stop|z|=4", STOP4)]:
            for vname, px in [("Binance", px_bn), ("Coinbase", px_cb)]:
                sh_h, sh_m, dd = run(px, npairs, cfg)
                print(f"{cfgname+' '+str(npairs)+'p':<22}{vname:<10}{sh_h:>9.2f}{sh_m:>10.2f}{dd*100:>9.1f}")

    print(f"\n[B] no-stop + REAL-TIME circuit breaker (either-leg 50% adverse + HALT) — deployable risk rule:")
    print(f"{'venue':<10}{'pairs':<7}{'rawMonthlyS':>13}{'+CB monthlyS':>13}{'+CB maxDD%':>12}")
    print("-" * 56)
    for vname, px in [("Binance", px_bn), ("Coinbase", px_cb)]:
        for npairs in (10, 20):
            raw_sh, _ = run_bs(px, npairs)
            cb_sh, cb_dd = run_bs(px, npairs, adverse_K=0.50, halt=True)
            print(f"{vname:<10}{str(npairs)+'p':<7}{raw_sh:>13.2f}{cb_sh:>13.2f}{cb_dd*100:>12.1f}")

    print(f"\n[C] ex-ACH (HINDSIGHT exclusion of one +308% pump — NOT deployable, for diagnosis only):")
    ov2 = [s for s in ov if s != "ACHUSDT"]
    for vname, d in [("Binance", BN_DIR), ("Coinbase", CB_DIR)]:
        px2 = load_dir(d, ov2)
        for npairs in (10, 20):
            _, sh, dd = run(px2, npairs, NOSTOP)
            print(f"  {vname:<10}{npairs}p no-stop ex-ACH: monthly={sh:+.2f}  maxDD={dd*100:.1f}%")
    print("\nReading: [A] raw replicates POSITIVE on Coinbase (~0.9-1.2) but ~50-60% weaker than Binance ~2.5.")
    print("[B] the circuit breaker caps the Coinbase DD (-98%->-48%) but barely moves the Sharpe (~1.0).")
    print("[C] ex-ACH 1.4-1.6 requires HINDSIGHT (un-tradeable) -> the honest venue-robust figure is ~1.0-1.2.")


if __name__ == "__main__":
    main()
