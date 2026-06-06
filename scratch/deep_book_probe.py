"""
Deep-book (levels 1-10) predictive probe.

Q: Does the DEEPER book carry a slower, possibly-tradeable signal beyond
top-of-book imbalance (which prior work found dies by ~10s)?

Features per 1s bar:
  (a) fair-value tilt  fv_tilt = (depth-weighted fair value - mid)/mid
  (b) deep imbalance   I_deep = (sum bid_sz_1..10 - sum ask_sz_1..10)/sum
      top imbalance    I_top  = (bid_sz_1 - ask_sz_1)/(bid_sz_1+ask_sz_1)
  (c) book slope       slope_bid, slope_net (cum size growth vs distance from mid)

Predictive test: future mid log-return over [t, t+h] regressed on each feature,
HAC (Newey-West) SEs. Report beta, t, R2 by horizon h in {1,5,10,30,60,300}s.
Incremental R2 of deep features over top-of-book in joint regression.
"""
import os, glob
import numpy as np
import pandas as pd
import statsmodels.api as sm

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DATES = [f"2024-01-{d:02d}" for d in range(2, 14)]  # 12 days
HORIZONS = [1, 5, 10, 30, 60, 300]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "l2")

BIDPX = [f"bid_px_{i}" for i in range(1, 11)]
ASKPX = [f"ask_px_{i}" for i in range(1, 11)]
BIDSZ = [f"bid_sz_{i}" for i in range(1, 11)]
ASKSZ = [f"ask_sz_{i}" for i in range(1, 11)]


def build_features(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    bpx = df[BIDPX].to_numpy(float)
    apx = df[ASKPX].to_numpy(float)
    bsz = df[BIDSZ].to_numpy(float)
    asz = df[ASKSZ].to_numpy(float)

    mid = (bpx[:, 0] + apx[:, 0]) / 2.0

    # (a) depth-weighted fair value across all 10 levels both sides
    notional = (bpx * bsz).sum(1) + (apx * asz).sum(1)
    totsz = bsz.sum(1) + asz.sum(1)
    fair = notional / totsz
    fv_tilt = (fair - mid) / mid

    # (b) imbalances
    sb = bsz.sum(1); sa = asz.sum(1)
    I_deep = (sb - sa) / (sb + sa)
    I_top = (bsz[:, 0] - asz[:, 0]) / (bsz[:, 0] + asz[:, 0])
    # imbalance of deep-only (levels 2-10), to isolate the "deeper" contribution
    sb2 = bsz[:, 1:].sum(1); sa2 = asz[:, 1:].sum(1)
    I_deeponly = (sb2 - sa2) / (sb2 + sa2)

    # (c) book slope: cum size per unit relative price distance from mid, per side.
    # slope = total size within 10 levels / mean fractional distance of that size from mid.
    bdist = (mid[:, None] - bpx) / mid[:, None]      # >=0
    adist = (apx - mid[:, None]) / mid[:, None]      # >=0
    eps = 1e-12
    bid_wdist = (bsz * bdist).sum(1) / (bsz.sum(1) + eps)   # size-weighted depth distance bid
    ask_wdist = (asz * adist).sum(1) / (asz.sum(1) + eps)
    slope_bid = bsz.sum(1) / (bid_wdist + eps)
    slope_ask = asz.sum(1) / (ask_wdist + eps)
    # net slope asymmetry, normalized
    slope_net = (slope_bid - slope_ask) / (slope_bid + slope_ask + eps)

    out = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "mid": mid,
        "fv_tilt": fv_tilt,
        "I_deep": I_deep,
        "I_top": I_top,
        "I_deeponly": I_deeponly,
        "slope_net": slope_net,
    })
    out["logmid"] = np.log(out["mid"])
    return out


def load_symbol(sym):
    frames = []
    for dt in DATES:
        p = os.path.join(DATA, sym, f"{dt}.parquet")
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        # one record per second already (1s bars); keep last per second to be safe
        d["sec"] = d["timestamp"].dt.floor("S")
        d = d.groupby("sec", as_index=False).last()
        f = build_features(d)
        f["date"] = dt
        # guard against bad rows (crossed/empty book)
        f = f[np.isfinite(f["fv_tilt"]) & np.isfinite(f["I_deep"]) & np.isfinite(f["slope_net"])]
        frames.append(f)
    return frames  # list of per-day frames (keep day boundaries for horizon construction)


def hac_reg(y, X, maxlags):
    X = sm.add_constant(X)
    m = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return m


def main():
    np.seterr(all="ignore")
    feats = ["fv_tilt", "I_deep", "I_deeponly", "slope_net", "I_top"]
    results = {}      # univariate
    joint_results = {}  # incremental

    for sym in SYMS:
        day_frames = load_symbol(sym)
        # Build forward returns within each day (no cross-day leakage), then concat
        per_h = {h: {"y": [], "X": {f: [] for f in feats}} for h in HORIZONS}
        # standardize features per symbol using pooled stats
        allf = pd.concat(day_frames, ignore_index=True)
        means = {f: allf[f].mean() for f in feats}
        stds = {f: allf[f].std() for f in feats}

        for df in day_frames:
            lm = df["logmid"].to_numpy()
            n = len(lm)
            fvals = {f: ((df[f].to_numpy() - means[f]) / stds[f]) for f in feats}
            for h in HORIZONS:
                if n <= h:
                    continue
                fwd = np.full(n, np.nan)
                fwd[:n - h] = lm[h:] - lm[:n - h]  # log return [t, t+h]
                valid = np.isfinite(fwd)
                per_h[h]["y"].append(fwd[valid])
                for f in feats:
                    per_h[h]["X"][f].append(fvals[f][valid])

        results[sym] = {}
        joint_results[sym] = {}
        for h in HORIZONS:
            if not per_h[h]["y"]:
                continue
            y = np.concatenate(per_h[h]["y"]) * 1e4  # bps
            Xd = {f: np.concatenate(per_h[h]["X"][f]) for f in feats}
            maxlags = h  # NW lags ~ overlap length
            # univariate per feature
            results[sym][h] = {}
            for f in feats:
                m = hac_reg(y, Xd[f].reshape(-1, 1), maxlags)
                results[sym][h][f] = (m.params[1], m.tvalues[1], m.rsquared, len(y))
            # joint incremental: top-of-book baseline vs +deep features
            base = sm.add_constant(Xd["I_top"].reshape(-1, 1))
            mb = sm.OLS(y, base, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
            Xfull = np.column_stack([Xd["I_top"], Xd["fv_tilt"], Xd["I_deeponly"], Xd["slope_net"]])
            mf = hac_reg(y, Xfull, maxlags)
            joint_results[sym][h] = {
                "r2_top": mb.rsquared,
                "r2_full": mf.rsquared,
                "incr": mf.rsquared - mb.rsquared,
                "t_fvtilt": mf.tvalues[2], "t_Ideeponly": mf.tvalues[3], "t_slope": mf.tvalues[4],
                "n": len(y),
            }

    # ---- Report ----
    print("=" * 96)
    print("DEEP-BOOK PREDICTIVE PROBE  | features standardized (z); future return in bps; HAC SEs")
    print(f"Symbols={SYMS}  Days={DATES[0]}..{DATES[-1]} ({len(DATES)}d)  Horizons(s)={HORIZONS}")
    print("=" * 96)

    for sym in SYMS:
        print(f"\n### {sym}  (N obs per horizon shown)")
        print("UNIVARIATE: beta(bps per 1sd) / t / R2(%)")
        header = "h(s) |" + "".join(f"{f:>22}" for f in feats)
        print(header)
        for h in HORIZONS:
            if h not in results[sym]:
                continue
            row = f"{h:>4} |"
            for f in feats:
                b, t, r2, n = results[sym][h][f]
                row += f"  {b:>7.2f}/{t:>5.1f}/{r2*100:>5.2f}"
            print(row)
        print(f"  (N at h=1: {results[sym].get(1,{}).get('fv_tilt',[None]*4)[3]})")

        print("\nINCREMENTAL R2 of deep features over top-of-book I_top (joint OLS):")
        print("h(s) | R2_top(%)  R2_full(%)  incrR2(%)  | t_fvtilt  t_Ideeponly  t_slope")
        for h in HORIZONS:
            if h not in joint_results[sym]:
                continue
            j = joint_results[sym][h]
            print(f"{h:>4} |  {j['r2_top']*100:>7.3f}  {j['r2_full']*100:>9.3f}  {j['incr']*100:>8.4f}  |"
                  f"  {j['t_fvtilt']:>7.1f}  {j['t_Ideeponly']:>10.1f}  {j['t_slope']:>7.1f}")

    # Pooled view: average R2 across symbols
    print("\n" + "=" * 96)
    print("POOLED (mean across 3 symbols)")
    print("=" * 96)
    print("Univariate R2(%) by horizon:")
    print("h(s) |" + "".join(f"{f:>14}" for f in feats))
    for h in HORIZONS:
        row = f"{h:>4} |"
        for f in feats:
            vals = [results[s][h][f][2] * 100 for s in SYMS if h in results[s]]
            row += f"{np.mean(vals):>14.3f}"
        print(row)
    print("\nUnivariate |t| by horizon (mean abs):")
    print("h(s) |" + "".join(f"{f:>14}" for f in feats))
    for h in HORIZONS:
        row = f"{h:>4} |"
        for f in feats:
            vals = [abs(results[s][h][f][1]) for s in SYMS if h in results[s]]
            row += f"{np.mean(vals):>14.1f}"
        print(row)
    print("\nIncremental R2(%) of deep over top-of-book (mean across symbols):")
    print("h(s) | R2_top  R2_full  incrR2")
    for h in HORIZONS:
        rt = np.mean([joint_results[s][h]["r2_top"] * 100 for s in SYMS if h in joint_results[s]])
        rf = np.mean([joint_results[s][h]["r2_full"] * 100 for s in SYMS if h in joint_results[s]])
        ic = np.mean([joint_results[s][h]["incr"] * 100 for s in SYMS if h in joint_results[s]])
        print(f"{h:>4} | {rt:>6.3f}  {rf:>6.3f}  {ic:>7.4f}")

    # Economic magnitude: best deep feature's predicted move at +-2sd vs costs
    print("\n" + "=" * 96)
    print("ECONOMIC CHECK: predicted mid move (bps) at +1sd of best deep feature, by horizon")
    print("vs BTC half-spread ~0.002-0.005bps, taker cost ~10bps")
    print("=" * 96)
    for sym in SYMS:
        print(f"\n{sym}: |beta| (bps per +1sd) for fv_tilt / I_deep / slope_net")
        for h in HORIZONS:
            if h not in results[sym]:
                continue
            b_fv = abs(results[sym][h]["fv_tilt"][0])
            b_id = abs(results[sym][h]["I_deep"][0])
            b_sl = abs(results[sym][h]["slope_net"][0])
            print(f"  h={h:>3}s:  fv_tilt={b_fv:>6.2f}   I_deep={b_id:>6.2f}   slope_net={b_sl:>6.2f}")


if __name__ == "__main__":
    main()
