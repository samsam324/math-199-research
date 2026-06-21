"""Harden the flagship Kalman-artifact result (Table 1): re-run the IDENTICAL screen
at several disjoint quarterly t0 anchors, with Wilson 95% CIs on every pass rate, to
show the artifact (real == placebo pass rate ~100%, clean EG low) is not specific to
2024-01-01. Reuses audit_part1's exact Kalman/ADF/placebo functions unchanged.

Per anchor: real Kalman OOS ADF rate, three placebos (independent RW, phase-randomized,
block-shuffled), and the clean Engle-Granger benchmark. 120 real pairs / 80 per placebo
(Wilson CI ~ +/-3pp at 100%). RNG reseeded per anchor for independence + reproducibility.

Run: python scratch/book_kalman_anchors.py
"""
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scratch.audit_part1 as ap  # reuse exact functions

ANCHORS = ["2023-07-01", "2023-10-01", "2024-01-01", "2024-04-01", "2024-07-01"]
N_REAL, N_PLAC = 120, 80
TRAIN_DAYS, TEST_DAYS = 90, 30


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (100 * max(0.0, c - half), 100 * min(1.0, c + half))


def load_panel_at(T0):
    full_start = T0 - pd.Timedelta(days=TRAIN_DAYS + TEST_DAYS)
    full_end = T0
    train_end = T0 - pd.Timedelta(days=TEST_DAYS)
    cols = {}
    for s in ap.UNI:
        c = pd.read_parquet(ROOT / f"data/spot_1h/{s}.parquet")["close"]
        c = c[(c.index >= full_start) & (c.index < full_end)]
        cols[s] = c
    panel = pd.DataFrame(cols).sort_index()
    cov = panel.notna().mean()
    panel = panel[cov[cov >= 0.80].index.tolist()].dropna(how="any")
    return panel, train_end, full_end


def screen_anchor(anchor, seed):
    ap.RNG = np.random.default_rng(seed)
    T0 = pd.Timestamp(anchor + "T00:00:00Z")
    panel, train_end, full_end = load_panel_at(T0)
    logp = np.log(panel)
    tr = logp[logp.index < train_end]
    te = logp[(logp.index >= train_end) & (logp.index < full_end)]
    syms = list(panel.columns)
    all_pairs = list(combinations(syms, 2))
    nt, ne = len(tr), len(te)

    # real Kalman + static, on a random subsample
    ridx = ap.RNG.choice(len(all_pairs), size=min(N_REAL, len(all_pairs)), replace=False)
    real_k = []
    for i in ridx:
        a, b = all_pairs[i]
        if len(tr) < 200 or len(te) < 100:
            continue
        pk, _ = ap.kalman_oos_p(tr[a].to_numpy(), tr[b].to_numpy(), te[a].to_numpy(), te[b].to_numpy())
        if np.isfinite(pk):
            real_k.append(pk)
    real_k = np.array(real_k)

    psub = [all_pairs[i] for i in ap.RNG.choice(len(all_pairs), size=min(N_PLAC, len(all_pairs)), replace=False)]
    sig = float(np.std(np.diff(tr[syms[0]].to_numpy())))

    rw_k = []
    for _ in range(N_PLAC):
        ay, axx = ap.sim_rw(nt + ne, sig), ap.sim_rw(nt + ne, sig)
        pk, _ = ap.kalman_oos_p(ay[:nt], axx[:nt], ay[nt:], axx[nt:])
        if np.isfinite(pk):
            rw_k.append(pk)
    rw_k = np.array(rw_k)

    pr_k, bs_k = [], []
    for a, b in psub:
        yf = logp[a].to_numpy()
        xf = ap.phase_randomize(logp[b].to_numpy())
        pk, _ = ap.kalman_oos_p(yf[:nt], xf[:nt], yf[nt:nt + ne], xf[nt:nt + ne])
        if np.isfinite(pk):
            pr_k.append(pk)
    pr_k = np.array(pr_k)
    for a, b in psub:
        yf = logp[a].to_numpy()
        xf = ap.block_shuffle(logp[b].to_numpy(), block=24)
        pk, _ = ap.kalman_oos_p(yf[:nt], xf[:nt], yf[nt:nt + ne], xf[nt:nt + ne])
        if np.isfinite(pk):
            bs_k.append(pk)
    bs_k = np.array(bs_k)

    eg_p = []
    from statsmodels.tsa.stattools import coint
    for a, b in all_pairs:
        yte, xte = te[a].to_numpy(), te[b].to_numpy()
        if len(yte) < 100:
            continue
        try:
            _, p, _ = coint(yte, xte, trend="c", autolag=None, maxlag=24)
            eg_p.append(float(p))
        except Exception:
            pass
    eg_p = np.array(eg_p)

    def cell(arr, thr=0.05):
        k = int(np.sum(arr < thr)); n = len(arr)
        lo, hi = wilson(k, n)
        return {"rate": 100 * k / n if n else float("nan"), "n": n, "lo": lo, "hi": hi}

    return {"anchor": anchor, "train_bars": nt, "test_bars": ne, "n_syms": len(syms),
            "real_k": cell(real_k), "rw": cell(rw_k), "phase": cell(pr_k),
            "block": cell(bs_k), "eg": cell(eg_p)}


if __name__ == "__main__":
    rows = []
    for j, anc in enumerate(ANCHORS):
        print(f"[{j+1}/{len(ANCHORS)}] anchor {anc} ...", flush=True)
        r = screen_anchor(anc, seed=12345 + j)
        rows.append(r)
        def fmt(c):
            return f"{c['rate']:5.1f}% [{c['lo']:.0f},{c['hi']:.0f}] (n={c['n']})"
        print(f"    real-Kalman {fmt(r['real_k'])} | RW {fmt(r['rw'])} | "
              f"phase {fmt(r['phase'])} | block {fmt(r['block'])} || clean-EG {fmt(r['eg'])}", flush=True)

    # flat CSV
    out = []
    for r in rows:
        row = {"anchor": r["anchor"], "n_syms": r["n_syms"], "train_bars": r["train_bars"], "test_bars": r["test_bars"]}
        for key in ["real_k", "rw", "phase", "block", "eg"]:
            row[f"{key}_rate"] = r[key]["rate"]
            row[f"{key}_lo"] = r[key]["lo"]
            row[f"{key}_hi"] = r[key]["hi"]
            row[f"{key}_n"] = r[key]["n"]
        out.append(row)
    pd.DataFrame(out).to_csv(ROOT / "scratch/kalman_anchors.csv", index=False)
    print("\nwrote scratch/kalman_anchors.csv")
    # headline
    real_all = [r["real_k"]["rate"] for r in rows]
    plac_all = [max(r["rw"]["rate"], r["phase"]["rate"], r["block"]["rate"]) for r in rows]
    eg_all = [r["eg"]["rate"] for r in rows]
    print(f"\nAcross {len(rows)} anchors: real-Kalman p<0.05 rate {min(real_all):.0f}-{max(real_all):.0f}%, "
          f"placebo max {min(plac_all):.0f}-{max(plac_all):.0f}%, clean-EG {min(eg_all):.0f}-{max(eg_all):.0f}%.")
    print("Artifact holds at every anchor if real ~ placebo ~ high and EG is low.")
    print("DONE.")
