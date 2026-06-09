"""
H1 task — STEP 1: pair selection (leakage-safe), done BEFORE touching the held-out tail.

Substitution (per docs/REMOTE_TASKS.md): the locked spec's "top 10 Kalman-cointegrated
pairs by OOS ADF p-value" is a known whitening artifact (loop 6). Replace with the
cleanly-selected pairs (scratch/clean_coint_pairs.csv, Engle-Granger cointegrated) ranked
by OOS REVERSION SPEED = OU kappa (the metric loop 7 validated, Spearman rho=0.46).

Leakage control: the H1 held-out tail is the LAST 20% of the 2024 1s L2 range. Selection
uses ONLY hourly closes (data/spot_1h) from 2021-01 up to the held-out boundary
(2024-10-19), strictly before the tail. OU kappa via AR(1) on the OLS-hedged log spread
(same machinery as scratch/persistence_test.ou_kappa_halflife / wf_backtest).

Outputs scratch/h1_selected_pairs.csv: sym_a, sym_b, beta_a_on_b, kappa, halflife_h, n_obs.

Run:  python scratch/h1_select.py
"""
import os, numpy as np, pandas as pd

ROOT = r"C:\Users\jackw\Desktop\math-199-research"
SPOT = os.path.join(ROOT, "data", "spot_1h")
HELDOUT_BOUNDARY = pd.Timestamp("2024-10-19T19:12:00Z")  # ~0.8 of the 2024 range; selection uses < this
TOP_N = 10
MIN_HL_H, MAX_HL_H = 2.0, 720.0   # sane reversion band (exclude sub-hourly noise / non-reverting)
EG_MAX = 0.05                     # "cleanly-selected" = genuinely Engle-Granger cointegrated (p<0.05)
# USD-pegged stablecoins: near-constant series give spurious "fast reversion"; not tradeable legs.
STABLES = {"USDCUSDT", "DAIUSDT", "TUSDUSDT", "USD1USDT", "FDUSDUSDT", "USDUCUSDT", "PAXGUSDT", "USDUSDT"}


def load_close(sym):
    f = os.path.join(SPOT, f"{sym}.parquet")
    if not os.path.exists(f):
        return None
    s = pd.read_parquet(f, columns=["close"])["close"]
    s.index = pd.to_datetime(s.index, utc=True)
    return s[~s.index.duplicated(keep="last")].sort_index()


def ou_kappa_halflife(spread):
    """AR(1): s_t = c + phi*s_{t-1} + e. kappa=-ln(phi); half-life=ln2/kappa (bars=hours)."""
    s = spread[np.isfinite(spread)]
    if len(s) < 500:
        return np.nan, np.nan
    s0, s1 = s[:-1], s[1:]
    X = np.column_stack([np.ones_like(s0), s0])
    coef, *_ = np.linalg.lstsq(X, s1, rcond=None)
    phi = coef[1]
    if not np.isfinite(phi) or phi <= 0 or phi >= 1:
        return np.nan, np.nan
    kappa = -np.log(phi)
    return kappa, np.log(2) / kappa


def main():
    pairs = pd.read_csv(os.path.join(ROOT, "scratch", "clean_coint_pairs.csv"))
    cache = {}
    def cl(sym):
        if sym not in cache:
            cache[sym] = load_close(sym)
        return cache[sym]

    rows = []
    pairs = pairs[pairs["eg_pvalue"] < EG_MAX].copy()   # genuinely cointegrated subset
    for _, r in pairs.iterrows():
        a, b = r["sym_a"].upper(), r["sym_b"].upper()
        if a in STABLES or b in STABLES:                 # drop stablecoin legs
            continue
        ca, cb = cl(a), cl(b)
        if ca is None or cb is None:
            continue
        df = pd.concat([np.log(ca).rename("a"), np.log(cb).rename("b")], axis=1).dropna()
        df = df[df.index < HELDOUT_BOUNDARY]      # strictly before the held-out tail
        if len(df) < 2000:
            continue
        X = np.column_stack([np.ones(len(df)), df["b"].values])
        coef, *_ = np.linalg.lstsq(X, df["a"].values, rcond=None)
        alpha, beta = coef
        if beta <= 0:                              # require co-moving hedge
            continue
        spread = df["a"].values - alpha - beta * df["b"].values
        kappa, hl = ou_kappa_halflife(spread)
        if not np.isfinite(kappa) or not (MIN_HL_H <= hl <= MAX_HL_H):
            continue
        rows.append(dict(sym_a=a, sym_b=b, eg_pvalue=r["eg_pvalue"],
                         alpha=alpha, beta_a_on_b=beta, kappa=kappa, halflife_h=hl, n_obs=len(df)))

    res = pd.DataFrame(rows).sort_values("kappa", ascending=False).reset_index(drop=True)
    top = res.head(TOP_N).copy()
    out = os.path.join(ROOT, "scratch", "h1_selected_pairs.csv")
    top.to_csv(out, index=False)
    print(f"Candidates scored: {len(res)}  | held-out boundary (selection uses data BEFORE this): {HELDOUT_BOUNDARY}")
    print(f"\nTOP {TOP_N} clean pairs by OU reversion speed (kappa), pre-tail hourly:")
    print(top[["sym_a","sym_b","kappa","halflife_h","beta_a_on_b","eg_pvalue","n_obs"]].to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
