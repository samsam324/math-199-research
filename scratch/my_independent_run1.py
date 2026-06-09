"""
INDEPENDENT re-derivation of crypto pairs-trading walk-forward backtest.
Written from scratch. Only reads raw parquet files.
"""
import os
import numpy as np
import pandas as pd
from itertools import combinations

START = "2021-01-01"
TRAIN_H = 4380   # ~6 months
TEST_H = 2190    # ~3 months
FFILL_LIMIT = 3
TOP_N = 20
Z_ENTRY = 2.0
Z_EXIT = 0.5
Z_STOP = 4.0
COST_RT = 0.0030  # 30 bps round trip; charged as 15 bps * |dpos|

def load_panel(folder, symbols):
    """Load 'close' for each symbol, reindex to continuous hourly grid from START, ffill<=3."""
    series = {}
    for sym in symbols:
        df = pd.read_parquet(os.path.join(folder, sym + ".parquet"))
        s = df["close"].copy()
        s.index = pd.to_datetime(s.index, utc=True)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        series[sym] = s
    panel = pd.DataFrame(series)
    # continuous hourly grid from START to max timestamp seen
    start = pd.Timestamp(START, tz="UTC")
    end = panel.index.max()
    grid = pd.date_range(start=start, end=end, freq="1h", tz="UTC")
    panel = panel.reindex(grid)
    panel = panel.ffill(limit=FFILL_LIMIT)
    return panel

def ar1_phi(x):
    """lag-1 autocorr coefficient via OLS of x[t] on x[t-1] (with intercept)."""
    x = np.asarray(x, dtype=float)
    x0 = x[:-1]
    x1 = x[1:]
    m = np.isfinite(x0) & np.isfinite(x1)
    if m.sum() < 30:
        return np.nan
    x0 = x0[m]; x1 = x1[m]
    # OLS slope of x1 ~ a + phi*x0
    xm = x0.mean(); ym = x1.mean()
    denom = np.sum((x0 - xm) ** 2)
    if denom <= 0:
        return np.nan
    phi = np.sum((x0 - xm) * (x1 - ym)) / denom
    return phi

def ols_ab(y, x):
    """OLS y ~ alpha + beta*x. Return alpha, beta."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 50:
        return np.nan, np.nan
    y = y[m]; x = x[m]
    xm = x.mean(); ym = y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom <= 0:
        return np.nan, np.nan
    beta = np.sum((x - xm) * (y - ym)) / denom
    alpha = ym - beta * xm
    return alpha, beta

def positions_from_z(z, use_stop):
    """Stateful entry/exit. Returns position array aligned to z (current-bar desired position)."""
    n = len(z)
    pos = np.zeros(n)
    cur = 0.0
    for t in range(n):
        zt = z[t]
        if not np.isfinite(zt):
            # hold previous position if z unavailable
            pos[t] = cur
            continue
        if use_stop and abs(zt) >= Z_STOP:
            cur = 0.0
        elif cur == 0.0:
            if zt >= Z_ENTRY:
                cur = -1.0
            elif zt <= -Z_ENTRY:
                cur = 1.0
        else:
            # in a position: exit when |z|<=0.5
            if abs(zt) <= Z_EXIT:
                cur = 0.0
            elif use_stop and abs(zt) >= Z_STOP:
                cur = 0.0
        pos[t] = cur
    return pos

def run_backtest(panel):
    logp = np.log(panel)
    symbols = list(panel.columns)
    n = len(panel)
    idx = panel.index

    results = {"nostop": [], "stop": []}  # list of per-window equal-weighted return Series

    starts = list(range(0, n - TRAIN_H - TEST_H + 1, TEST_H))
    for s0 in starts:
        tr_lo = s0
        tr_hi = s0 + TRAIN_H
        te_lo = tr_hi
        te_hi = min(tr_hi + TEST_H, n)
        if te_hi - te_lo < 50:
            continue

        train = logp.iloc[tr_lo:tr_hi]
        test = logp.iloc[te_lo:te_hi]

        # build candidate pairs on TRAIN only
        cands = []
        for a, b in combinations(symbols, 2):
            la_tr = train[a].values
            lb_tr = train[b].values
            m = np.isfinite(la_tr) & np.isfinite(lb_tr)
            if m.sum() < TRAIN_H * 0.8:
                continue
            alpha, beta = ols_ab(la_tr, lb_tr)
            if not np.isfinite(beta):
                continue
            spread_tr = la_tr - (alpha + beta * lb_tr)
            phi = ar1_phi(spread_tr)
            if not np.isfinite(phi) or not (0.0 < phi < 1.0):
                continue
            kappa = -np.log(phi)
            sp_valid = spread_tr[np.isfinite(spread_tr)]
            mu = sp_valid.mean()
            sd = sp_valid.std(ddof=1)
            if not np.isfinite(sd) or sd <= 0:
                continue
            cands.append((kappa, a, b, alpha, beta, mu, sd))

        if not cands:
            continue
        cands.sort(key=lambda r: r[0], reverse=True)
        selected = cands[:TOP_N]

        # On TEST: compute pnl per pair for both variants
        win_nostop = []
        win_stop = []
        for kappa, a, b, alpha, beta, mu, sd in selected:
            la_te = test[a].values
            lb_te = test[b].values
            spread_te = la_te - (alpha + beta * lb_te)
            z = (spread_te - mu) / sd
            dspread = np.diff(spread_te, prepend=np.nan)  # spread[t]-spread[t-1]

            for variant, store in (("nostop", win_nostop), ("stop", win_stop)):
                use_stop = (variant == "stop")
                pos = positions_from_z(z, use_stop)
                # strictly causal: pnl[t] = pos[t-1]*(spread[t]-spread[t-1])
                pos_prev = np.concatenate([[0.0], pos[:-1]])
                gross = pos_prev * dspread
                gross = np.where(np.isfinite(gross), gross, 0.0)
                # costs on position change
                dpos = np.abs(np.diff(pos, prepend=0.0))
                cost = (COST_RT / 2.0) * dpos
                net = gross - cost
                store.append(pd.Series(net, index=test.index))

        if win_nostop:
            ew_ns = pd.concat(win_nostop, axis=1).mean(axis=1)
            results["nostop"].append(ew_ns)
        if win_stop:
            ew_st = pd.concat(win_stop, axis=1).mean(axis=1)
            results["stop"].append(ew_st)

    out = {}
    for variant in ("nostop", "stop"):
        if not results[variant]:
            out[variant] = np.nan
            continue
        full = pd.concat(results[variant]).sort_index()
        # in case of overlap (shouldn't, windows non-overlapping) keep
        full = full.groupby(level=0).sum()
        monthly = full.resample("MS").sum()
        monthly = monthly[np.isfinite(monthly)]
        mu = monthly.mean()
        sd = monthly.std(ddof=1)
        sharpe = (mu / sd) * np.sqrt(12) if sd > 0 else np.nan
        out[variant] = sharpe
    return out

if __name__ == "__main__":
    binance = set(f for f in os.listdir("data/spot_1h") if f.endswith(".parquet"))
    coinbase = set(f for f in os.listdir("data/coinbase_1h") if f.endswith(".parquet"))
    syms = sorted(f.replace(".parquet", "") for f in (binance & coinbase))
    print("Universe:", len(syms), "symbols")

    print("\n=== BINANCE ===")
    pb = load_panel("data/spot_1h", syms)
    print("panel shape", pb.shape, "from", pb.index.min(), "to", pb.index.max())
    rb = run_backtest(pb)
    print("Binance no-stop monthly Sharpe:", rb["nostop"])
    print("Binance stop|z|=4 monthly Sharpe:", rb["stop"])

    print("\n=== COINBASE ===")
    pc = load_panel("data/coinbase_1h", syms)
    print("panel shape", pc.shape, "from", pc.index.min(), "to", pc.index.max())
    rc = run_backtest(pc)
    print("Coinbase no-stop monthly Sharpe:", rc["nostop"])
    print("Coinbase stop|z|=4 monthly Sharpe:", rc["stop"])

    ratio = rc["nostop"] / rb["nostop"]
    print("\nCoinbase/Binance no-stop ratio:", ratio)
