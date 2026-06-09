"""
Independent re-derivation (run #2) of crypto pairs-trading walk-forward backtest.
Written from scratch. Reads only raw parquet files.
"""
import os
import numpy as np
import pandas as pd
from itertools import combinations

BASE = r'C:/Users/jackw/Desktop/math-199-research'
BIN = os.path.join(BASE, 'data', 'spot_1h')
CB = os.path.join(BASE, 'data', 'coinbase_1h')

TRAIN = 4380   # ~6 months hourly
TEST = 2190    # ~3 months hourly
TOPN = 20
Z_ENTRY = 2.0
Z_EXIT = 0.5
Z_STOP = 4.0
RT_COST = 0.0030          # 30 bps round trip
HALF_COST = RT_COST / 2   # charged per unit |pos change|, doubled across full round trip

START = pd.Timestamp('2021-01-01', tz='UTC')


def common_symbols():
    a = set(f for f in os.listdir(BIN) if f.endswith('.parquet'))
    b = set(f for f in os.listdir(CB) if f.endswith('.parquet'))
    return sorted(f[:-8] for f in (a & b))


def build_panel(folder, syms):
    """Load close prices, reindex to continuous hourly grid from START, ffill <=3 hrs."""
    series = {}
    maxend = None
    for s in syms:
        df = pd.read_parquet(os.path.join(folder, s + '.parquet'), columns=['close'])
        c = df['close']
        # ensure UTC tz-aware, sorted, unique
        if c.index.tz is None:
            c.index = c.index.tz_localize('UTC')
        c = c[~c.index.duplicated(keep='last')].sort_index()
        series[s] = c
        e = c.index.max()
        if maxend is None or e > maxend:
            maxend = e
    grid = pd.date_range(START, maxend, freq='1h', tz='UTC')
    panel = pd.DataFrame(index=grid)
    for s in syms:
        c = series[s].reindex(grid)
        # forward-fill gaps up to 3 hours (3 consecutive missing bars)
        c = c.ffill(limit=3)
        panel[s] = c
    return panel


def ols_beta_alpha(y, x):
    """OLS y ~ alpha + beta*x. Returns (alpha, beta)."""
    n = len(x)
    xm = x.mean()
    ym = y.mean()
    xc = x - xm
    denom = np.dot(xc, xc)
    if denom <= 0:
        return None
    beta = np.dot(xc, (y - ym)) / denom
    alpha = ym - beta * xm
    return alpha, beta


def ar1_phi(spread):
    """AR(1) lag-1 autocorr coefficient: regress s[t] on s[t-1] (with intercept)."""
    s = np.asarray(spread, dtype=float)
    s0 = s[:-1]
    s1 = s[1:]
    res = ols_beta_alpha(s1, s0)
    if res is None:
        return None
    _, phi = res
    return phi


def run_variant(panel, use_stop):
    """Returns a concatenated test return series (pd.Series) of equal-weight portfolio returns."""
    n = len(panel)
    logp = np.log(panel)
    syms = list(panel.columns)
    idx = panel.index

    all_returns = []  # list of pd.Series of portfolio per-bar returns over each test window

    start = 0
    while start + TRAIN + TEST <= n:
        tr_lo, tr_hi = start, start + TRAIN
        te_lo, te_hi = start + TRAIN, start + TRAIN + TEST

        train_log = logp.iloc[tr_lo:tr_hi]
        test_log = logp.iloc[te_lo:te_hi]

        # candidate pairs evaluated on TRAIN
        cands = []
        for a, b in combinations(syms, 2):
            la = train_log[a].values
            lb = train_log[b].values
            mask = np.isfinite(la) & np.isfinite(lb)
            if mask.sum() < TRAIN * 0.8:  # need enough overlapping train data
                continue
            ya = la[mask]
            yb = lb[mask]
            res = ols_beta_alpha(ya, yb)
            if res is None:
                continue
            alpha, beta = res
            spread_tr = ya - (alpha + beta * yb)
            if not np.all(np.isfinite(spread_tr)) or spread_tr.std() == 0:
                continue
            phi = ar1_phi(spread_tr)
            if phi is None or not (0 < phi < 1):
                continue
            kappa = -np.log(phi)
            mu = spread_tr.mean()
            sd = spread_tr.std(ddof=1)
            if sd <= 0 or not np.isfinite(sd):
                continue
            cands.append((kappa, a, b, alpha, beta, mu, sd))

        if not cands:
            start += TEST
            continue

        cands.sort(key=lambda r: r[0], reverse=True)
        selected = cands[:TOPN]

        # build per-pair test returns
        pair_rets = []
        for kappa, a, b, alpha, beta, mu, sd in selected:
            la = test_log[a].values
            lb = test_log[b].values
            spread = la - (alpha + beta * lb)
            z = (spread - mu) / sd

            T = len(spread)
            pos = np.zeros(T)
            cur = 0.0
            for t in range(T):
                zt = z[t]
                if not np.isfinite(zt):
                    # hold previous position when z undefined
                    pos[t] = cur
                    continue
                if cur == 0.0:
                    if zt >= Z_ENTRY:
                        cur = -1.0
                    elif zt <= -Z_ENTRY:
                        cur = 1.0
                else:
                    # in a position: check exit / stop
                    if abs(zt) <= Z_EXIT:
                        cur = 0.0
                    elif use_stop and abs(zt) >= Z_STOP:
                        cur = 0.0
                pos[t] = cur

            # per-bar pnl: pos[t-1] * (spread[t]-spread[t-1]), strictly causal
            dspread = np.diff(spread, prepend=np.nan)  # dspread[t] = spread[t]-spread[t-1]
            prev_pos = np.concatenate(([0.0], pos[:-1]))
            pnl = prev_pos * dspread
            pnl[~np.isfinite(pnl)] = 0.0

            # costs on |position change|: HALF_COST * |pos[t]-pos[t-1]|
            dpos = np.abs(np.diff(pos, prepend=0.0))
            cost = HALF_COST * dpos
            ret = pnl - cost
            pair_rets.append(ret)

        pair_mat = np.vstack(pair_rets)            # (npairs, T)
        port = pair_mat.mean(axis=0)               # equal weight across pairs
        all_returns.append(pd.Series(port, index=test_log.index))

        start += TEST

    if not all_returns:
        return pd.Series(dtype=float)
    return pd.concat(all_returns)


def monthly_sharpe(ret_series):
    if len(ret_series) == 0:
        return np.nan
    monthly = ret_series.resample('ME').sum()
    monthly = monthly[monthly != 0] if False else monthly  # keep all months
    m = monthly.mean()
    s = monthly.std(ddof=1)
    if s == 0 or not np.isfinite(s):
        return np.nan
    return m / s * np.sqrt(12)


def main():
    syms = common_symbols()
    print(f'Common symbols: {len(syms)}')

    results = {}
    for venue, folder in [('Binance', BIN), ('Coinbase', CB)]:
        panel = build_panel(folder, syms)
        print(f'{venue} panel: {panel.shape}, range {panel.index[0]} .. {panel.index[-1]}')
        for stop in [False, True]:
            ret = run_variant(panel, use_stop=stop)
            sh = monthly_sharpe(ret)
            nmonths = len(ret.resample('ME').sum()) if len(ret) else 0
            key = f'{venue}_{"stop" if stop else "nostop"}'
            results[key] = sh
            print(f'  {key}: monthly Sharpe = {sh:.4f}  (months={nmonths}, total ret pts={len(ret)})')

    print('\n=== SUMMARY ===')
    for k, v in results.items():
        print(f'{k}: {v:.4f}')
    bn = results['Binance_nostop']
    cb = results['Coinbase_nostop']
    print(f'Coinbase/Binance no-stop ratio: {cb/bn:.4f}')

    import json
    print('JSONRESULT ' + json.dumps(results))


if __name__ == '__main__':
    main()
