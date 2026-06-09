"""
Independent re-derivation (run #3) of the crypto pairs-trading walk-forward backtest.
Written from scratch. Only reads raw parquet files.
"""
import glob, os
import numpy as np
import pandas as pd

DATA_DIRS = {
    'binance': 'data/spot_1h',
    'coinbase': 'data/coinbase_1h',
}

TRAIN_H = 4380   # ~6 months
TEST_H = 2190    # ~3 months
START = '2021-01-01'
FFILL_LIMIT = 3
TOP_N = 20
Z_ENTRY = 2.0
Z_EXIT = 0.5
Z_STOP = 4.0
COST_RT_BPS = 30.0          # round-trip bps
COST_PER_UNIT = (COST_RT_BPS / 1e4) / 2.0   # charged per |pos change|; 15 bps * |dpos|


def common_universe():
    b = set(os.path.basename(f) for f in glob.glob(os.path.join(DATA_DIRS['binance'], '*.parquet')))
    c = set(os.path.basename(f) for f in glob.glob(os.path.join(DATA_DIRS['coinbase'], '*.parquet')))
    return sorted(x.replace('.parquet', '') for x in (b & c))


def build_panel(venue, symbols):
    """Build hourly close panel on continuous grid from START, ffill gaps <=3h."""
    series = {}
    last_ts = None
    for sym in symbols:
        df = pd.read_parquet(os.path.join(DATA_DIRS[venue], sym + '.parquet'), columns=['close'])
        s = df['close']
        if s.index.tz is None:
            s.index = s.index.tz_localize('UTC')
        else:
            s.index = s.index.tz_convert('UTC')
        s = s[~s.index.duplicated(keep='last')].sort_index()
        series[sym] = s
        mx = s.index.max()
        if last_ts is None or mx > last_ts:
            last_ts = mx
    grid = pd.date_range(start=pd.Timestamp(START, tz='UTC'), end=last_ts, freq='h')
    panel = pd.DataFrame(index=grid)
    for sym, s in series.items():
        panel[sym] = s.reindex(grid).ffill(limit=FFILL_LIMIT)
    return panel


def ols_ab(y, x):
    """OLS y = alpha + beta*x. Returns (alpha, beta)."""
    n = len(x)
    sx = x.sum(); sy = y.sum()
    sxx = (x * x).sum(); sxy = (x * y).sum()
    denom = n * sxx - sx * sx
    if denom == 0:
        return np.nan, np.nan
    beta = (n * sxy - sx * sy) / denom
    alpha = (sy - beta * sx) / n
    return alpha, beta


def ar1_phi(spread):
    """AR(1) lag-1 autocorr coefficient: regress s[t] on s[t-1] (mean-removed slope)."""
    s = spread - spread.mean()
    s0 = s[:-1]; s1 = s[1:]
    denom = (s0 * s0).sum()
    if denom == 0:
        return np.nan
    return (s0 * s1).sum() / denom


def run_venue(panel, symbols):
    logp = np.log(panel)
    n = len(panel)
    # walk-forward windows
    starts = list(range(0, n - TRAIN_H - TEST_H + 1, TEST_H))
    ret_nostop = []
    ret_stop = []

    for st in starts:
        tr0, tr1 = st, st + TRAIN_H
        te0, te1 = tr1, tr1 + TEST_H
        train = logp.iloc[tr0:tr1]
        test = logp.iloc[te0:te1]

        candidates = []  # (kappa, a, b, alpha, beta, mean, std)
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                a, b = symbols[i], symbols[j]
                la = train[a].values; lb = train[b].values
                m = np.isfinite(la) & np.isfinite(lb)
                if m.sum() < TRAIN_H * 0.5:
                    continue
                la_, lb_ = la[m], lb[m]
                alpha, beta = ols_ab(la_, lb_)
                if not np.isfinite(beta):
                    continue
                spread = la_ - (alpha + beta * lb_)
                if not np.all(np.isfinite(spread)) or spread.std() == 0:
                    continue
                phi = ar1_phi(spread)
                if not np.isfinite(phi) or not (0 < phi < 1):
                    continue
                kappa = -np.log(phi)
                candidates.append((kappa, a, b, alpha, beta, spread.mean(), spread.std()))

        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:TOP_N]

        # accumulate equal-weight pair returns over test window
        win_nostop = np.zeros(len(test))
        win_stop = np.zeros(len(test))
        cnt = 0
        for kappa, a, b, alpha, beta, mu, sd in top:
            la = test[a].values; lb = test[b].values
            spread = la - (alpha + beta * lb)
            if not np.all(np.isfinite(spread)):
                # fill non-finite spreads -> treat as flat (no trade) where NaN
                pass
            z = (spread - mu) / sd

            for variant, out in (('nostop', win_nostop), ('stop', win_stop)):
                pos = np.zeros(len(z))
                cur = 0.0
                for t in range(len(z)):
                    zt = z[t]
                    if not np.isfinite(zt):
                        # hold previous position if z undefined
                        pos[t] = cur
                        continue
                    if variant == 'stop' and abs(zt) >= Z_STOP:
                        cur = 0.0
                    elif zt >= Z_ENTRY:
                        cur = -1.0
                    elif zt <= -Z_ENTRY:
                        cur = 1.0
                    elif abs(zt) <= Z_EXIT:
                        cur = 0.0
                    # else hold
                    pos[t] = cur

                dspread = np.diff(spread, prepend=spread[0])
                dspread[0] = 0.0
                # strictly causal: pnl[t] = pos[t-1]*(spread[t]-spread[t-1])
                prevpos = np.concatenate([[0.0], pos[:-1]])
                pnl = prevpos * dspread
                pnl = np.nan_to_num(pnl, nan=0.0)
                dpos = np.abs(np.diff(pos, prepend=0.0))
                cost = COST_PER_UNIT * dpos
                net = pnl - cost
                out += np.nan_to_num(net, nan=0.0)
            cnt += 1

        if cnt > 0:
            ret_nostop.append(pd.Series(win_nostop / cnt, index=test.index))
            ret_stop.append(pd.Series(win_stop / cnt, index=test.index))

    rn = pd.concat(ret_nostop) if ret_nostop else pd.Series(dtype=float)
    rs = pd.concat(ret_stop) if ret_stop else pd.Series(dtype=float)
    return rn, rs


def monthly_sharpe(ret):
    if len(ret) == 0:
        return np.nan
    monthly = ret.resample('ME').sum()
    monthly = monthly[monthly != 0] if False else monthly  # keep all months
    mu = monthly.mean()
    sd = monthly.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return mu / sd * np.sqrt(12)


def main():
    symbols = common_universe()
    print(f"Universe: {len(symbols)} symbols")
    results = {}
    for venue in ('binance', 'coinbase'):
        panel = build_panel(venue, symbols)
        print(f"{venue}: panel shape {panel.shape}, span {panel.index[0]} -> {panel.index[-1]}")
        rn, rs = run_venue(panel, symbols)
        sn = monthly_sharpe(rn)
        ss = monthly_sharpe(rs)
        results[venue] = (sn, ss, len(rn))
        print(f"{venue}: no-stop monthly Sharpe = {sn:.4f}  | stop monthly Sharpe = {ss:.4f}  (bars={len(rn)})")

    bn = results['binance'][0]
    cn = results['coinbase'][0]
    bs = results['binance'][1]
    cs = results['coinbase'][1]
    ratio = cn / bn if bn else np.nan
    print("\n==== SUMMARY ====")
    print(f"Binance no-stop : {bn:.4f}")
    print(f"Coinbase no-stop: {cn:.4f}")
    print(f"Binance stop    : {bs:.4f}")
    print(f"Coinbase stop   : {cs:.4f}")
    print(f"Coinbase/Binance no-stop ratio: {ratio:.4f}")


if __name__ == '__main__':
    main()
