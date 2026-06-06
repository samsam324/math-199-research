"""
Book-OFI (Cont, Kukanov, Stoikov 2014) vs Trade-OFI: incremental information.

Computes BOOK-side order flow imbalance from RAW per-update L2 snapshots
(book_snapshot_25) using the exact CKS best-level event formula, aggregates
to a 1s grid, and tests whether it carries information BEYOND trade flow.

Key metric: INCREMENTAL R^2 of book-OFI over trade-OFI in a joint model
(and vice versa), plus the contemporaneous decomposition.
"""
import warnings, os, gzip
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\jackw\Desktop\math-199-research'
DATES = [f'2024-01-{d:02d}' for d in range(2, 7)]   # 2024-01-02 .. 2024-01-06 (5 days)
HORIZONS = [1, 5, 10, 30]
SYMS = ['BTCUSDT', 'ETHUSDT']

BOOK_COLS = ['timestamp', 'asks[0].price', 'asks[0].amount', 'bids[0].price', 'bids[0].amount']
TR_COLS = ['timestamp', 'side', 'amount']


def book_ofi_events(bid_px, bid_sz, ask_px, ask_sz):
    """Exact CKS best-level order-flow event e_n from consecutive snapshots.

    Bid side (demand): bid_px up  -> +bid_sz (new size posted)
                       bid_px same-> +(bid_sz - prev_bid_sz)   (size change)
                       bid_px down -> -prev_bid_sz             (level wiped)
    Ask side (supply): ask_px down -> -ask_sz                  (adds = negative OFI)
                       ask_px same -> -(ask_sz - prev_ask_sz)
                       ask_px up   -> +prev_ask_sz             (level wiped)
    e_n = bid_term + ask_term.  (Ask ADDS push OFI negative.)
    """
    n = len(bid_px)
    e = np.zeros(n)
    pbp, pbs = bid_px[:-1], bid_sz[:-1]
    cbp, cbs = bid_px[1:],  bid_sz[1:]
    pap, pas_ = ask_px[:-1], ask_sz[:-1]
    cap, cas = ask_px[1:],  ask_sz[1:]

    # bid contribution
    db = np.where(cbp > pbp, cbs,
         np.where(cbp == pbp, cbs - pbs, -pbs))
    # ask contribution (note signs: ask add => negative)
    da = np.where(cap < pap, -cas,
         np.where(cap == pap, -(cas - pas_), pas_))
    e[1:] = db + da
    return e


def load_book(sym, date):
    p = os.path.join(ROOT, 'data', 'l2_raw', 'binance', 'book_snapshot_25', sym, date + '.csv.gz')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, usecols=BOOK_COLS)
    df = df.rename(columns={'asks[0].price': 'ap', 'asks[0].amount': 'asz',
                            'bids[0].price': 'bp', 'bids[0].amount': 'bsz'})
    # drop crossed/empty
    df = df[(df.ap > 0) & (df.bp > 0) & (df.ap > df.bp)].reset_index(drop=True)
    df['mid'] = (df.bp + df.ap) / 2.0
    den = df.bsz + df.asz
    df['micro'] = np.where(den > 0, (df.bsz * df.ap + df.asz * df.bp) / den, df['mid'])
    e = book_ofi_events(df.bp.values, df.bsz.values, df.ap.values, df.asz.values)
    df['e'] = e
    df['ts'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
    df['sec'] = df['ts'].dt.floor('1s')
    return df


def load_trades(sym, date):
    p = os.path.join(ROOT, 'data', 'l2_raw', 'binance', 'trades', sym, date + '.csv.gz')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, usecols=TR_COLS)
    # side = aggressor side. buy = market buy lifts ask -> +ve trade-OFI
    sign = np.where(df.side.values == 'buy', 1.0, -1.0)
    df['sv'] = sign * df.amount.values
    df['ts'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
    df['sec'] = df['ts'].dt.floor('1s')
    return df


def build_day(sym, date):
    bk = load_book(sym, date)
    tr = load_trades(sym, date)
    if bk is None or tr is None:
        return None
    grid = pd.date_range(date + ' 00:00:00+00:00', date + ' 23:59:59+00:00', freq='1s')

    # book-OFI per second = sum of events; mid/micro = last snapshot in second
    book_g = bk.groupby('sec').agg(book_ofi=('e', 'sum'),
                                   mid=('mid', 'last'),
                                   micro=('micro', 'last'))
    book_g = book_g.reindex(grid)
    book_g['book_ofi'] = book_g['book_ofi'].fillna(0.0)
    book_g['mid'] = book_g['mid'].ffill()
    book_g['micro'] = book_g['micro'].ffill()

    # trade-OFI per second
    trade_g = tr.groupby('sec')['sv'].sum().reindex(grid).fillna(0.0)

    out = pd.DataFrame(index=grid)
    out['book_ofi'] = book_g['book_ofi'].values
    out['trade_ofi'] = trade_g.values
    out['mid'] = book_g['mid'].values
    out['micro'] = book_g['micro'].values
    out['logmid'] = np.log(out['mid'])
    out['date'] = date
    return out


def hac_multi(y, X_cols, df, maxlags):
    """OLS with HAC SEs. X_cols: list of column names. Returns dict with
    per-regressor (beta,t), R^2, n. Standardizes regressors so betas comparable."""
    sub = df[['_y'] + X_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 100:
        return None
    y = sub['_y'].values
    Xraw = sub[X_cols].values
    # standardize regressors (z-score) so multi-regressor betas are comparable & R^2 unaffected
    Xz = (Xraw - Xraw.mean(0)) / Xraw.std(0)
    X = sm.add_constant(Xz)
    m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': max(1, maxlags)})
    res = {'r2': m.rsquared, 'n': int(m.nobs), 'betas': {}, 'ts': {}}
    for i, c in enumerate(X_cols):
        res['betas'][c] = m.params[i + 1]
        res['ts'][c] = m.tvalues[i + 1]
    return res


def analyze(sym, trail=10):
    days = [build_day(sym, d) for d in DATES]
    days = [d for d in days if d is not None]
    if not days:
        print(f"{sym}: no data"); return
    df = pd.concat(days)
    n_days = len(days)

    # trailing-`trail`s sums of OFI (per-day to avoid boundary leak)
    parts = []
    for _, g in df.groupby('date'):
        g = g.copy()
        g['bofi_tr'] = g['book_ofi'].rolling(trail, min_periods=1).sum()
        g['tofi_tr'] = g['trade_ofi'].rolling(trail, min_periods=1).sum()
        lm = g['logmid'].values
        # contemporaneous 1s return [t-1, t] paired with same-second flow
        g['ret_contemp'] = g['logmid'].diff()
        for h in HORIZONS:
            fut = np.full(len(lm), np.nan)
            fut[:-h] = lm[h:] - lm[:-h]
            g[f'fwd{h}'] = fut
        parts.append(g)
    df = pd.concat(parts)

    print(f"\n{'='*70}\n{sym}  ({n_days} days, {DATES[0]}..{DATES[-1]}, n={len(df):,} sec, trail={trail}s)\n{'='*70}")

    # ---------- PREDICTIVE: future return on OFI ----------
    print(f"\n--- PREDICTIVE: future log-return [t,t+h] on trailing-{trail}s OFI (HAC) ---")
    print(f"{'h':>3} | {'trade-only R2':>13} {'t':>6} | {'book-only R2':>12} {'t':>6} | "
          f"{'joint R2':>9} {'t_tr':>6} {'t_bk':>6} | {'incr_book':>9} {'incr_trade':>10}")
    pred_rows = []
    for h in HORIZONS:
        df['_y'] = df[f'fwd{h}']
        rT = hac_multi(df['_y'], ['tofi_tr'], df, h)
        rB = hac_multi(df['_y'], ['bofi_tr'], df, h)
        rJ = hac_multi(df['_y'], ['tofi_tr', 'bofi_tr'], df, h)
        incr_book = rJ['r2'] - rT['r2']
        incr_trade = rJ['r2'] - rB['r2']
        print(f"{h:>3} | {rT['r2']:>13.5f} {rT['ts']['tofi_tr']:>6.1f} | "
              f"{rB['r2']:>12.5f} {rB['ts']['bofi_tr']:>6.1f} | "
              f"{rJ['r2']:>9.5f} {rJ['ts']['tofi_tr']:>6.1f} {rJ['ts']['bofi_tr']:>6.1f} | "
              f"{incr_book:>9.5f} {incr_trade:>10.5f}")
        pred_rows.append((h, rT['r2'], rB['r2'], rJ['r2'], incr_book, incr_trade))

    # ---------- CONTEMPORANEOUS decomposition ----------
    # same-second OFI (not trailing) vs the [t-1,t] move it accompanies
    df['bofi_1s'] = df['book_ofi']
    df['tofi_1s'] = df['trade_ofi']
    df['_y'] = df['ret_contemp']
    cT = hac_multi(df['_y'], ['tofi_1s'], df, 5)
    cB = hac_multi(df['_y'], ['bofi_1s'], df, 5)
    cJ = hac_multi(df['_y'], ['tofi_1s', 'bofi_1s'], df, 5)
    print(f"\n--- CONTEMPORANEOUS: 1s return [t-1,t] on same-second OFI (HAC) ---")
    print(f"  trade-only : R2={cT['r2']:.5f}  t={cT['ts']['tofi_1s']:.1f}")
    print(f"  book-only  : R2={cB['r2']:.5f}  t={cB['ts']['bofi_1s']:.1f}")
    print(f"  joint      : R2={cJ['r2']:.5f}  t_tr={cJ['ts']['tofi_1s']:.1f}  t_bk={cJ['ts']['bofi_1s']:.1f}")
    print(f"  incr book over trade : {cJ['r2']-cT['r2']:.5f}")
    print(f"  incr trade over book : {cJ['r2']-cB['r2']:.5f}")

    return pred_rows, (cT['r2'], cB['r2'], cJ['r2'])


if __name__ == '__main__':
    for s in SYMS:
        analyze(s, trail=10)
