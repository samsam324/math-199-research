"""STRETCH: classify best-level size DECREASES as TRADE vs CANCEL by reconciling
snapshot bid/ask size drops with trade prints in the same 1s interval, then test
whether cancellation imbalance adds predictive info over trade-OFI + book-OFI.

Per second, on each side:
  size_decrease_total = sum of (negative) best-level size changes at same price
  traded_volume       = volume of prints that hit that side's best
  cancel_volume       = max(size_decrease_total - traded_volume, 0)
Cancel imbalance = bid_cancel - ask_cancel (more bid cancels = bearish pressure).
"""
import warnings, os
import numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings('ignore')
import sys; sys.path.insert(0, os.path.dirname(__file__))
import book_ofi_incremental as M

ROOT = M.ROOT; DATES = M.DATES


def build_day_cancel(sym, date):
    bk = M.load_book(sym, date)
    tr = M.load_trades(sym, date)
    if bk is None or tr is None:
        return None
    grid = pd.date_range(date + ' 00:00:00+00:00', date + ' 23:59:59+00:00', freq='1s')

    # per-snapshot best-level size deltas WHEN PRICE UNCHANGED (pure size moves)
    bp, bs, ap, asz = bk.bp.values, bk.bsz.values, bk.ap.values, bk.asz.values
    bid_same = np.zeros(len(bk)); ask_same = np.zeros(len(bk))
    bid_same[1:] = np.where(bp[1:] == bp[:-1], bs[1:] - bs[:-1], 0.0)
    ask_same[1:] = np.where(ap[1:] == ap[:-1], asz[1:] - asz[:-1], 0.0)
    bk = bk.assign(bid_dec=np.where(bid_same < 0, -bid_same, 0.0),
                   ask_dec=np.where(ask_same < 0, -ask_same, 0.0))
    g = bk.groupby('sec').agg(bid_dec=('bid_dec', 'sum'), ask_dec=('ask_dec', 'sum'),
                              book_ofi=('e', 'sum'), mid=('mid', 'last'))
    g = g.reindex(grid)
    g[['bid_dec', 'ask_dec', 'book_ofi']] = g[['bid_dec', 'ask_dec', 'book_ofi']].fillna(0.0)
    g['mid'] = g['mid'].ffill()

    # traded volume by side: sell aggressor hits bid; buy aggressor hits ask
    tr = tr.assign(buyvol=np.where(tr.side == 'buy', tr.amount, 0.0),
                   sellvol=np.where(tr.side == 'sell', tr.amount, 0.0))
    tg = tr.groupby('sec').agg(buyvol=('buyvol', 'sum'), sellvol=('sellvol', 'sum'),
                               sv=('sv', 'sum')).reindex(grid).fillna(0.0)

    out = pd.DataFrame(index=grid)
    # cancel volume = size decrease not explained by trades hitting that side
    out['bid_cancel'] = np.maximum(g.bid_dec.values - tg.sellvol.values, 0.0)
    out['ask_cancel'] = np.maximum(g.ask_dec.values - tg.buyvol.values, 0.0)
    out['cancel_imb'] = out['bid_cancel'] - out['ask_cancel']  # +ve => more bid cancels => bearish
    out['book_ofi'] = g.book_ofi.values
    out['trade_ofi'] = tg.sv.values
    out['bid_dec'] = g.bid_dec.values; out['ask_dec'] = g.ask_dec.values
    out['traded'] = tg.buyvol.values + tg.sellvol.values
    out['logmid'] = np.log(g.mid.values)
    out['date'] = date
    return out


def hac(y, Xcols, df, lags):
    sub = df[['_y'] + Xcols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 100: return None
    Xz = (sub[Xcols].values - sub[Xcols].values.mean(0)) / sub[Xcols].values.std(0)
    m = sm.OLS(sub['_y'].values, sm.add_constant(Xz)).fit(cov_type='HAC', cov_kwds={'maxlags': max(1, lags)})
    return m.rsquared, int(m.nobs), m.tvalues[1:]


def analyze(sym, trail=10):
    days = [build_day_cancel(sym, d) for d in DATES]
    days = [d for d in days if d is not None]
    df = pd.concat(days)
    # cancel-rate diagnostics: fraction of best-level size decreases that are cancels (not trades)
    tot_dec = df.bid_dec.sum() + df.ask_dec.sum()
    tot_cancel = df.bid_cancel.sum() + df.ask_cancel.sum()
    print(f"\n=== {sym} cancel-rate stretch ({len(days)} days) ===")
    print(f"  best-level size decrease attributed to CANCEL: {tot_cancel/tot_dec:.3f} "
          f"(TRADE: {1-tot_cancel/tot_dec:.3f})")

    parts = []
    for _, gg in df.groupby('date'):
        gg = gg.copy()
        for c in ['cancel_imb', 'book_ofi', 'trade_ofi']:
            gg[c + '_tr'] = gg[c].rolling(trail, min_periods=1).sum()
        lm = gg.logmid.values
        for h in M.HORIZONS:
            f = np.full(len(lm), np.nan); f[:-h] = lm[h:] - lm[:-h]; gg[f'fwd{h}'] = f
        parts.append(gg)
    df = pd.concat(parts)

    print(f"  Does cancel-imbalance add over trade+book OFI? (trail={trail}s, HAC)")
    print(f"  {'h':>3} | {'tr+bk R2':>9} | {'tr+bk+cancel R2':>15} | {'incr_cancel':>11} | {'t_cancel':>8}")
    for h in M.HORIZONS:
        df['_y'] = df[f'fwd{h}']
        base = hac(df['_y'], ['trade_ofi_tr', 'book_ofi_tr'], df, h)
        full = hac(df['_y'], ['trade_ofi_tr', 'book_ofi_tr', 'cancel_imb_tr'], df, h)
        if base and full:
            print(f"  {h:>3} | {base[0]:>9.5f} | {full[0]:>15.5f} | {full[0]-base[0]:>11.5f} | {full[2][-1]:>8.1f}")


if __name__ == '__main__':
    for s in M.SYMS:
        analyze(s)
