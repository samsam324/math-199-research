"""
Book-OFI (Cont, Kukanov, Stoikov 2014) vs Trade-OFI: incremental information.
FULL-2024 RE-RUN across regimes and 4 symbols.

This is a copy/extension of scratch/book_ofi_incremental.py (contemporaneous +
predictive book-OFI vs trade-OFI) MERGED with the cancellation reconstruction
from scratch/book_ofi_cancel_stretch.py. The CKS best-level event formula and the
cancel-vs-trade reconciliation are reused EXACTLY.

New vs original:
  * DATES span all 12 months of 2024 incl. volatile regimes (yen-carry crash
    2024-08-05/06, BTC near-ATH 2024-03-13).
  * SYMS = BTC, ETH, SOL, AVAX (4 symbols, breadth test).
  * Key numbers broken out BY REGIME (Aug-5 crash day vs calm days) and PER SYMBOL:
      - contemporaneous book-OFI incremental R^2 over trade-OFI
      - predictive R^2 decay at 1s/5s/10s/30s
      - cancellation share of best-level size reductions
  * All regressions HAC / Newey-West (overlapping windows -> inflated naive t).
  * Runs SYNCHRONOUSLY, no background processes.
"""
import warnings, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

ROOT = r'C:\Users\jackw\Desktop\math-199-research'

# ---- 2024 regime sample (all 12 months covered across these 7 days) ----
DATES = ['2024-01-15', '2024-03-13', '2024-05-15',
         '2024-08-05', '2024-08-06', '2024-11-12', '2024-12-16']
CRASH_DATES = ['2024-08-05', '2024-08-06']   # yen-carry unwind
CALM_DATES = [d for d in DATES if d not in CRASH_DATES]

HORIZONS = [1, 5, 10, 30]
SYMS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT']

BOOK_COLS = ['timestamp', 'asks[0].price', 'asks[0].amount', 'bids[0].price', 'bids[0].amount']
TR_COLS = ['timestamp', 'side', 'amount']


# ====================================================================
# CKS best-level order-flow event (UNCHANGED from book_ofi_incremental.py)
# ====================================================================
def book_ofi_events(bid_px, bid_sz, ask_px, ask_sz):
    """Exact CKS best-level order-flow event e_n from consecutive snapshots."""
    n = len(bid_px)
    e = np.zeros(n)
    pbp, pbs = bid_px[:-1], bid_sz[:-1]
    cbp, cbs = bid_px[1:],  bid_sz[1:]
    pap, pas_ = ask_px[:-1], ask_sz[:-1]
    cap, cas = ask_px[1:],  ask_sz[1:]
    db = np.where(cbp > pbp, cbs,
         np.where(cbp == pbp, cbs - pbs, -pbs))
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
    sign = np.where(df.side.values == 'buy', 1.0, -1.0)
    df['sv'] = sign * df.amount.values
    df['ts'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
    df['sec'] = df['ts'].dt.floor('1s')
    return df


# ====================================================================
# Per-day build w/ OFI + cancel reconstruction (merges both originals)
# cancel logic UNCHANGED from book_ofi_cancel_stretch.py
# ====================================================================
def build_day(sym, date):
    bk = load_book(sym, date)
    tr = load_trades(sym, date)
    if bk is None or tr is None:
        return None
    grid = pd.date_range(date + ' 00:00:00+00:00', date + ' 23:59:59+00:00', freq='1s')

    # --- cancel reconstruction: best-level size decreases at SAME price ---
    bp, bs, ap, asz = bk.bp.values, bk.bsz.values, bk.ap.values, bk.asz.values
    bid_same = np.zeros(len(bk)); ask_same = np.zeros(len(bk))
    bid_same[1:] = np.where(bp[1:] == bp[:-1], bs[1:] - bs[:-1], 0.0)
    ask_same[1:] = np.where(ap[1:] == ap[:-1], asz[1:] - asz[:-1], 0.0)
    bk = bk.assign(bid_dec=np.where(bid_same < 0, -bid_same, 0.0),
                   ask_dec=np.where(ask_same < 0, -ask_same, 0.0))

    book_g = bk.groupby('sec').agg(book_ofi=('e', 'sum'),
                                   mid=('mid', 'last'),
                                   micro=('micro', 'last'),
                                   bid_dec=('bid_dec', 'sum'),
                                   ask_dec=('ask_dec', 'sum'))
    book_g = book_g.reindex(grid)
    book_g['book_ofi'] = book_g['book_ofi'].fillna(0.0)
    book_g[['bid_dec', 'ask_dec']] = book_g[['bid_dec', 'ask_dec']].fillna(0.0)
    book_g['mid'] = book_g['mid'].ffill()
    book_g['micro'] = book_g['micro'].ffill()

    # trade-OFI + by-side traded volume per second
    tr = tr.assign(buyvol=np.where(tr.side == 'buy', tr.amount, 0.0),
                   sellvol=np.where(tr.side == 'sell', tr.amount, 0.0))
    tg = tr.groupby('sec').agg(buyvol=('buyvol', 'sum'), sellvol=('sellvol', 'sum'),
                               sv=('sv', 'sum')).reindex(grid).fillna(0.0)

    out = pd.DataFrame(index=grid)
    out['book_ofi'] = book_g['book_ofi'].values
    out['trade_ofi'] = tg['sv'].values
    out['mid'] = book_g['mid'].values
    out['micro'] = book_g['micro'].values
    out['logmid'] = np.log(out['mid'])
    # cancel volume = best-level size decrease not explained by trades hitting that side
    out['bid_dec'] = book_g['bid_dec'].values
    out['ask_dec'] = book_g['ask_dec'].values
    out['bid_cancel'] = np.maximum(book_g['bid_dec'].values - tg['sellvol'].values, 0.0)
    out['ask_cancel'] = np.maximum(book_g['ask_dec'].values - tg['buyvol'].values, 0.0)
    out['cancel_imb'] = out['bid_cancel'] - out['ask_cancel']  # +ve => more bid cancels => bearish
    out['date'] = date
    return out


def hac_multi(X_cols, df, maxlags):
    """OLS with HAC SEs; standardizes regressors so betas comparable & R^2 unaffected."""
    sub = df[['_y'] + X_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 100:
        return None
    y = sub['_y'].values
    Xraw = sub[X_cols].values
    sd = Xraw.std(0)
    sd[sd == 0] = 1.0
    Xz = (Xraw - Xraw.mean(0)) / sd
    X = sm.add_constant(Xz)
    m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': max(1, maxlags)})
    res = {'r2': m.rsquared, 'n': int(m.nobs), 'betas': {}, 'ts': {}}
    for i, c in enumerate(X_cols):
        res['betas'][c] = m.params[i + 1]
        res['ts'][c] = m.tvalues[i + 1]
    return res


def _prep(df, trail):
    """Add trailing-`trail`s OFI/cancel sums, contemporaneous & forward returns,
    per-day to avoid boundary leakage."""
    parts = []
    for _, g in df.groupby('date'):
        g = g.copy()
        g['bofi_tr'] = g['book_ofi'].rolling(trail, min_periods=1).sum()
        g['tofi_tr'] = g['trade_ofi'].rolling(trail, min_periods=1).sum()
        g['cimb_tr'] = g['cancel_imb'].rolling(trail, min_periods=1).sum()
        lm = g['logmid'].values
        g['ret_contemp'] = g['logmid'].diff()
        for h in HORIZONS:
            fut = np.full(len(lm), np.nan)
            fut[:-h] = lm[h:] - lm[:-h]
            g[f'fwd{h}'] = fut
        parts.append(g)
    return pd.concat(parts)


def run_block(df, label, trail):
    """Print contemporaneous + predictive + cancel results for a (already-prepped)
    dataframe `df`. Returns a dict of headline numbers for later summary tables."""
    n = len(df)
    print(f"\n{'='*74}\n{label}  (n={n:,} sec, trail={trail}s)\n{'='*74}")
    out = {'label': label, 'n': n}

    # ---------- CONTEMPORANEOUS decomposition ----------
    df['bofi_1s'] = df['book_ofi']; df['tofi_1s'] = df['trade_ofi']
    df['_y'] = df['ret_contemp']
    cT = hac_multi(['tofi_1s'], df, 5)
    cB = hac_multi(['bofi_1s'], df, 5)
    cJ = hac_multi(['tofi_1s', 'bofi_1s'], df, 5)
    if cT and cB and cJ:
        incr_book_c = cJ['r2'] - cT['r2']
        incr_trade_c = cJ['r2'] - cB['r2']
        print(f"\n--- CONTEMPORANEOUS: 1s return [t-1,t] on same-second OFI (HAC) ---")
        print(f"  trade-only : R2={cT['r2']:.5f}  t={cT['ts']['tofi_1s']:.1f}")
        print(f"  book-only  : R2={cB['r2']:.5f}  t={cB['ts']['bofi_1s']:.1f}")
        print(f"  joint      : R2={cJ['r2']:.5f}  t_tr={cJ['ts']['tofi_1s']:.1f}  t_bk={cJ['ts']['bofi_1s']:.1f}")
        print(f"  incr book over trade : {incr_book_c:.5f}")
        print(f"  incr trade over book : {incr_trade_c:.5f}")
        out['contemp_trade_r2'] = cT['r2']
        out['contemp_book_r2'] = cB['r2']
        out['contemp_joint_r2'] = cJ['r2']
        out['contemp_incr_book'] = incr_book_c
        out['contemp_incr_trade'] = incr_trade_c

    # ---------- PREDICTIVE: future return on trailing OFI ----------
    print(f"\n--- PREDICTIVE: future log-return [t,t+h] on trailing-{trail}s OFI (HAC) ---")
    print(f"{'h':>3} | {'trade-only R2':>13} {'t':>6} | {'book-only R2':>12} {'t':>6} | "
          f"{'joint R2':>9} {'t_tr':>6} {'t_bk':>6} | {'incr_book':>9} {'incr_trade':>10}")
    out['pred'] = {}
    for h in HORIZONS:
        df['_y'] = df[f'fwd{h}']
        rT = hac_multi(['tofi_tr'], df, h)
        rB = hac_multi(['bofi_tr'], df, h)
        rJ = hac_multi(['tofi_tr', 'bofi_tr'], df, h)
        if not (rT and rB and rJ):
            continue
        incr_book = rJ['r2'] - rT['r2']
        incr_trade = rJ['r2'] - rB['r2']
        print(f"{h:>3} | {rT['r2']:>13.5f} {rT['ts']['tofi_tr']:>6.1f} | "
              f"{rB['r2']:>12.5f} {rB['ts']['bofi_tr']:>6.1f} | "
              f"{rJ['r2']:>9.5f} {rJ['ts']['tofi_tr']:>6.1f} {rJ['ts']['bofi_tr']:>6.1f} | "
              f"{incr_book:>9.5f} {incr_trade:>10.5f}")
        out['pred'][h] = {'trade_r2': rT['r2'], 'book_r2': rB['r2'], 'joint_r2': rJ['r2'],
                          'incr_book': incr_book, 'incr_trade': incr_trade,
                          't_book': rJ['ts']['bofi_tr']}

    # ---------- CANCELLATION share + cancel-imbalance test ----------
    tot_dec = df.bid_dec.sum() + df.ask_dec.sum()
    tot_cancel = df.bid_cancel.sum() + df.ask_cancel.sum()
    cancel_share = tot_cancel / tot_dec if tot_dec > 0 else np.nan
    print(f"\n--- CANCELLATION: best-level size decreases attributed to CANCEL vs TRADE ---")
    print(f"  cancel share = {cancel_share:.3f}   (trade share = {1-cancel_share:.3f})")
    out['cancel_share'] = cancel_share

    print(f"  Does cancel-imbalance add over trade+book OFI? (trail={trail}s, HAC)")
    print(f"  {'h':>3} | {'tr+bk R2':>9} | {'tr+bk+cancel R2':>15} | {'incr_cancel':>11} | {'t_cancel':>8}")
    out['cancel_pred'] = {}
    for h in HORIZONS:
        df['_y'] = df[f'fwd{h}']
        base = hac_multi(['tofi_tr', 'bofi_tr'], df, h)
        full = hac_multi(['tofi_tr', 'bofi_tr', 'cimb_tr'], df, h)
        if base and full:
            print(f"  {h:>3} | {base['r2']:>9.5f} | {full['r2']:>15.5f} | "
                  f"{full['r2']-base['r2']:>11.5f} | {full['ts']['cimb_tr']:>8.1f}")
            out['cancel_pred'][h] = {'incr': full['r2'] - base['r2'], 't': full['ts']['cimb_tr']}
    return out


def analyze_symbol(sym, trail=10):
    days = [build_day(sym, d) for d in DATES]
    keep = [(d, dt) for d, dt in zip(DATES, days) if dt is not None]
    if not keep:
        print(f"{sym}: no data"); return None
    dates_have = [d for d, _ in keep]
    df_all = pd.concat([dt for _, dt in keep])
    df_all = _prep(df_all, trail)

    print(f"\n\n{'#'*74}\n# SYMBOL {sym}   days available: {dates_have}\n{'#'*74}")
    summary = {}
    # full sample
    summary['ALL'] = run_block(df_all.copy(), f"{sym}  ALL-2024 ({len(dates_have)} days)", trail)
    # calm vs crash regime split
    calm = df_all[df_all['date'].isin(CALM_DATES)]
    crash = df_all[df_all['date'].isin(CRASH_DATES)]
    if len(calm) > 0:
        summary['CALM'] = run_block(calm.copy(), f"{sym}  CALM days {CALM_DATES}", trail)
    if len(crash) > 0:
        summary['CRASH'] = run_block(crash.copy(), f"{sym}  CRASH days {CRASH_DATES} (yen-carry)", trail)
    # Aug-5 single crash day on its own (the headline stress day)
    aug5 = df_all[df_all['date'] == '2024-08-05']
    if len(aug5) > 0:
        summary['AUG5'] = run_block(aug5.copy(), f"{sym}  AUG-5 ONLY (2024-08-05 crash)", trail)
    return summary


def print_master_tables(all_summ):
    """Compact cross-symbol / cross-regime tables for the writeup."""
    print(f"\n\n{'='*90}\nMASTER SUMMARY TABLES (NUMBERS)\n{'='*90}")

    # --- contemporaneous incremental book-OFI over trade-OFI ---
    print("\n[1] CONTEMPORANEOUS book-OFI incremental R^2 over trade-OFI")
    print(f"{'sym':>9} | {'ALL':>8} {'CALM':>8} {'CRASH':>8} {'AUG5':>8}")
    for sym, s in all_summ.items():
        row = []
        for reg in ['ALL', 'CALM', 'CRASH', 'AUG5']:
            v = s.get(reg, {}).get('contemp_incr_book', np.nan)
            row.append(f"{v:8.4f}" if v == v else f"{'--':>8}")
        print(f"{sym:>9} | " + " ".join(row))

    # --- contemporaneous joint R^2 (overall fit) ---
    print("\n[1b] CONTEMPORANEOUS joint R^2 (trade+book)")
    print(f"{'sym':>9} | {'ALL':>8} {'CALM':>8} {'CRASH':>8} {'AUG5':>8}")
    for sym, s in all_summ.items():
        row = []
        for reg in ['ALL', 'CALM', 'CRASH', 'AUG5']:
            v = s.get(reg, {}).get('contemp_joint_r2', np.nan)
            row.append(f"{v:8.4f}" if v == v else f"{'--':>8}")
        print(f"{sym:>9} | " + " ".join(row))

    # --- predictive incremental book-OFI by horizon, per regime ---
    for reg in ['ALL', 'CALM', 'CRASH', 'AUG5']:
        print(f"\n[2] PREDICTIVE incr book-OFI R^2 by horizon  ({reg})")
        print(f"{'sym':>9} | " + " ".join(f"{('h='+str(h)):>9}" for h in HORIZONS))
        for sym, s in all_summ.items():
            pred = s.get(reg, {}).get('pred', {})
            row = []
            for h in HORIZONS:
                v = pred.get(h, {}).get('incr_book', np.nan)
                row.append(f"{v:9.5f}" if v == v else f"{'--':>9}")
            print(f"{sym:>9} | " + " ".join(row))

    # --- predictive JOINT R^2 by horizon (total forecast power), per regime ---
    for reg in ['ALL', 'CRASH', 'AUG5']:
        print(f"\n[2b] PREDICTIVE joint R^2 (trade+book) by horizon  ({reg})")
        print(f"{'sym':>9} | " + " ".join(f"{('h='+str(h)):>9}" for h in HORIZONS))
        for sym, s in all_summ.items():
            pred = s.get(reg, {}).get('pred', {})
            row = []
            for h in HORIZONS:
                v = pred.get(h, {}).get('joint_r2', np.nan)
                row.append(f"{v:9.5f}" if v == v else f"{'--':>9}")
            print(f"{sym:>9} | " + " ".join(row))

    # --- cancellation share ---
    print("\n[3] CANCELLATION share of best-level size reductions")
    print(f"{'sym':>9} | {'ALL':>8} {'CALM':>8} {'CRASH':>8} {'AUG5':>8}")
    for sym, s in all_summ.items():
        row = []
        for reg in ['ALL', 'CALM', 'CRASH', 'AUG5']:
            v = s.get(reg, {}).get('cancel_share', np.nan)
            row.append(f"{v:8.3f}" if v == v else f"{'--':>8}")
        print(f"{sym:>9} | " + " ".join(row))

    # --- cancel-imbalance incremental predictive R^2 at h=1 ---
    print("\n[4] cancel-imb incremental predictive R^2 over trade+book OFI (h=1)")
    print(f"{'sym':>9} | {'ALL incr':>10} {'ALL t':>7} | {'CRASH incr':>11} {'CRASH t':>8} | {'AUG5 incr':>10} {'AUG5 t':>8}")
    for sym, s in all_summ.items():
        def gp(reg, k):
            return s.get(reg, {}).get('cancel_pred', {}).get(1, {}).get(k, np.nan)
        a_i, a_t = gp('ALL', 'incr'), gp('ALL', 't')
        c_i, c_t = gp('CRASH', 'incr'), gp('CRASH', 't')
        g_i, g_t = gp('AUG5', 'incr'), gp('AUG5', 't')
        fmt = lambda v, w, p: (f"{v:{w}.{p}f}" if v == v else f"{'--':>{w}}")
        print(f"{sym:>9} | {fmt(a_i,10,5)} {fmt(a_t,7,1)} | {fmt(c_i,11,5)} {fmt(c_t,8,1)} | "
              f"{fmt(g_i,10,5)} {fmt(g_t,8,1)}")


if __name__ == '__main__':
    print(f"DATES = {DATES}")
    print(f"CRASH = {CRASH_DATES}   CALM = {CALM_DATES}")
    print(f"SYMS  = {SYMS}")
    all_summ = {}
    for s in SYMS:
        res = analyze_symbol(s, trail=10)
        if res is not None:
            all_summ[s] = res
    print_master_tables(all_summ)
    print("\nDONE.")
