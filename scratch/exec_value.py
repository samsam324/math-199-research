"""
Execution-cost experiment: does an L3-from-L2 microstructure signal (book-OFI +
best-level cancellation imbalance) produce a real EXECUTION edge (bps) when used
to decide passive-post vs cross-aggressive?

Measures implementation shortfall (cost vs arrival mid, in bps) for three methods:
  1. AGGRESSIVE      cross immediately at t0, walk the book (VWAP fill).
  2. PASSIVE_NAIVE   post at best same-side price, queue model, cross at t0+H if
                     unfilled (H = 30s, 60s).
  3. PASSIVE_L3AWARE post passive only when trailing book-OFI / cancellation
                     imbalance says flow is non-adverse; else cross immediately.

Honesty variants: back-of-queue (pessimistic) and front-of-queue (optimistic).

Event-level raw data only. One (symbol, day) loaded at a time to bound memory.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = r"C:\Users\jackw\Desktop\math-199-research"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]
DATES = [f"2024-01-{d:02d}" for d in range(2, 10)]  # 2024-01-02 .. 2024-01-09 (8 days)
NOTIONALS = [10_000.0, 50_000.0]
HORIZONS_S = [30.0, 60.0]            # passive horizons (seconds)
CADENCE_S = 300.0                    # one parent order every 5 min per symbol-day
SIGNAL_WIN_S = 1.5                   # trailing window for book-OFI / cancellation signal
K_LEVELS = 25                        # book depth available
RNG_SEED = 12345

US = 1_000_000  # microseconds per second


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _book_path(sym, date):
    return os.path.join(ROOT, "data", "l2_raw", "binance", "book_snapshot_25", sym, date + ".csv.gz")


def _trade_path(sym, date):
    return os.path.join(ROOT, "data", "l2_raw", "binance", "trades", sym, date + ".csv.gz")


def load_book(sym, date):
    """Return dict of numpy arrays for the full K-level book, sorted by ts (us)."""
    p = _book_path(sym, date)
    if not os.path.exists(p):
        return None
    bid_px_c = [f"bids[{i}].price" for i in range(K_LEVELS)]
    bid_sz_c = [f"bids[{i}].amount" for i in range(K_LEVELS)]
    ask_px_c = [f"asks[{i}].price" for i in range(K_LEVELS)]
    ask_sz_c = [f"asks[{i}].amount" for i in range(K_LEVELS)]
    cols = ["timestamp"] + bid_px_c + bid_sz_c + ask_px_c + ask_sz_c
    df = pd.read_csv(p, usecols=cols)
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].to_numpy(np.int64)
    bid_px = df[bid_px_c].to_numpy(np.float64)
    bid_sz = df[bid_sz_c].to_numpy(np.float64)
    ask_px = df[ask_px_c].to_numpy(np.float64)
    ask_sz = df[ask_sz_c].to_numpy(np.float64)
    # keep only rows with a valid, uncrossed top of book
    good = (bid_px[:, 0] > 0) & (ask_px[:, 0] > 0) & (ask_px[:, 0] > bid_px[:, 0])
    return {
        "ts": ts[good],
        "bid_px": bid_px[good],
        "bid_sz": bid_sz[good],
        "ask_px": ask_px[good],
        "ask_sz": ask_sz[good],
    }


def load_trades(sym, date):
    """Return arrays sorted by ts (us): ts, side (+1 buy-aggressor, -1 sell), price, amount."""
    p = _trade_path(sym, date)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, usecols=["timestamp", "side", "price", "amount"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    sign = np.where(df["side"].to_numpy() == "buy", 1, -1).astype(np.int8)
    return {
        "ts": df["timestamp"].to_numpy(np.int64),
        "side": sign,
        "price": df["price"].to_numpy(np.float64),
        "amount": df["amount"].to_numpy(np.float64),
    }


# --------------------------------------------------------------------------- #
# Aggressive book walk: VWAP fill price for `notional_usd` on `side`
# --------------------------------------------------------------------------- #
def walk_book(px_row, sz_row, notional_usd):
    """Walk one side's levels to fill notional_usd. Return VWAP fill price.
    px_row/sz_row are length-K arrays (best level first)."""
    remaining = notional_usd
    cost = 0.0      # dollars spent
    units = 0.0     # base units acquired
    last_px = px_row[0]
    for lvl in range(len(px_row)):
        p, s = px_row[lvl], sz_row[lvl]
        if not np.isfinite(p) or p <= 0 or not np.isfinite(s) or s <= 0:
            continue
        last_px = p
        level_usd = p * s
        take = min(remaining, level_usd)
        u = take / p
        cost += u * p
        units += u
        remaining -= take
        if remaining <= 1e-9:
            break
    if remaining > 1e-9:
        # book too thin: fill residual at worst-seen level
        u = remaining / last_px
        cost += u * last_px
        units += u
    return cost / units  # VWAP


# --------------------------------------------------------------------------- #
# Book-OFI (CKS best-level) + cancellation imbalance over a trailing window
# --------------------------------------------------------------------------- #
def signals_over_window(book, i0, i1):
    """Compute (book_ofi, cancel_imbalance) from book snapshots in [i0, i1].

    book_ofi: sum of CKS best-level order-flow events e_n across the window.
      bid side: px up -> +sz ; px same -> +(sz-prev) ; px down -> -prev
      ask side: px down -> -sz; px same -> -(sz-prev); px up   -> +prev
      (ask adds push OFI negative). Positive book_ofi = net demand pressure.

    cancel_imbalance: (bid size REDUCTIONS at a stable best bid) minus
      (ask size reductions at a stable best ask), summed across the window,
      normalized by total reduction. Size pulled from the bid (negative number)
      = bid being cancelled (bearish for a resting buyer). We define
      cancel_imb = (ask_cancels - bid_cancels)/(ask_cancels+bid_cancels):
      positive => asks being pulled (supply withdrawing, supportive of a buy);
      negative => bids being pulled (demand withdrawing, adverse for a buy).
    """
    if i1 <= i0:
        return 0.0, 0.0
    bp = book["bid_px"][i0 : i1 + 1, 0]
    bs = book["bid_sz"][i0 : i1 + 1, 0]
    ap = book["ask_px"][i0 : i1 + 1, 0]
    asz = book["ask_sz"][i0 : i1 + 1, 0]
    pbp, pbs, cbp, cbs = bp[:-1], bs[:-1], bp[1:], bs[1:]
    pap, pas, cap, cas = ap[:-1], asz[:-1], ap[1:], asz[1:]

    db = np.where(cbp > pbp, cbs, np.where(cbp == pbp, cbs - pbs, -pbs))
    da = np.where(cap < pap, -cas, np.where(cap == pap, -(cas - pas), pas))
    book_ofi = float(np.sum(db + da))

    # cancellation imbalance: size REDUCTIONS while best price is UNCHANGED
    bid_red = np.where((cbp == pbp) & (cbs < pbs), pbs - cbs, 0.0).sum()
    ask_red = np.where((cap == pap) & (cas < pas), pas - cas, 0.0).sum()
    tot = bid_red + ask_red
    cancel_imb = float((ask_red - bid_red) / tot) if tot > 0 else 0.0
    return book_ofi, cancel_imb


# --------------------------------------------------------------------------- #
# Passive queue fill simulation
# --------------------------------------------------------------------------- #
def passive_fill(trades, t0, t_end, side, post_px, queue_ahead, order_units):
    """Walk the trade tape (t0, t_end]; for a BUY posted at bid `post_px`, count
    SELL-aggressor volume trading at price <= post_px (these execute against
    resting bids at our level or better). Filled once cumulative such volume
    >= queue_ahead + order_units. Symmetric for SELL.

    Returns (filled: bool, fill_ts_us: int or None).
    """
    need = queue_ahead + order_units
    if need <= 0:
        return True, t0
    lo = np.searchsorted(trades["ts"], t0, side="right")
    hi = np.searchsorted(trades["ts"], t_end, side="right")
    cum = 0.0
    if side == "buy":
        for j in range(lo, hi):
            if trades["side"][j] == -1 and trades["price"][j] <= post_px:
                cum += trades["amount"][j]
                if cum >= need:
                    return True, int(trades["ts"][j])
    else:  # sell posted at ask: count BUY-aggressor trades at price >= post_px
        for j in range(lo, hi):
            if trades["side"][j] == 1 and trades["price"][j] >= post_px:
                cum += trades["amount"][j]
                if cum >= need:
                    return True, int(trades["ts"][j])
    return False, None


def book_idx_at(book_ts, t):
    """Index of latest snapshot at-or-before t (us). -1 if none."""
    return int(np.searchsorted(book_ts, t, side="right")) - 1


# --------------------------------------------------------------------------- #
# Per (symbol, day) simulation
# --------------------------------------------------------------------------- #
def simulate_day(sym, date, rng):
    book = load_book(sym, date)
    trades = load_trades(sym, date)
    if book is None or trades is None or len(book["ts"]) < 100 or len(trades["ts"]) < 100:
        return []
    bts = book["ts"]
    day_start = bts[0]
    day_end = bts[-1]

    rows = []
    # parent order times on a fixed cadence; leave 60s tail so horizon fits
    t = day_start + int(60 * US)  # warm-up so trailing signal window has data
    end_cut = day_end - int(70 * US)
    while t <= end_cut:
        i0 = book_idx_at(bts, t)
        if i0 < 1:
            t += int(CADENCE_S * US)
            continue
        best_bid = book["bid_px"][i0, 0]
        best_ask = book["ask_px"][i0, 0]
        bid_sz0 = book["bid_sz"][i0, 0]
        ask_sz0 = book["ask_sz"][i0, 0]
        m0 = 0.5 * (best_bid + best_ask)
        if not np.isfinite(m0) or m0 <= 0 or best_ask <= best_bid:
            t += int(CADENCE_S * US)
            continue

        # randomized side per order (seeded)
        side = "buy" if rng.random() < 0.5 else "sell"
        sgn = 1.0 if side == "buy" else -1.0

        for notional in NOTIONALS:
            order_units = notional / m0

            # ---- 1) AGGRESSIVE at t0 ----
            if side == "buy":
                agg_vwap = walk_book(book["ask_px"][i0], book["ask_sz"][i0], notional)
            else:
                agg_vwap = walk_book(book["bid_px"][i0], book["bid_sz"][i0], notional)
            cost_agg = sgn * (agg_vwap - m0) / m0 * 1e4

            # posted price + queue ahead (back-of-queue uses displayed size)
            post_px = best_bid if side == "buy" else best_ask
            q_back = bid_sz0 if side == "buy" else ask_sz0

            # ---- 2) PASSIVE_NAIVE for each horizon, each queue assumption ----
            passive_results = {}  # (H, queue_mode) -> (cost_bps, filled, fill_dt_s)
            for H in HORIZONS_S:
                t_end = t + int(H * US)
                for qmode, q_ahead in (("back", q_back), ("front", 0.0)):
                    filled, fts = passive_fill(
                        trades, t, t_end, side, post_px, q_ahead, order_units
                    )
                    if filled:
                        # earned the spread: fill at posted price
                        cost = sgn * (post_px - m0) / m0 * 1e4
                        fill_dt = (fts - t) / US
                    else:
                        # cancel + cross aggressively at book as of t0+H
                        iH = book_idx_at(bts, t_end)
                        if iH < 0:
                            iH = i0
                        if side == "buy":
                            vwap_H = walk_book(book["ask_px"][iH], book["ask_sz"][iH], notional)
                        else:
                            vwap_H = walk_book(book["bid_px"][iH], book["bid_sz"][iH], notional)
                        cost = sgn * (vwap_H - m0) / m0 * 1e4
                        fill_dt = np.nan
                    passive_results[(H, qmode)] = (cost, filled, fill_dt)

            # ---- 3) PASSIVE_L3AWARE ----
            # trailing-window signal ending at t0
            t_sig0 = t - int(SIGNAL_WIN_S * US)
            j0 = book_idx_at(bts, t_sig0)
            if j0 < 0:
                j0 = 0
            book_ofi, cancel_imb = signals_over_window(book, j0, i0)
            # normalize book_ofi by recent typical |event| scale -> use sign + magnitude
            # Decision rule (to BUY): post passive only if flow NON-adverse:
            #   book_ofi >= 0  (net demand, price not pushing down through our bid)
            #   AND cancel_imb >= 0 (asks pulled more than bids; our bid not being abandoned)
            # To SELL: mirror (book_ofi <= 0 and cancel_imb <= 0).
            if side == "buy":
                post_ok = (book_ofi >= 0.0) and (cancel_imb >= 0.0)
            else:
                post_ok = (book_ofi <= 0.0) and (cancel_imb <= 0.0)

            l3_results = {}  # (H, qmode) -> (cost_bps, posted_passive, filled)
            for H in HORIZONS_S:
                for qmode in ("back", "front"):
                    if post_ok:
                        cost, filled, _ = passive_results[(H, qmode)]
                        l3_results[(H, qmode)] = (cost, True, filled)
                    else:
                        # adverse flow -> cross immediately (same as aggressive)
                        l3_results[(H, qmode)] = (cost_agg, False, True)

            rec = {
                "sym": sym, "date": date, "t0_us": t, "side": side,
                "notional": notional, "m0": m0,
                "spread_bps": (best_ask - best_bid) / m0 * 1e4,
                "cost_agg": cost_agg,
                "book_ofi": book_ofi, "cancel_imb": cancel_imb,
                "l3_posted": post_ok,
            }
            for H in HORIZONS_S:
                for qmode in ("back", "front"):
                    c, f, dt = passive_results[(H, qmode)]
                    tag = f"H{int(H)}_{qmode}"
                    rec[f"naive_{tag}_cost"] = c
                    rec[f"naive_{tag}_fill"] = int(f)
                    rec[f"naive_{tag}_dt"] = dt
                    lc, lp, lf = l3_results[(H, qmode)]
                    rec[f"l3_{tag}_cost"] = lc
            rows.append(rec)
        t += int(CADENCE_S * US)
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def dist(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan,) * 4
    return (np.mean(x), np.median(x), np.percentile(x, 90), np.percentile(x, 95))


def main():
    t_start = time.time()
    rng = np.random.default_rng(RNG_SEED)
    all_rows = []
    for sym in SYMBOLS:
        for date in DATES:
            r = simulate_day(sym, date, rng)
            all_rows.extend(r)
            print(f"  loaded {sym} {date}: {len(r)} order-rows "
                  f"(elapsed {time.time()-t_start:.0f}s)", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(ROOT, "scratch", "exec_value_orders.csv"), index=False)
    print(f"\nTotal order-rows: {len(df)}  (n unique order times x sizes)\n", flush=True)

    print("=" * 110)
    print("MAIN TABLE  (cost in bps vs arrival mid; positive = worse than mid). "
          "Back-of-queue (pessimistic) fills.")
    print("=" * 110)
    hdr = (f"{'sym':<8}{'size':>7} | {'AGG mean':>9} | "
           f"{'NAIVE30 mean/med/p90/p95 (fill%)':>40} | "
           f"{'NAIVE60 mean (fill%)':>20} | "
           f"{'L3-30 mean':>10} {'L3-60 mean':>10}")
    print(hdr)
    print("-" * 150)
    summary = []
    for sym in SYMBOLS:
        for notional in NOTIONALS:
            sub = df[(df["sym"] == sym) & (df["notional"] == notional)]
            if len(sub) == 0:
                continue
            agg_m = np.mean(sub["cost_agg"])
            n30 = sub["naive_H30_back_cost"]; n30f = sub["naive_H30_back_fill"].mean() * 100
            n60 = sub["naive_H60_back_cost"]; n60f = sub["naive_H60_back_fill"].mean() * 100
            l3_30 = sub["l3_H30_back_cost"]; l3_60 = sub["l3_H60_back_cost"]
            d30 = dist(n30)
            print(f"{sym:<8}{int(notional):>7} | {agg_m:>9.2f} | "
                  f"{d30[0]:>7.2f}/{d30[1]:>6.2f}/{d30[2]:>6.2f}/{d30[3]:>6.2f} ({n30f:>4.0f}%) | "
                  f"{np.mean(n60):>13.2f} ({n60f:>4.0f}%) | "
                  f"{np.mean(l3_30):>10.2f} {np.mean(l3_60):>10.2f}")
            summary.append(dict(
                sym=sym, size=int(notional), n=len(sub),
                agg=agg_m,
                naive30=np.mean(n30), naive30_fill=n30f,
                naive60=np.mean(n60), naive60_fill=n60f,
                l3_30=np.mean(l3_30), l3_60=np.mean(l3_60),
                naive30_med=d30[1], naive30_p90=d30[2], naive30_p95=d30[3],
                l3_posted_pct=sub["l3_posted"].mean() * 100,
                spread=np.mean(sub["spread_bps"]),
            ))

    sdf = pd.DataFrame(summary)
    print("\n" + "=" * 110)
    print("HEADLINE: bps saved by L3-aware (H=30s, back-of-queue)")
    print("  save_vs_naive = naive30 - l3_30  (>0 = L3 cheaper than naive passive: pure selection skill)")
    print("  save_vs_agg   = agg - l3_30      (>0 = L3 cheaper than always-cross)")
    print("=" * 110)
    print(f"{'sym':<8}{'size':>7} | {'AGG':>7} {'NAIVE30':>8} {'L3-30':>7} | "
          f"{'save_vs_naive':>13} {'save_vs_agg':>11} | {'L3 post%':>8} {'spread':>7}")
    print("-" * 90)
    for r in summary:
        print(f"{r['sym']:<8}{r['size']:>7} | {r['agg']:>7.2f} {r['naive30']:>8.2f} {r['l3_30']:>7.2f} | "
              f"{r['naive30']-r['l3_30']:>13.2f} {r['agg']-r['l3_30']:>11.2f} | "
              f"{r['l3_posted_pct']:>7.0f}% {r['spread']:>7.2f}")

    # ----- Honesty check: front-of-queue (optimistic) -----
    print("\n" + "=" * 110)
    print("HONESTY (i): FRONT-OF-QUEUE (optimistic, queue_ahead=0) -- does L3 saving survive?")
    print("=" * 110)
    print(f"{'sym':<8}{'size':>7} | {'NAIVE30(f)':>10} {'L3-30(f)':>9} {'save_vs_naive':>13} | "
          f"{'NAIVEfill%':>10}")
    print("-" * 70)
    for sym in SYMBOLS:
        for notional in NOTIONALS:
            sub = df[(df["sym"] == sym) & (df["notional"] == notional)]
            if len(sub) == 0:
                continue
            nf = np.mean(sub["naive_H30_front_cost"])
            lf = np.mean(sub["l3_H30_front_cost"])
            fillf = sub["naive_H30_front_fill"].mean() * 100
            print(f"{sym:<8}{int(notional):>7} | {nf:>10.2f} {lf:>9.2f} {nf-lf:>13.2f} | {fillf:>9.0f}%")

    # ----- Honesty check (ii): pooled selection skill -----
    print("\n" + "=" * 110)
    print("HONESTY (ii): pure selection skill. Both naive & L3 capture spread when filled;")
    print("  the gap is ONLY order selection. Pooled across all symbols/sizes (back-of-queue, H=30):")
    print("=" * 110)
    for size in NOTIONALS:
        sub = df[df["notional"] == size]
        agg = np.mean(sub["cost_agg"])
        nai = np.mean(sub["naive_H30_back_cost"])
        l3 = np.mean(sub["l3_H30_back_cost"])
        # restrict to orders where L3 chose to CROSS (post_ok False) -> did it avoid bad passive fills?
        crossed = sub[~sub["l3_posted"]]
        posted = sub[sub["l3_posted"]]
        print(f"\n  size ${int(size):,}  (n={len(sub)}):")
        print(f"    pooled mean cost  AGG={agg:6.2f}  NAIVE={nai:6.2f}  L3={l3:6.2f}  "
              f"| L3 save vs naive={nai-l3:6.2f}  vs agg={agg-l3:6.2f}")
        if len(crossed) and len(posted):
            # on orders L3 crossed, what would naive passive have cost?
            nai_on_crossed = np.mean(crossed["naive_H30_back_cost"])
            agg_on_crossed = np.mean(crossed["cost_agg"])
            nai_on_posted = np.mean(posted["naive_H30_back_cost"])
            print(f"    L3 CROSSED {len(crossed)} orders ({len(crossed)/len(sub)*100:.0f}%): "
                  f"naive-passive would have cost {nai_on_crossed:6.2f} there, agg cost {agg_on_crossed:6.2f} "
                  f"-> selection {'AVOIDED' if nai_on_crossed>agg_on_crossed else 'HURT'} "
                  f"{nai_on_crossed-agg_on_crossed:+.2f} bps")
            print(f"    L3 POSTED  {len(posted)} orders ({len(posted)/len(sub)*100:.0f}%): "
                  f"naive-passive cost {nai_on_posted:6.2f} there")

    # ----- Honesty check (iii): PLACEBO -- is the signal better than a coin flip? -----
    # L3 mixes passive (when signal says safe) and aggressive (when adverse). A naive
    # gap vs always-passive could be purely mechanical (aggressive is cheaper). The
    # real test: does the SIGNAL beat a RANDOM post/cross decision at the SAME post-rate?
    print("\n" + "=" * 110)
    print("HONESTY (iii): PLACEBO -- L3 signal vs RANDOM post/cross at the SAME post-rate.")
    print("  If L3 has real selection skill it should beat the random placebo (lower cost).")
    print("=" * 110)
    pl_rng = np.random.default_rng(7)
    for size in NOTIONALS:
        sub = df[df["notional"] == size]
        p_post = sub["l3_posted"].mean()
        naive_c = sub["naive_H30_back_cost"].to_numpy()
        agg_c = sub["cost_agg"].to_numpy()
        sims = []
        for _ in range(300):
            mask = pl_rng.random(len(sub)) < p_post
            sims.append(np.where(mask, naive_c, agg_c).mean())
        rand_mean, rand_sd = float(np.mean(sims)), float(np.std(sims))
        l3 = float(sub["l3_H30_back_cost"].mean())
        z = (l3 - rand_mean) / rand_sd if rand_sd > 0 else float("nan")
        verdict = "BEATS placebo" if l3 < rand_mean else "WORSE than placebo"
        print(f"  size ${int(size):>6,}: L3={l3:6.3f}  random-same-rate placebo={rand_mean:6.3f}"
              f" (sd={rand_sd:.3f})  edge={rand_mean - l3:+.3f} bps (z={z:+.2f})  -> {verdict}")
        # does the signal flag the genuinely-worse passive orders to cross?
        posted = sub[sub["l3_posted"]]["naive_H30_back_cost"].mean()
        crossed = sub[~sub["l3_posted"]]["naive_H30_back_cost"].mean()
        good = crossed > posted
        print(f"             naive-passive cost on L3-POSTED={posted:5.2f} vs L3-CROSSED={crossed:5.2f}"
              f"  -> signal flags worse-passive orders to cross? {good}")

    sdf.to_csv(os.path.join(ROOT, "scratch", "exec_value_summary.csv"), index=False)
    print(f"\nDone in {time.time()-t_start:.0f}s. Wrote exec_value_orders.csv, exec_value_summary.csv")


if __name__ == "__main__":
    main()
