"""
Execution-cost experiment, FULL-2024 RE-RUN (larger sample of the Jan-2024 null).

Identical simulation to scratch/exec_value.py, but DATES now sample all 12 months of
2024 INCLUDING volatile regimes (notably 2024-08-05 yen-carry-unwind crash, 03-13 BTC
near ATH). Adds the verification logic from scratch/exec_value_verify.py (oracle +
sign-flip + per-feature correlation) AND a per-month / regime (crash vs calm) breakdown,
all in one synchronous run.

Does an L3-from-L2 microstructure signal (book-OFI + best-level cancellation imbalance)
produce a real EXECUTION edge (bps) when used to decide passive-post vs cross-aggressive?

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
# Full-2024 sample across all 12 months, deliberately including volatile periods.
# 2024-08-05 = yen-carry-unwind crash; 2024-03-13 = BTC near ATH.
DATES = [
    "2024-01-15", "2024-02-15", "2024-03-13", "2024-04-16", "2024-05-15",
    "2024-06-14", "2024-07-15", "2024-08-05", "2024-08-06", "2024-09-16",
    "2024-10-15", "2024-11-12", "2024-12-16",
]
CRASH_DATES = {"2024-08-05"}  # the high-vol crash day to break out separately
NOTIONALS = [10_000.0, 50_000.0]
HORIZONS_S = [30.0, 60.0]            # passive horizons (seconds)
CADENCE_S = 300.0                    # one parent order every 5 min per symbol-day
SIGNAL_WIN_S = 1.5                   # trailing window for book-OFI / cancellation signal
K_LEVELS = 25                        # book depth available
RNG_SEED = 12345

US = 1_000_000  # microseconds per second

ORDERS_CSV = os.path.join(ROOT, "scratch", "exec_value_2024_orders.csv")
SUMMARY_CSV = os.path.join(ROOT, "scratch", "exec_value_2024_summary.csv")


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
    """Compute (book_ofi, cancel_imbalance) from book snapshots in [i0, i1]."""
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
    """Walk the trade tape (t0, t_end]; fill once enough opposing aggressor volume
    trades through our posted price. Returns (filled: bool, fill_ts_us: int or None)."""
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
                        cost = sgn * (post_px - m0) / m0 * 1e4
                        fill_dt = (fts - t) / US
                    else:
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
            t_sig0 = t - int(SIGNAL_WIN_S * US)
            j0 = book_idx_at(bts, t_sig0)
            if j0 < 0:
                j0 = 0
            book_ofi, cancel_imb = signals_over_window(book, j0, i0)
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
                        l3_results[(H, qmode)] = (cost_agg, False, True)

            rec = {
                "sym": sym, "date": date, "t0_us": t, "side": side,
                "regime": "crash" if date in CRASH_DATES else "calm",
                "month": date[:7],
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
# Reporting helpers
# --------------------------------------------------------------------------- #
def dist(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan,) * 4
    return (np.mean(x), np.median(x), np.percentile(x, 90), np.percentile(x, 95))


NAIVE = "naive_H30_back_cost"
AGG = "cost_agg"
L3 = "l3_H30_back_cost"


def placebo_mean_sd(sub, p_post, rng, n=2000):
    naive_c = sub[NAIVE].to_numpy(); agg_c = sub[AGG].to_numpy()
    sims = np.empty(n)
    for k in range(n):
        mask = rng.random(len(sub)) < p_post
        sims[k] = np.where(mask, naive_c, agg_c).mean()
    return float(sims.mean()), float(sims.std())


def block_metrics(sub, rng, label):
    """Print the headline pooled metrics + oracle + sign-flip + signal corr for a slice."""
    if len(sub) < 30:
        print(f"\n--- {label}: n={len(sub)} (too small, skipped) ---")
        return None
    agg = sub[AGG].mean()
    naive = sub[NAIVE].mean()
    l3 = sub[L3].mean()
    posted = sub["l3_posted"].astype(bool)
    p_post = posted.mean()
    l3_flip = np.where(~posted, sub[NAIVE], sub[AGG]).mean()
    oracle = np.minimum(sub[NAIVE], sub[AGG]).mean()
    antior = np.maximum(sub[NAIVE], sub[AGG]).mean()
    rmean, rsd = placebo_mean_sd(sub, p_post, rng)
    edge = rmean - l3                      # >0 => L3 beats random placebo
    z_l3 = (l3 - rmean) / rsd if rsd > 0 else np.nan
    sig_dir = sub[AGG].copy()  # placeholder; recompute below from book_ofi
    sd = sub["book_ofi"] * np.where(sub["side"] == "buy", 1.0, -1.0)
    pma = sub[NAIVE] - sub[AGG]
    c = pd.DataFrame({"sd": sd, "pma": pma}).dropna()
    corr = np.corrcoef(c["sd"], c["pma"])[0, 1] if len(c) > 30 else np.nan
    fill = sub["naive_H30_back_fill"].mean() * 100
    spread = sub["spread_bps"].mean()
    print(f"\n--- {label}  (n={len(sub)}, post-rate={p_post*100:.0f}%, "
          f"fill%={fill:.0f}, spread={spread:.2f}bps) ---")
    print(f"    AGG={agg:7.3f}  NAIVE={naive:7.3f}  L3={l3:7.3f}  "
          f"L3flip={l3_flip:7.3f}  ORACLE={oracle:7.3f}  ANTI={antior:7.3f}")
    print(f"    placebo={rmean:7.3f}(sd{rsd:.3f})  L3-edge_vs_placebo={edge:+.3f}bps "
          f"(z={z_l3:+.2f})  oracle_saves_vs_agg={agg-oracle:+.3f}bps")
    print(f"    corr(book-OFI dir, passive-minus-agg)={corr:+.4f}")
    return dict(label=label, n=len(sub), post_rate=p_post*100, fill_pct=fill,
                spread=spread, agg=agg, naive=naive, l3=l3, l3_flip=l3_flip,
                oracle=oracle, anti=antior, placebo=rmean, placebo_sd=rsd,
                edge_vs_placebo=edge, z_l3=z_l3, oracle_save_vs_agg=agg-oracle,
                corr_sig=corr)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
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
    df.to_csv(ORDERS_CSV, index=False)
    print(f"\nTotal order-rows: {len(df)}  across {df['date'].nunique()} dates, "
          f"{df['sym'].nunique()} symbols\n", flush=True)

    # ===================================================================== #
    # 1) MAIN TABLE per symbol/size (back-of-queue, pessimistic)
    # ===================================================================== #
    print("=" * 110)
    print("MAIN TABLE  (cost in bps vs arrival mid; positive=worse). Back-of-queue fills. ALL 2024.")
    print("=" * 110)
    print(f"{'sym':<8}{'size':>7} | {'AGG':>7} | "
          f"{'NAIVE30 mean/med/p90/p95 (fill%)':>40} | {'NAIVE60 (fill%)':>18} | "
          f"{'L3-30':>7} {'L3-60':>7}")
    print("-" * 130)
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
            print(f"{sym:<8}{int(notional):>7} | {agg_m:>7.2f} | "
                  f"{d30[0]:>7.2f}/{d30[1]:>6.2f}/{d30[2]:>6.2f}/{d30[3]:>6.2f} ({n30f:>4.0f}%) | "
                  f"{np.mean(n60):>11.2f} ({n60f:>4.0f}%) | "
                  f"{np.mean(l3_30):>7.2f} {np.mean(l3_60):>7.2f}")
            summary.append(dict(
                sym=sym, size=int(notional), n=len(sub), agg=agg_m,
                naive30=np.mean(n30), naive30_fill=n30f,
                naive60=np.mean(n60), naive60_fill=n60f,
                l3_30=np.mean(l3_30), l3_60=np.mean(l3_60),
                l3_posted_pct=sub["l3_posted"].mean() * 100,
                spread=np.mean(sub["spread_bps"]),
            ))
    pd.DataFrame(summary).to_csv(SUMMARY_CSV, index=False)

    # ===================================================================== #
    # 2) POOLED (all 2024) headline + verification (oracle/sign-flip/corr)
    # ===================================================================== #
    print("\n" + "=" * 110)
    print("POOLED (ALL 2024) HEADLINE + VERIFICATION  (H=30s, back-of-queue, lower=better)")
    print("  edge_vs_placebo>0 => L3 signal beats a random post/cross at same post-rate (real skill)")
    print("  oracle_saves_vs_agg ~0 => NO post/cross rule can beat just crossing")
    print("  corr ~0 => book-OFI signal carries no execution-timing info")
    print("=" * 110)
    vrng = np.random.default_rng(7)
    for size in [10000, 50000, "POOLED"]:
        sub = df if size == "POOLED" else df[df["notional"] == size]
        lbl = "POOLED (both sizes)" if size == "POOLED" else f"${int(size):,}"
        block_metrics(sub, vrng, lbl)

    # ===================================================================== #
    # 3) PLACEBO (matches original script's honesty (iii)) pooled by size
    # ===================================================================== #
    print("\n" + "=" * 110)
    print("PLACEBO check (L3 vs random post/cross at same post-rate), per size, ALL 2024")
    print("=" * 110)
    pl_rng = np.random.default_rng(7)
    for size in NOTIONALS:
        sub = df[df["notional"] == size]
        p_post = sub["l3_posted"].mean()
        naive_c = sub["naive_H30_back_cost"].to_numpy(); agg_c = sub["cost_agg"].to_numpy()
        sims = []
        for _ in range(300):
            mask = pl_rng.random(len(sub)) < p_post
            sims.append(np.where(mask, naive_c, agg_c).mean())
        rand_mean, rand_sd = float(np.mean(sims)), float(np.std(sims))
        l3 = float(sub["l3_H30_back_cost"].mean())
        z = (l3 - rand_mean) / rand_sd if rand_sd > 0 else float("nan")
        verdict = "BEATS placebo" if l3 < rand_mean else "WORSE than placebo"
        print(f"  size ${int(size):>6,}: L3={l3:6.3f}  placebo={rand_mean:6.3f}"
              f" (sd={rand_sd:.3f})  edge={rand_mean - l3:+.3f}bps (z={z:+.2f}) -> {verdict}")

    # ===================================================================== #
    # 4) REGIME BREAKDOWN: crash (Aug-5) vs calm; per month
    # ===================================================================== #
    print("\n" + "=" * 110)
    print("REGIME BREAKDOWN: does the L3 signal acquire an execution edge in high-vol?")
    print("  Compare crash day (2024-08-05) vs all calm days. (pooled over symbols+sizes)")
    print("=" * 110)
    rrng = np.random.default_rng(99)
    crash = df[df["regime"] == "crash"]
    calm = df[df["regime"] == "calm"]
    block_metrics(calm, rrng, "CALM days (12 dates)")
    block_metrics(crash, rrng, "CRASH day 2024-08-05")
    # also Aug 6 (aftermath) and Mar 13 (BTC near ATH) as extra high-interest slices
    block_metrics(df[df["date"] == "2024-08-06"], rrng, "2024-08-06 (crash aftermath)")
    block_metrics(df[df["date"] == "2024-03-13"], rrng, "2024-03-13 (BTC near ATH)")

    print("\n" + "=" * 110)
    print("PER-MONTH (pooled symbols+sizes): mean costs, L3 edge vs placebo, signal corr")
    print("=" * 110)
    mrng = np.random.default_rng(123)
    per_month = []
    for date in DATES:
        sub = df[df["date"] == date]
        m = block_metrics(sub, mrng, f"{date}{' [CRASH]' if date in CRASH_DATES else ''}")
        if m:
            m["date"] = date
            per_month.append(m)

    # compact per-date table
    print("\n" + "-" * 110)
    print(f"{'date':<12}{'n':>5}{'AGG':>8}{'NAIVE':>8}{'L3':>8}{'edge_vs_plac':>13}"
          f"{'oracle_save':>12}{'corr':>9}{'spread':>8}")
    print("-" * 110)
    for m in per_month:
        print(f"{m['date']:<12}{m['n']:>5}{m['agg']:>8.2f}{m['naive']:>8.2f}{m['l3']:>8.2f}"
              f"{m['edge_vs_placebo']:>+13.3f}{m['oracle_save_vs_agg']:>+12.3f}"
              f"{m['corr_sig']:>+9.4f}{m['spread']:>8.2f}")

    # ===================================================================== #
    # 5) CAPTURABILITY: per-feature correlations (pooled, all 2024)
    # ===================================================================== #
    print("\n" + "=" * 110)
    print("CAPTURABILITY (pooled all-2024): corr of L2 features w/ per-order execution advantage")
    print("  target1 = passive_minus_agg (negative => posting cheaper). target2 = naive fill (1=filled).")
    print("=" * 110)
    df["sig_dir"] = df["book_ofi"] * np.where(df["side"] == "buy", 1.0, -1.0)
    df["passive_minus_agg"] = df[NAIVE] - df[AGG]
    df["naive_fill"] = df["naive_H30_back_fill"].astype(float)
    df["cancel_dir"] = df["cancel_imb"] * np.where(df["side"] == "buy", 1.0, -1.0)
    df["abs_ofi"] = df["book_ofi"].abs()
    colmap = {"book_ofi (dir)": "sig_dir", "cancel_imb (dir)": "cancel_dir",
              "spread_bps": "spread_bps", "|book_ofi|": "abs_ofi"}
    for name, col in colmap.items():
        c = df[[col, "passive_minus_agg", "naive_fill"]].dropna()
        r1 = np.corrcoef(c[col], c["passive_minus_agg"])[0, 1]
        r2 = np.corrcoef(c[col], c["naive_fill"])[0, 1]
        print(f"  {name:<18} corr(.,passive-agg)={r1:+.4f}   corr(.,fill)={r2:+.4f}")

    # same correlations split crash vs calm (key regime test for the signal)
    print("\n  -- book-OFI(dir) corr split by regime --")
    for lab, sl in (("calm", calm), ("crash", crash)):
        sd = sl["book_ofi"] * np.where(sl["side"] == "buy", 1.0, -1.0)
        pma = sl[NAIVE] - sl[AGG]
        cc = pd.DataFrame({"sd": sd, "pma": pma}).dropna()
        r = np.corrcoef(cc["sd"], cc["pma"])[0, 1] if len(cc) > 30 else np.nan
        print(f"    {lab:<6} corr(book-OFI dir, passive-agg) = {r:+.4f}  (n={len(cc)})")

    print(f"\nDone in {time.time()-t_start:.0f}s. Wrote {os.path.basename(ORDERS_CSV)}, "
          f"{os.path.basename(SUMMARY_CSV)}")


if __name__ == "__main__":
    main()
