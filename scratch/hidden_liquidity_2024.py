"""
Hidden / iceberg liquidity inferred from L2 + trade tape (Binance spot).

Idea: True L3 (order-by-order) is unavailable. But one L3 quantity is partially
inferable from L2: HIDDEN liquidity. When MORE volume executes at a price than was
ever DISPLAYED at that price, the excess must have come from hidden/iceberg orders
(replenishing icebergs or fully hidden orders).

Detection (per best-price "episode"):
  - An episode at the best ASK is a maximal run where the best-ask price == P.
    BUY-aggressor trades executing at price==P consume that ask.
    hidden = max(0, executed_buy_vol_at_P  -  max_displayed_ask_size_at_P_during_episode)
  - Symmetric for the best BID with SELL-aggressor trades.
  - Using MAX displayed size (not start) is conservative: ordinary visible refills
    that are captured by a later snapshot are NOT counted as hidden.

Caveat: snapshots are post-update, so we UNDER-detect (a hidden fill immediately
followed by a same-price visible refill in the next snapshot is missed). All hidden
numbers are therefore a LOWER BOUND on true hidden activity.

Outputs: per-symbol & per-hour hidden_fraction and iceberg-episode rate, plus an
honest HAC (Newey-West) predictive test for (a) next-hour realized vol and
(b) price direction.

RUN SYNCHRONOUSLY. No background processes.
"""

import os
import gzip
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

BASE = r"C:\Users\jackw\Desktop\math-199-research\data\l2_raw\binance"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]
# Full-2024 sample spanning all 12 months incl. volatile regimes
# (2024-08-05 yen-carry crash, 2024-03-13 BTC-near-ATH).
DATES = ['2024-01-15', '2024-02-15', '2024-03-13', '2024-04-16', '2024-05-15',
         '2024-06-14', '2024-07-15', '2024-08-05', '2024-08-06', '2024-09-16',
         '2024-10-15', '2024-11-12', '2024-12-16']

# Price match tolerance as a fraction of price (handles float repr noise on the
# exact-equality "price == P" condition). 1e-9 is far tighter than any tick.
REL_TOL = 1e-9


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# Episode detection for one symbol-day.
# We interleave book best-quote updates and trades on a single time axis.
# At each instant we know current best-ask price+size and best-bid price+size.
# An "episode" on the ask side starts when best-ask PRICE changes to a new level
# and ends when it changes again. Within the episode we accumulate executed BUY
# volume at exactly that price, and track the max displayed ask size seen.
# ----------------------------------------------------------------------------

def read_book_best(path):
    """Read only best bid/ask price+size columns from a book snapshot file."""
    cols = ["timestamp", "asks[0].price", "asks[0].amount",
            "bids[0].price", "bids[0].amount"]
    df = pd.read_csv(path, compression="gzip", usecols=cols)
    df = df.rename(columns={
        "asks[0].price": "ask_p", "asks[0].amount": "ask_s",
        "bids[0].price": "bid_p", "bids[0].amount": "bid_s",
    })
    df = df.dropna(subset=["timestamp", "ask_p", "bid_p"])
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def read_trades(path):
    cols = ["timestamp", "side", "price", "amount"]
    df = pd.read_csv(path, compression="gzip", usecols=cols)
    df = df.dropna(subset=["timestamp", "side", "price", "amount"])
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def detect_episodes_day(book, trades):
    """
    Build a merged event stream and walk it once.

    Returns a list of episode dicts. Each episode is one side ('ask' or 'bid'),
    a price level, the executed aggressor volume at that price during the run,
    and the max displayed size at that price during the run.

    Algorithm:
      - Merge book updates (type=0) and trades (type=1) by timestamp; on ties,
        process the BOOK update first so the displayed size at a price is known
        before trades at that instant are attributed (book snapshot is post-update,
        so the size shown already reflects state at/after that ts).
      - Maintain current best ask price/size and best bid price/size.
      - When the best-ask PRICE changes, close the open ask-episode and open a new
        one at the new price (seeded with the newly displayed size as max).
      - For a BUY-aggressor trade, if its price == current best-ask price, add its
        amount to the open ask-episode's executed volume. (Buys lift the ask.)
      - Symmetric for bid / sell-aggressor.
    """
    b = book[["timestamp", "ask_p", "ask_s", "bid_p", "bid_s"]].to_numpy()
    t = trades[["timestamp", "price", "amount"]].copy()
    t["is_buy"] = (trades["side"].to_numpy() == "buy")
    tt = t[["timestamp", "price", "amount", "is_buy"]].to_numpy()

    nb, nt = len(b), len(tt)
    ib = it = 0

    # current best state
    cur_ask_p = np.nan
    cur_ask_s = np.nan
    cur_bid_p = np.nan
    cur_bid_s = np.nan

    # open episodes
    ask_ep = None  # dict: price, max_disp, exec_vol, n_trades, start_ts
    bid_ep = None

    episodes = []

    def close(ep, side):
        if ep is not None and ep["exec_vol"] > 0.0:
            episodes.append({
                "side": side,
                "price": ep["price"],
                "max_disp": ep["max_disp"],
                "exec_vol": ep["exec_vol"],
                "start_ts": ep["start_ts"],
                "end_ts": ep["end_ts"],
            })

    def price_eq(a, bp):
        return abs(a - bp) <= REL_TOL * bp

    while ib < nb or it < nt:
        take_book = False
        if ib < nb and it < nt:
            # tie -> book first
            take_book = b[ib, 0] <= tt[it, 0]
        elif ib < nb:
            take_book = True
        else:
            take_book = False

        if take_book:
            ts, ap, asz, bp, bsz = b[ib]
            ib += 1

            # ----- ASK side -----
            if not np.isnan(cur_ask_p) and not price_eq(ap, cur_ask_p):
                # best-ask price moved: close current ask episode, open new
                if ask_ep is not None:
                    ask_ep["end_ts"] = ts
                close(ask_ep, "ask")
                ask_ep = {"price": ap, "max_disp": asz, "exec_vol": 0.0,
                          "start_ts": ts, "end_ts": ts}
            elif np.isnan(cur_ask_p):
                ask_ep = {"price": ap, "max_disp": asz, "exec_vol": 0.0,
                          "start_ts": ts, "end_ts": ts}
            else:
                # same ask price: update max displayed size
                if asz > ask_ep["max_disp"]:
                    ask_ep["max_disp"] = asz
            cur_ask_p, cur_ask_s = ap, asz

            # ----- BID side -----
            if not np.isnan(cur_bid_p) and not price_eq(bp, cur_bid_p):
                if bid_ep is not None:
                    bid_ep["end_ts"] = ts
                close(bid_ep, "bid")
                bid_ep = {"price": bp, "max_disp": bsz, "exec_vol": 0.0,
                          "start_ts": ts, "end_ts": ts}
            elif np.isnan(cur_bid_p):
                bid_ep = {"price": bp, "max_disp": bsz, "exec_vol": 0.0,
                          "start_ts": ts, "end_ts": ts}
            else:
                if bsz > bid_ep["max_disp"]:
                    bid_ep["max_disp"] = bsz
            cur_bid_p, cur_bid_s = bp, bsz

        else:
            ts, pr, amt, is_buy = tt[it]
            it += 1
            if is_buy:
                # buy aggressor lifts the ask; count if at current best-ask price
                if ask_ep is not None and not np.isnan(cur_ask_p) and price_eq(pr, cur_ask_p):
                    ask_ep["exec_vol"] += amt
                    ask_ep["end_ts"] = ts
            else:
                if bid_ep is not None and not np.isnan(cur_bid_p) and price_eq(pr, cur_bid_p):
                    bid_ep["exec_vol"] += amt
                    bid_ep["end_ts"] = ts

    # close trailing episodes
    if ask_ep is not None:
        close(ask_ep, "ask")
    if bid_ep is not None:
        close(bid_ep, "bid")

    return episodes


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def episodes_to_frame(episodes):
    if not episodes:
        return pd.DataFrame(columns=["side", "price", "max_disp", "exec_vol",
                                     "start_ts", "end_ts", "hidden"])
    df = pd.DataFrame(episodes)
    df["hidden"] = np.maximum(0.0, df["exec_vol"] - df["max_disp"])
    return df


def main():
    log("=" * 78)
    log("HIDDEN / ICEBERG LIQUIDITY — inferring an L3 quantity from L2 + tape")
    log("Symbols: %s" % SYMBOLS)
    log("Dates  : %s .. %s (%d days)" % (DATES[0], DATES[-1], len(DATES)))
    log("=" * 78)

    all_ep = []          # per-episode rows with symbol & hour
    per_sym_rows = []    # descriptive per-symbol summary

    for sym in SYMBOLS:
        sym_eps = []
        for d in DATES:
            bpath = os.path.join(BASE, "book_snapshot_25", sym, d + ".csv.gz")
            tpath = os.path.join(BASE, "trades", sym, d + ".csv.gz")
            if not (os.path.exists(bpath) and os.path.exists(tpath)):
                log("  MISSING %s %s -> skip" % (sym, d))
                continue
            book = read_book_best(bpath)
            trades = read_trades(tpath)
            eps = detect_episodes_day(book, trades)
            edf = episodes_to_frame(eps)
            if len(edf):
                # hour bucket from end_ts (us -> hour)
                edf["sym"] = sym
                edf["date"] = d
                edf["hour"] = (edf["end_ts"] // 3_600_000_000).astype("int64")
                sym_eps.append(edf)
            log("  %s %s: book=%d trades=%d episodes(traded)=%d" %
                (sym, d, len(book), len(trades), len(edf)))
            del book, trades

        if sym_eps:
            sdf = pd.concat(sym_eps, ignore_index=True)
        else:
            sdf = episodes_to_frame([])
            sdf["sym"] = sym
            sdf["hour"] = 0
        all_ep.append(sdf)

        tot_exec = sdf["exec_vol"].sum()
        tot_hidden = sdf["hidden"].sum()
        hidden_frac = (tot_hidden / tot_exec) if tot_exec > 0 else np.nan
        n_ep = len(sdf)
        n_iceberg = int((sdf["hidden"] > 0).sum())
        iceberg_rate = (n_iceberg / n_ep) if n_ep > 0 else np.nan
        # value-weighted hidden notional share too (exec & hidden are base units;
        # multiply by price for a $ view)
        hidden_notional = (sdf["hidden"] * sdf["price"]).sum()
        exec_notional = (sdf["exec_vol"] * sdf["price"]).sum()
        hidden_frac_notional = (hidden_notional / exec_notional) if exec_notional > 0 else np.nan

        per_sym_rows.append({
            "sym": sym,
            "traded_episodes": n_ep,
            "iceberg_episodes": n_iceberg,
            "iceberg_rate_pct": 100 * iceberg_rate,
            "exec_vol_base": tot_exec,
            "hidden_vol_base": tot_hidden,
            "hidden_frac_pct": 100 * hidden_frac,
            "hidden_frac_notional_pct": 100 * hidden_frac_notional,
        })

    full = pd.concat(all_ep, ignore_index=True)

    log("")
    log("#" * 78)
    log("# (1) DESCRIPTIVE — hidden fraction & iceberg-episode rate per symbol")
    log("#     (LOWER BOUND: post-update snapshots under-detect hidden fills)")
    log("#" * 78)
    psum = pd.DataFrame(per_sym_rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda x: f"{x:,.4f}"):
        log(psum.to_string(index=False))

    # save per-symbol summary
    out_csv = r"C:\Users\jackw\Desktop\math-199-research\scratch\hidden_liquidity_2024_persym.csv"
    psum.to_csv(out_csv, index=False)
    log("\n[saved] %s" % out_csv)

    # ------------------------------------------------------------------
    # (1b) PER-DAY x SYMBOL descriptive — to expose regime dependence
    #      (esp. the 2024-08-05 yen-carry crash day).
    # ------------------------------------------------------------------
    log("")
    log("#" * 78)
    log("# (1b) PER-DAY x SYMBOL hidden fraction & iceberg rate (regime view)")
    log("#" * 78)
    if "date" in full.columns:
        gd = full.groupby(["date", "sym"], sort=True)
        dayrows = []
        for (dd, ss), grp in gd:
            ev = grp["exec_vol"].sum()
            hv = grp["hidden"].sum()
            ne = len(grp)
            ni = int((grp["hidden"] > 0).sum())
            dayrows.append({
                "date": dd, "sym": ss,
                "episodes": ne,
                "iceberg_rate_pct": 100 * (ni / ne) if ne else np.nan,
                "hidden_frac_pct": 100 * (hv / ev) if ev > 0 else np.nan,
            })
        dday = pd.DataFrame(dayrows)
        with pd.option_context("display.width", 200, "display.max_columns", 20,
                               "display.float_format", lambda x: f"{x:,.4f}"):
            log(dday.to_string(index=False))
        out_day = r"C:\Users\jackw\Desktop\math-199-research\scratch\hidden_liquidity_2024_perday.csv"
        dday.to_csv(out_day, index=False)
        log("\n[saved] %s" % out_day)
        # crash-day vs non-crash mean per symbol
        log("\n  Aug-5 crash-day vs full-sample-mean (hidden_frac_pct / iceberg_rate_pct):")
        for ss in SYMBOLS:
            sub = dday[dday["sym"] == ss]
            crash = sub[sub["date"] == "2024-08-05"]
            base = sub[sub["date"] != "2024-08-05"]
            if len(crash) and len(base):
                log("    %-8s crash hf=%.3f%% ir=%.2f%%  | non-crash-mean hf=%.3f%% ir=%.2f%%" % (
                    ss,
                    crash["hidden_frac_pct"].iloc[0], crash["iceberg_rate_pct"].iloc[0],
                    base["hidden_frac_pct"].mean(), base["iceberg_rate_pct"].mean()))

    # ------------------------------------------------------------------
    # Build hourly panel for predictive tests
    # ------------------------------------------------------------------
    log("")
    log("#" * 78)
    log("# (2) PREDICTIVE — hourly panel + Newey-West (HAC) tests")
    log("#" * 78)

    panel = build_hourly_panel(full)
    if panel is None or len(panel) == 0:
        log("  No panel rows; aborting predictive tests.")
        return

    out_panel = r"C:\Users\jackw\Desktop\math-199-research\scratch\hidden_liquidity_2024_panel.csv"
    panel.to_csv(out_panel, index=False)
    log("[saved] %s  (rows=%d)" % (out_panel, len(panel)))

    run_predictive(panel)

    log("")
    log("DONE.")


def build_hourly_panel(full):
    """
    Hourly features per symbol:
      - hidden_fraction   = sum(hidden)/sum(exec_vol) within the hour (all episodes)
      - iceberg_rate      = frac of traded episodes with hidden>0 in the hour
      - signed_hidden     = (hidden on ASK - hidden on BID) / exec_vol
                            ASK-hidden = absorbed BUYING; BID-hidden = absorbed SELLING
      - trade_count proxy = number of traded episodes (intensity)
      - exec_vol          = total executed base volume
    Plus price & return features computed from a separate trade-price reconstruction.
    """
    # Hidden / exec aggregates by (sym,hour)
    g = full.groupby(["sym", "hour"], sort=True)
    agg = g.agg(
        exec_vol=("exec_vol", "sum"),
        hidden=("hidden", "sum"),
        n_ep=("side", "size"),
        n_iceberg=("hidden", lambda s: int((s > 0).sum())),
    ).reset_index()

    # signed hidden: ask-hidden minus bid-hidden
    ask = full[full["side"] == "ask"].groupby(["sym", "hour"])["hidden"].sum().rename("hidden_ask")
    bid = full[full["side"] == "bid"].groupby(["sym", "hour"])["hidden"].sum().rename("hidden_bid")
    agg = agg.merge(ask, on=["sym", "hour"], how="left").merge(bid, on=["sym", "hour"], how="left")
    agg["hidden_ask"] = agg["hidden_ask"].fillna(0.0)
    agg["hidden_bid"] = agg["hidden_bid"].fillna(0.0)

    agg["hidden_fraction"] = np.where(agg["exec_vol"] > 0, agg["hidden"] / agg["exec_vol"], 0.0)
    agg["iceberg_rate"] = np.where(agg["n_ep"] > 0, agg["n_iceberg"] / agg["n_ep"], 0.0)
    agg["signed_hidden"] = np.where(agg["exec_vol"] > 0,
                                    (agg["hidden_ask"] - agg["hidden_bid"]) / agg["exec_vol"], 0.0)

    # Now build price/return/vol/OFI/spread controls from raw trades+book per sym-hour.
    pricefeat = build_price_features()
    panel = agg.merge(pricefeat, on=["sym", "hour"], how="inner")

    # next-hour targets (within symbol, sorted by hour)
    panel = panel.sort_values(["sym", "hour"]).reset_index(drop=True)
    panel["rv_next"] = panel.groupby("sym")["rv"].shift(-1)
    panel["ret_next"] = panel.groupby("sym")["ret"].shift(-1)
    # drop hours with gaps (next hour not exactly hour+1)
    panel["hour_next"] = panel.groupby("sym")["hour"].shift(-1)
    panel["contig"] = (panel["hour_next"] - panel["hour"]) == 1
    return panel


def build_price_features():
    """
    Per (sym,hour): hourly return, realized vol (from 1-min subsampled returns),
    signed order-flow imbalance (OFI from trade signs), trade_count, mean quoted
    spread. Computed directly from raw files (independent of episode logic).
    """
    rows = []
    for sym in SYMBOLS:
        for d in DATES:
            tpath = os.path.join(BASE, "trades", sym, d + ".csv.gz")
            bpath = os.path.join(BASE, "book_snapshot_25", sym, d + ".csv.gz")
            if not (os.path.exists(tpath) and os.path.exists(bpath)):
                continue
            tr = pd.read_csv(tpath, compression="gzip",
                             usecols=["timestamp", "side", "price", "amount"])
            tr = tr.dropna()
            tr["hour"] = (tr["timestamp"] // 3_600_000_000).astype("int64")
            tr["minute"] = (tr["timestamp"] // 60_000_000).astype("int64")
            tr["signed"] = np.where(tr["side"].to_numpy() == "buy", 1.0, -1.0) * tr["amount"].to_numpy()

            # quoted spread from book (mean over hour)
            bk = pd.read_csv(bpath, compression="gzip",
                             usecols=["timestamp", "asks[0].price", "bids[0].price"])
            bk = bk.dropna()
            bk["hour"] = (bk["timestamp"] // 3_600_000_000).astype("int64")
            mid = 0.5 * (bk["asks[0].price"] + bk["bids[0].price"])
            bk["rel_spread"] = (bk["asks[0].price"] - bk["bids[0].price"]) / mid
            sp = bk.groupby("hour")["rel_spread"].mean().rename("spread")

            for hr, grp in tr.groupby("hour"):
                # minute-bar last prices for RV
                mb = grp.groupby("minute")["price"].last()
                logp = np.log(mb.to_numpy())
                if len(logp) >= 2:
                    r1 = np.diff(logp)
                    rv = float(np.sqrt(np.sum(r1 ** 2)))  # realized vol (sqrt of sum sq min returns)
                else:
                    rv = np.nan
                p0 = grp["price"].iloc[0]
                p1 = grp["price"].iloc[-1]
                ret = float(np.log(p1) - np.log(p0))
                vol_base = float(grp["amount"].sum())
                ofi = float(grp["signed"].sum())
                ofi_norm = ofi / vol_base if vol_base > 0 else 0.0
                rows.append({
                    "sym": sym, "hour": int(hr),
                    "ret": ret, "rv": rv,
                    "trade_count": int(len(grp)),
                    "ofi_norm": ofi_norm,
                    "spread": float(sp.get(hr, np.nan)),
                })
            del tr, bk
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# HAC (Newey-West) regression utilities
# ----------------------------------------------------------------------------

def hac_ols(y, X, maxlags=2):
    import statsmodels.api as sm
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, Xc, missing="drop")
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return res


def run_predictive(panel):
    import statsmodels.api as sm

    df = panel[panel["contig"]].copy()
    # within-symbol standardization of the features so coefficients are comparable
    feats = ["hidden_fraction", "iceberg_rate", "signed_hidden",
             "rv", "trade_count", "ofi_norm", "spread", "ret"]
    df = df.dropna(subset=["rv", "rv_next", "ret_next", "trade_count", "spread"])
    # winsorize rv lightly to avoid one-bar dominance
    for c in ["rv", "rv_next"]:
        hi = df[c].quantile(0.999)
        df[c] = df[c].clip(upper=hi)

    log("\nPanel for regressions: N=%d sym-hours across %d symbols" %
        (len(df), df["sym"].nunique()))
    log("Feature means (per symbol):")
    with pd.option_context("display.width", 200, "display.float_format", lambda x: f"{x:.5f}"):
        log(df.groupby("sym")[["hidden_fraction", "iceberg_rate", "signed_hidden",
                               "rv", "ofi_norm", "spread"]].mean().to_string())

    # Standardize features within symbol (z-score) to pool across symbols fairly.
    def z(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd > 0 else s * 0.0

    for c in feats + ["rv_next"]:
        df[c + "_z"] = df.groupby("sym")[c].transform(z)
    df["ret_next_z"] = df.groupby("sym")["ret_next"].transform(z)

    # ===========================================================
    # (a) NEXT-HOUR realized vol: baseline vs +hidden activity
    # ===========================================================
    log("\n" + "-" * 70)
    log("(a) Predict NEXT-hour realized vol. Incremental over rv + trade intensity.")
    log("-" * 70)

    base_X = df[["rv_z", "trade_count_z"]]
    y = df["rv_next_z"]
    res_base = hac_ols(y, base_X, maxlags=3)
    r2_base = res_base.rsquared

    full_X = df[["rv_z", "trade_count_z", "hidden_fraction_z", "iceberg_rate_z"]]
    res_full = hac_ols(y, full_X, maxlags=3)
    r2_full = res_full.rsquared

    log("Baseline  R^2 = %.5f  (rv, trade_count)" % r2_base)
    log("Full      R^2 = %.5f  (+ hidden_fraction, iceberg_rate)" % r2_full)
    log("Incremental R^2 = %.5f" % (r2_full - r2_base))
    log("\nHAC (Newey-West, L=3) coefficients on the FULL model:")
    log(coef_table(res_full))

    # ===========================================================
    # (b) Signed hidden vs price direction (contemp + next hour)
    # ===========================================================
    log("\n" + "-" * 70)
    log("(b) Signed hidden activity vs price direction.")
    log("    ASK-hidden = absorbed buying; BID-hidden = absorbed selling.")
    log("-" * 70)

    # Contemporaneous: this-hour return on this-hour signed_hidden (+ OFI control)
    res_c = hac_ols(df["ret_z"], df[["signed_hidden_z", "ofi_norm_z"]], maxlags=3)
    log("\nCONTEMPORANEOUS: ret ~ signed_hidden + ofi_norm   R^2=%.5f" % res_c.rsquared)
    log(coef_table(res_c))

    # Next-hour: next return on this-hour signed_hidden (+ OFI control)
    res_n = hac_ols(df["ret_next_z"], df[["signed_hidden_z", "ofi_norm_z"]], maxlags=3)
    log("\nNEXT-HOUR: ret_next ~ signed_hidden + ofi_norm    R^2=%.5f" % res_n.rsquared)
    log(coef_table(res_n))

    # Also: does signed_hidden add to OFI for next-hour return? incremental
    res_ofi = hac_ols(df["ret_next_z"], df[["ofi_norm_z"]], maxlags=3)
    log("\nNext-hour return incremental R^2 from adding signed_hidden to OFI: %.5f"
        % (res_n.rsquared - res_ofi.rsquared))

    log("\n" + "=" * 70)
    log("INTERPRETATION GUIDE:")
    log(" - |HAC t| < ~2.0  => not significant; treat as null.")
    log(" - Incremental R^2 near 0 + small t on hidden terms => near-efficiency:")
    log("   hidden liquidity is REAL & measurable but adds no predictive edge.")
    log("=" * 70)


def coef_table(res):
    params = res.params
    t = res.tvalues
    p = res.pvalues
    lines = []
    lines.append(f"{'term':<22}{'coef':>12}{'HAC_t':>10}{'p':>10}")
    for name in params.index:
        lines.append(f"{name:<22}{params[name]:>12.5f}{t[name]:>10.3f}{p[name]:>10.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
