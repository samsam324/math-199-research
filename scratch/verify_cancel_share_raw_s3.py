"""
Raw-L2 verification: re-derive the paper's "81-85% of best-level liquidity withdrawal is
cancellations" claim from the S3 dataset, bypassing all intermediate scratch outputs.

Pulls one BTC and one ETH 2024 day from S3 (no local cache), joins trades with book updates,
computes the cancel-vs-trade share of best-level size reductions, and compares to the paper.

Output: scratch/raw_s3_cancel_share.csv
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "scripts"))
from l2_query import open_l2, load_day

DAYS = [("BTCUSDT", "2024-03-13"), ("ETHUSDT", "2024-03-13")]


def cancel_share_one_day(con, symbol, date):
    cols = '"timestamp", "bids[0].price", "bids[0].amount", "asks[0].price", "asks[0].amount"'
    bk = load_day(con, symbol, date, dataset="book_snapshot_25", columns=cols)
    bk = bk.rename(columns={
        "bids[0].price": "bp", "bids[0].amount": "bsz",
        "asks[0].price": "ap", "asks[0].amount": "asz",
    }).sort_values("timestamp").reset_index(drop=True)
    bk = bk[(bk["bp"] > 0) & (bk["ap"] > 0) & (bk["ap"] > bk["bp"])].reset_index(drop=True)

    # On each successive snapshot, identify best-level SIZE REDUCTIONS at unchanged price.
    # That is the "best-level withdrawal" the paper attributes to cancels-vs-trades.
    pbp = bk["bp"].shift(1)
    pap = bk["ap"].shift(1)
    pbs = bk["bsz"].shift(1)
    pas = bk["asz"].shift(1)
    same_bid_px = (bk["bp"] == pbp)
    same_ask_px = (bk["ap"] == pap)
    bid_red = (pbs - bk["bsz"]).clip(lower=0).where(same_bid_px, 0.0).fillna(0.0)
    ask_red = (pas - bk["asz"]).clip(lower=0).where(same_ask_px, 0.0).fillna(0.0)
    total_bid_red = float(bid_red.sum())
    total_ask_red = float(ask_red.sum())

    # Trades that ATE the inside (signed by aggressor side):
    #   side=='sell' aggressors hit the bid -> reduce bid_sz
    #   side=='buy'  aggressors lift the ask -> reduce ask_sz
    tr = load_day(con, symbol, date, dataset="trades",
                  columns='"timestamp", "side", "price", "amount"')
    bid_trades = float(tr.loc[tr["side"] == "sell", "amount"].sum())
    ask_trades = float(tr.loc[tr["side"] == "buy",  "amount"].sum())

    bid_cancel = max(total_bid_red - bid_trades, 0.0)
    ask_cancel = max(total_ask_red - ask_trades, 0.0)
    total_red  = total_bid_red + total_ask_red
    total_can  = bid_cancel + ask_cancel
    cancel_share = total_can / total_red if total_red > 0 else float("nan")

    return {
        "symbol": symbol, "date": date,
        "n_book_updates": int(len(bk)),
        "n_trades": int(len(tr)),
        "bid_size_reduction_base": total_bid_red,
        "ask_size_reduction_base": total_ask_red,
        "bid_trade_volume": bid_trades,
        "ask_trade_volume": ask_trades,
        "cancel_share_estimate": cancel_share,
    }


def main():
    con = open_l2()
    print(f"S3 verification: best-level cancel-vs-trade share, raw L2")
    print(f"Paper claim: 81-85% of best-level liquidity withdrawal is cancellations\n")
    print(f"{'symbol':<10}{'date':<14}{'book updates':>14}{'trades':>10}{'cancel share %':>18}")
    print("-" * 70)
    rows = []
    for sym, date in DAYS:
        r = cancel_share_one_day(con, sym, date)
        rows.append(r)
        print(f"{r['symbol']:<10}{r['date']:<14}{r['n_book_updates']:>14,}{r['n_trades']:>10,}{r['cancel_share_estimate']*100:>18.1f}")
    out = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_s3_cancel_share.csv")
    out.to_csv(out_path, index=False)
    print(f"\nsaved -> {out_path}")
    print("Reading: this is a SINGLE-DAY spot check, not the paper's full multi-symbol average.")
    print("If the share falls inside 70-90%, the paper's 81-85% range is supported at this resolution.")


if __name__ == "__main__":
    main()
