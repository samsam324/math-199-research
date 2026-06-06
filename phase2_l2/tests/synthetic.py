"""
Synthetic L2 + trades generators for tests. Produces canonical-schema frames
without needing real Tardis data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.l2_store import L2Config, expected_columns as l2_expected_columns
from src.trade_store import CANONICAL_COLUMNS as TRADE_COLUMNS


def make_synthetic_book(
    start: str = "2026-01-01",
    n_rows: int = 3600,
    freq_ms: int = 1000,
    mid_start: float = 100.0,
    sigma: float = 0.001,
    spread_bps: float = 1.0,
    levels: int = 10,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Generates an L2 book frame in canonical schema. Mid follows GBM with
    sigma per step; bid/ask are mid +/- spread/2; sizes are exponential.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n_rows, freq=f"{freq_ms}ms", tz="UTC")
    rets = rng.normal(0, sigma, n_rows)
    mid = mid_start * np.exp(np.cumsum(rets))
    half_spread = mid * (spread_bps / 1e4) / 2.0
    bid1 = mid - half_spread
    ask1 = mid + half_spread

    cfg = L2Config(levels=levels)
    out = pd.DataFrame(index=range(n_rows))
    out["timestamp"] = ts
    out["local_timestamp"] = ts
    out["seq"] = np.arange(n_rows, dtype=np.uint64)
    for i in range(1, levels + 1):
        step = i * half_spread
        out[f"bid_px_{i}"] = bid1 - (i - 1) * 2 * half_spread
        out[f"bid_sz_{i}"] = rng.exponential(10.0, n_rows)
        out[f"ask_px_{i}"] = ask1 + (i - 1) * 2 * half_spread
        out[f"ask_sz_{i}"] = rng.exponential(10.0, n_rows)
    out["is_snapshot"] = True
    out["is_delta"] = False
    return out[l2_expected_columns(cfg)].copy()


def make_synthetic_trades(
    book: pd.DataFrame,
    avg_trades_per_bar: float = 5.0,
    avg_notional: float = 5_000.0,
    inst_share: float = 0.05,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Generates a trades frame referencing a book frame's timestamps. Each book
    row spawns ~Poisson(avg_trades_per_bar) trades clustered at that timestamp
    + jitter. Notional is exponential with `avg_notional`; with probability
    `inst_share` a trade is multiplied to land in the institutional bucket
    (>$100k by default). Side is random buy/sell.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _, br in book.iterrows():
        n = int(rng.poisson(avg_trades_per_bar))
        for _ in range(n):
            jitter_us = int(rng.integers(0, 1_000_000))
            ts = br["timestamp"] + pd.Timedelta(microseconds=jitter_us)
            side = "buy" if rng.random() < 0.5 else "sell"
            notional = rng.exponential(avg_notional)
            if rng.random() < inst_share:
                notional *= 50.0
            mid = (float(br["bid_px_1"]) + float(br["ask_px_1"])) / 2.0
            # Trade at touch on the aggressor side
            price = float(br["ask_px_1"]) if side == "buy" else float(br["bid_px_1"])
            amount = notional / price
            rows.append({
                "timestamp": ts, "local_timestamp": ts,
                "trade_id": "", "side": side,
                "price": price, "amount": amount, "notional": notional,
            })
    if not rows:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["side"] = pd.Categorical(df["side"], categories=["buy", "sell", ""])
    return df[TRADE_COLUMNS].copy()


def make_synthetic_pair_books(
    n_rows: int = 3600, seed: int = 7,
) -> tuple:
    """
    Two cointegrated books: log_b ~ GBM; log_a = alpha + beta * log_b + AR(1) noise.
    Returns (book_a, trades_a, book_b, trades_b) with matched timestamps.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n_rows, freq="1s", tz="UTC")

    # B is a GBM
    rets_b = rng.normal(0, 0.0005, n_rows)
    p_b = 100.0 * np.exp(np.cumsum(rets_b))

    # A is alpha + beta * B + AR(1)
    alpha = 0.1; beta = 1.5
    log_b = np.log(p_b)
    eps = np.zeros(n_rows)
    rho = 0.95
    for t in range(1, n_rows):
        eps[t] = rho * eps[t-1] + rng.normal(0, 0.0008)
    log_a = alpha + beta * log_b + eps
    p_a = np.exp(log_a)

    book_a = make_synthetic_book(n_rows=n_rows, seed=seed)
    book_a["timestamp"] = ts; book_a["local_timestamp"] = ts
    half_a = p_a * 1e-4 / 2
    for i in range(1, 11):
        book_a[f"bid_px_{i}"] = p_a - half_a - (i - 1) * 2 * half_a
        book_a[f"ask_px_{i}"] = p_a + half_a + (i - 1) * 2 * half_a

    book_b = make_synthetic_book(n_rows=n_rows, seed=seed + 1)
    book_b["timestamp"] = ts; book_b["local_timestamp"] = ts
    half_b = p_b * 1e-4 / 2
    for i in range(1, 11):
        book_b[f"bid_px_{i}"] = p_b - half_b - (i - 1) * 2 * half_b
        book_b[f"ask_px_{i}"] = p_b + half_b + (i - 1) * 2 * half_b

    trades_a = make_synthetic_trades(book_a, seed=seed + 10)
    trades_b = make_synthetic_trades(book_b, seed=seed + 11)
    return book_a, trades_a, book_b, trades_b
