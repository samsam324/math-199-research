# store_data.py
from __future__ import annotations

from src.data_store import StoreConfig, sync_store_all


def main():
    cfg = StoreConfig(
        base_url="https://api.binance.us",
        interval="1h",
        quote_asset="USDT",
        data_dir="data",
        max_symbols=None,
    )

    synced, failures = sync_store_all(cfg)

    print(f"\nSynced symbols: {len(synced)}")
    print(f"Failures: {len(failures)}")
    if failures:
        print("First 10 failures:")
        for s, err in failures[:10]:
            print(" ", s, "->", err)


if __name__ == "__main__":
    main()
