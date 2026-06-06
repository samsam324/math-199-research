# Tardis ingestion

## Verified facts (as of 2026-06-03)

- **Base URL:** `https://datasets.tardis.dev/v1/{exchange}/{data_type}/{YYYY}/{MM}/{DD}/{SYMBOL}.csv.gz`
- **Auth:** `Authorization: Bearer <API_KEY>` header.
- **Free tier:** the first day of any month is downloadable without auth. Use
  this for schema validation and integration tests.
- **Binance perps exchange code:** `binance-futures`. Spot is `binance`.
- **Symbol format:** plain ticker, e.g. `BTCUSDT`, `ETHUSDT`.
- **Timestamps:** integer microseconds since UNIX epoch (UTC). Both
  `timestamp` (exchange-side) and `local_timestamp` (Tardis receive) columns
  are present.

## Verified column layouts

### `book_snapshot_25`
```
exchange, symbol, timestamp, local_timestamp,
asks[0].price, asks[0].amount, bids[0].price, bids[0].amount,
asks[1].price, asks[1].amount, bids[1].price, bids[1].amount,
... (through level 24)
```

### `trades`
```
exchange, symbol, timestamp, local_timestamp,
id, side, price, amount
```

`side` is one of `buy` / `sell` and represents the aggressor side as
reported by the exchange.

## Real-data sanity check

One day of `binance-futures` BTCUSDT 2024-04-01:
- book_snapshot_25 file: ~86 MB gzipped, ~1.5M rows after parsing
- trades file: ~35 MB gzipped, ~3.9M rows after parsing
- Wall-clock for download + parse + write to canonical parquet: ~90s
  sequential. With `max_concurrency=4` the per-day wall-clock for trades
  is closer to ~10s on consumer bandwidth.

## API key handling

The API key MUST NOT be committed to the repo. Options, in order of
preference:

1. **Environment variable** (recommended):
   ```bash
   export TARDIS_API_KEY="TD.xxx.yyy..."   # add to ~/.bashrc / ~/.zshrc / shell profile
   ```
   The downloader reads this automatically via `TardisConfig().api_key` ->
   `os.environ.get("TARDIS_API_KEY")`.

2. **--api-key CLI flag**:
   ```bash
   python scripts/ingest_tardis.py --api-key TD.xxx ...
   ```
   Avoid this in shared shells; the key ends up in your shell history.

3. **`.env` file at repo root** (gitignored):
   ```
   TARDIS_API_KEY=TD.xxx...
   ```
   Add `python-dotenv` and `load_dotenv()` if you go this route. Not done by
   default to keep dependencies minimal.

The default `.gitignore` excludes `.env` and Tardis cache directories.

## Tardis exchange codes (for later)

If we extend the universe beyond Binance perps:
- `binance` — Binance.com spot
- `binance-futures` — Binance.com USDT-margined perpetuals (current)
- `binance-delivery` — Binance.com coin-margined perpetuals
- `okex-swap`, `bybit`, `deribit`, `coinbase`, `kraken`, ...

Symbol naming differs per venue; e.g. Deribit uses `BTC-PERPETUAL` rather
than `BTCUSDT`. The canonical schema is venue-agnostic but the ingestion
seam in `src/tardis_ingest.py` may need per-exchange column adapters when
we extend.

## What we deliberately DID NOT implement

- **`incremental_book_L2` reconstruction.** `book_snapshot_25` samples the
  book every ~100ms which is sufficient at 1-second bar cadence.
  Reconstructing the full book from incrementals is roughly 10x the
  engineering effort for a marginal fidelity gain.
- **`derivative_ticker`** (funding rate, OI, mark price). Schema is
  documented but we have not added the parser. Add when funding is wired
  into PnL.
- **`options_chain`** (Deribit). Not needed for phase 2 spot/perp pair
  spread work.

## Costs / rate limits

Tardis bills per gigabyte downloaded. Their free tier (first day of each
month) is unmetered. For paid days, downloading the full top-30 perps
universe over 30 days at book_snapshot_25 + trades is on the order of
200 GB. Confirm with Mihai before bulk pulls.
