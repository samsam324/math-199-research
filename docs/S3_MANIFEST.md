# Shared data on S3

Private bucket holding the project's local data so co-authors can pull it without
re-downloading from Tardis or the exchange APIs.

- **Bucket:** `math199-statarb-data-873750256216` (region `us-west-2`)
- **Access:** private (Block Public Access on). Share individual files with time-limited
  presigned links (`scripts/presign.py`), or grant a collaborator a read-only IAM policy.
- **Do not make this public.** The L2 stores are Tardis.dev data; redistributing them
  publicly may violate Tardis's license. Sharing within the research team is fine.

## Layout (`s3://<bucket>/data/...`)

| Store | What | Size | Storage class |
|---|---|---|---|
| `l2/` | Binance L2, resampled 1 Hz, 10-level book (50 syms x Jan-Jul 2024) | 68 GB | Standard-IA |
| `trades/` | Binance aggregated trade bars (50 syms x 2024) | 9.8 GB | Standard-IA |
| `microstructure/` | generated hourly features (VPIN, Kyle lambda, OFI) | 11 MB | Standard-IA |
| `spot_1h/` | Binance.US hourly USDT klines (204 syms, 2019-2026) | 121 MB | Standard-IA |
| `coinbase_1h/` | Coinbase hourly cross-check (41 syms) | 53 MB | Standard-IA |
| `spot_1h_delisted/` | FTT/LUNA/LUNC/UST collapse klines | 2.6 MB | Standard-IA |
| `l2_samples/` | single-day Tardis L2 samples (BTC/ETH) | 221 MB | Standard-IA |
| `metadata/` | sync-run receipts | 2 KB | Standard-IA |
| `l2_raw/` | raw Tardis csv.gz (book_snapshot_25 + trades) — **only if synced with `-Scope full`** | 391 GB | Deep Archive |

Not uploaded: the 114 MB regenerable `exec_broad_orders.csv`, build artifacts, and the
`scratch/` analysis caches (those live in git).

## Upload (owner)

```powershell
.\scripts\sync_to_s3.ps1            # curated ~78 GB -> Standard-IA
.\scripts\sync_to_s3.ps1 -Scope full   # also l2_raw 391 GB -> Deep Archive
```

## Download (collaborator)

A presigned link (no AWS account needed):
```
python scripts/presign.py --list data/spot_1h          # see keys
python scripts/presign.py data/spot_1h/BTCUSDT.parquet # 7-day link, curl/browser
```
Or, with read-only creds, the whole store:
```
aws s3 sync s3://math199-statarb-data-873750256216/data/l2/ ./l2/
```
`l2_raw/` is in Deep Archive: `aws s3 restore` first (12-48 h) before download.

## Rough cost

Curated 78 GB in Standard-IA ~ $1/mo; l2_raw 391 GB in Deep Archive ~ $0.40/mo. First
100 GB/month of egress is free, then ~$0.09/GB.
