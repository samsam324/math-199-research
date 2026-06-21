<#
Sync the math-199 research data to the shared, private S3 bucket.

  .\scripts\sync_to_s3.ps1                 # curated ~78 GB  -> S3 Standard-IA
  .\scripts\sync_to_s3.ps1 -Scope full     # ALSO archive l2_raw (391 GB) -> Glacier Deep Archive

Only named data stores are synced, so secrets (.env at repo root) and regenerable
intermediates are never touched. aws s3 sync is resumable: re-run if interrupted.
#>
param([ValidateSet('curated', 'full')][string]$Scope = 'curated')

$ErrorActionPreference = 'Stop'
$Bucket = 'math199-statarb-data-873750256216'
$Root = Split-Path -Parent $PSScriptRoot
$Data = Join-Path $Root 'data'

# collaborator-useful processed stores -> Standard-IA (cheap, instant access)
$Curated = 'l2', 'trades', 'microstructure', 'spot_1h', 'coinbase_1h', 'spot_1h_delisted', 'l2_samples', 'metadata'
foreach ($s in $Curated) {
    $src = Join-Path $Data $s
    if (Test-Path $src) {
        Write-Host "==> $s  ->  s3://$Bucket/data/$s/  (STANDARD_IA)"
        aws s3 sync "$src" "s3://$Bucket/data/$s/" --storage-class STANDARD_IA --only-show-errors
    }
}

if ($Scope -eq 'full') {
    $raw = Join-Path $Data 'l2_raw'
    Write-Host "==> l2_raw (391 GB)  ->  s3://$Bucket/data/l2_raw/  (DEEP_ARCHIVE)"
    aws s3 sync "$raw" "s3://$Bucket/data/l2_raw/" --storage-class DEEP_ARCHIVE --only-show-errors
}
Write-Host "done. listing top level:"
aws s3 ls "s3://$Bucket/data/"
