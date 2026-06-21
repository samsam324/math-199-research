"""Mint a time-limited shareable download link to a file in the research S3 bucket.
Recipients need NO AWS account. Links expire (SigV4 max 7 days).

  python scripts/presign.py data/spot_1h/BTCUSDT.parquet              # 7-day link
  python scripts/presign.py data/l2/BTCUSDT/2024-01-01.parquet -d 3   # 3-day link
  python scripts/presign.py --list data/metadata                     # list keys under a store
"""
import argparse
import boto3
from botocore.config import Config

BUCKET = "math199-statarb-data-873750256216"
REGION = "us-west-2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key", nargs="?", help="object key, e.g. data/spot_1h/BTCUSDT.parquet")
    ap.add_argument("-d", "--days", type=int, default=7, help="link lifetime in days (max 7)")
    ap.add_argument("--list", dest="list_prefix", help="list keys under a prefix instead")
    a = ap.parse_args()
    # regional endpoint + virtual addressing so presigned URLs don't 307-redirect
    # (the global endpoint redirects for non-us-east-1 buckets and breaks the signature)
    s3 = boto3.client(
        "s3", region_name=REGION,
        endpoint_url=f"https://s3.{REGION}.amazonaws.com",
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))

    if a.list_prefix:
        pre = a.list_prefix.rstrip("/") + "/"
        tok = None
        while True:
            kw = {"Bucket": BUCKET, "Prefix": pre}
            if tok:
                kw["ContinuationToken"] = tok
            r = s3.list_objects_v2(**kw)
            for o in r.get("Contents", []):
                print(f"{o['Size']/1e6:10.2f} MB  {o['Key']}")
            if not r.get("IsTruncated"):
                break
            tok = r["NextContinuationToken"]
        return

    if not a.key:
        ap.error("provide a key, or use --list <prefix>")
    days = max(1, min(7, a.days))  # SigV4 presigned URLs cap at 7 days
    url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": BUCKET, "Key": a.key}, ExpiresIn=days * 86400)
    print(url)


if __name__ == "__main__":
    main()
