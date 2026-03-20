#!/usr/bin/env python3
from datetime import datetime, timezone
import os
import sys
import argparse
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed

import redis
import json
import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

# ---------- S3 helpers ----------

def get_s3():
    """
    Build an S3 client that works for AWS S3, Cloudflare R2, or MinIO via env.
    Env:
      S3_ENDPOINT (optional): e.g. https://<accountid>.r2.cloudflarestorage.com
      AWS_REGION / AWS_DEFAULT_REGION: for AWS. For R2, 'auto' is OK.
    """
    endpoint = os.getenv("S3_ENDPOINT")  # None for AWS S3
    # Prefer explicit region if set; default 'us-east-1' is safer than 'auto' on AWS.
    region = (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or ("auto" if endpoint else "us-east-1")
    )

    # You can switch addressing style if needed:
    # s3={'addressing_style': 'virtual'} is fine for AWS/R2 bucket names.
    cfg = Config(
        signature_version="s3v4",
        retries={"max_attempts": 5, "mode": "standard"},
        s3={"addressing_style": "virtual"},
    )
    return boto3.client("s3", endpoint_url=endpoint, region_name=region, config=cfg)

def guess_content_type(path: str) -> str | None:
    # Good enough defaults for .wav/.flac/.zip plus fallback to mimetypes
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return "audio/wav"
    if ext == ".flac":
        return "audio/flac"
    if ext == ".zip":
        return "application/zip"
    ctype, _ = mimetypes.guess_type(path)
    return ctype or None

def s3_upload(s3, bucket: str, local_path: str, key: str) -> bool:
    extra = {}
    ctype = guess_content_type(local_path)
    if ctype:
        extra["ContentType"] = ctype

    try:
        s3.upload_file(local_path, bucket, key, ExtraArgs=extra or None)
        print(f"[s3] Uploaded s3://{bucket}/{key}")
        return True
    except (BotoCoreError, ClientError) as e:
        print(f"[s3] Upload failed for {local_path} -> {key}: {e}")
        return False

# ---------- Main logic ----------

def collect_files(root: str) -> list[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            # Upload only relevant types from your pipeline; tweak as needed
            if fn.lower().endswith((".wav", ".flac", ".zip", ".txt")):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)

def key_for(local_path: str, base_dir: str, prefix: str = "") -> str:
    rel = os.path.relpath(local_path, start=base_dir).replace("\\", "/")
    if prefix:
        return f"{prefix.rstrip('/')}/{rel}"
    return rel

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

# Redis
def make_redis_from_env():
    host = os.getenv("REDIS_HOST")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD")

    if not host:
        raise RuntimeError("REDIS_HOST not set")
    return redis.Redis(host=host, port=port, password=password, decode_responses=True)

def main():
    parser = argparse.ArgumentParser(description="Upload a directory of files to S3/R2/MinIO.")
    parser.add_argument("--dir", required=True, help="Directory of files to upload")
    parser.add_argument("--delete", action="store_true", help="Delete local file if upload succeeds")
    parser.add_argument("--prefix", default=os.getenv("S3_PREFIX", ""), help="S3 key prefix")
    parser.add_argument("--workers", type=int, default=4, help="Parallel upload workers")
    
    parser.add_argument("--device-id", required=True, help="Device identifier (e.g. deviceA)")
    parser.add_argument("--queue", default=os.getenv("REDIS_QUEUE", "bird_jobs"), help="Redis list name")

    args = parser.parse_args()

    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        print("[s3] S3_BUCKET not set; aborting")
        return 2

    base_dir = os.path.abspath(args.dir)
    if not os.path.isdir(base_dir):
        print(f"[s3] Not a directory: {base_dir}")
        return 2

    files = collect_files(base_dir)
    if not files:
        print(f"[s3] No matching files found in {base_dir}")
        return 0

    rds = None
    try: 
        rds = make_redis_from_env()
        rds.ping()
        print("[redis] Connected")
    except Exception as e:
        print("[redis] Not connected: {e}")
        rds = None 

    s3 = get_s3()
    ok = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {}
        for path in files:
            key = key_for(path, base_dir, args.prefix)
            fut = pool.submit(s3_upload, s3, bucket, path, key)
            futs[fut] = path

        for fut in as_completed(futs):
            path = futs[fut]
            try:
                success = fut.result()
            except Exception as e:
                print(f"[s3] Unexpected error uploading {path}: {e}")
                success = False
            
            if success:
                ok += 1

                if rds and key.lower().endswith((".flac", ".wav")):
                    job = {
                        "recording_id": key,          # simple idempotency key
                        "device_id": args.device_id,
                        "bucket": args.bucket,
                        "key": key,
                        "recorded_at": now_iso_utc(),
                    }
                    
                    rds.rpush(args.queue, json.dumps(job))
                    print("[redis] Uploaded + enqueued:", job)
                if args.delete:
                    try:
                        os.remove(path)
                        print(f"[local] Deleted {path}")
                    except OSError as e:
                        print(f"[local] Failed to delete {path}: {e}")
            else:
                fail += 1

    print(f"[s3] Done. Success={ok} Fail={fail}")
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
