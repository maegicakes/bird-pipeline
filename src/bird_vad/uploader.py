# src/bird_vad/uploader.py

from __future__ import annotations

import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError


@dataclass(frozen=True)
class S3Settings:
    """
    S3-compatible uploader modeled after bird-files-main/record/upload_to_s3.py env style.

    Env vars:
      - S3_BUCKET (required if uploading)
      - S3_PREFIX (optional)
      - S3_ENDPOINT (optional, for MinIO/R2/etc)
      - AWS_REGION or AWS_DEFAULT_REGION (optional)

      - UPLOAD_DELETE (optional): 1/0 whether to delete local files after successful upload

    Credentials are read by boto3 from:
      - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
      - or ~/.aws/credentials
      - or other default provider chains
    """
    bucket: str
    prefix: str = ""
    endpoint_url: Optional[str] = None
    region: str = "us-east-1"
    delete_after_upload: bool = False
    max_retries: int = 5
    retry_backoff_s: float = 0.8


def load_s3_settings_from_env() -> S3Settings:
    bucket = os.getenv("S3_BUCKET", "").strip()
    if not bucket:
        raise ValueError("S3_BUCKET is required to upload results/clips.")

    prefix = os.getenv("S3_PREFIX", "").strip()

    endpoint = os.getenv("S3_ENDPOINT")
    endpoint = endpoint.strip() if endpoint else None

    region = (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or ("auto" if endpoint else "us-east-1")
    )

    delete_after = os.getenv("UPLOAD_DELETE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}

    return S3Settings(
        bucket=bucket,
        prefix=prefix,
        endpoint_url=endpoint,
        region=region,
        delete_after_upload=delete_after,
    )


def _make_s3_client(settings: S3Settings):
    # More tolerant of spotty Wi-Fi/ethernet; similar spirit to bird-files uploader.
    boto_cfg = BotoConfig(
        retries={"max_attempts": 10, "mode": "standard"},
        connect_timeout=10,
        read_timeout=60,
    )
    return boto3.client(
        "s3",
        region_name=settings.region,
        endpoint_url=settings.endpoint_url,
        config=boto_cfg,
    )


def _guess_content_type(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".json":
        return "application/json"
    if suf == ".wav":
        return "audio/wav"
    ct, _ = mimetypes.guess_type(str(path))
    return ct or "application/octet-stream"


def _join_prefix(prefix: str, name: str) -> str:
    prefix = (prefix or "").strip("/")
    return f"{prefix}/{name}" if prefix else name


def upload_file(path: Path, *, key_name: Optional[str] = None, settings: Optional[S3Settings] = None) -> str:
    """
    Upload any single file (JSON, WAV clips, etc.) to S3.

    Returns the object key that was uploaded.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    settings = settings or load_s3_settings_from_env()
    s3 = _make_s3_client(settings)

    key = _join_prefix(settings.prefix, key_name or path.name)
    content_type = _guess_content_type(path)

    last_err: Optional[Exception] = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            s3.upload_file(
                Filename=str(path),
                Bucket=settings.bucket,
                Key=key,
                ExtraArgs={"ContentType": content_type},
            )
            if settings.delete_after_upload:
                path.unlink(missing_ok=True)
            return key
        except (ClientError, BotoCoreError, OSError) as e:
            last_err = e
            time.sleep(settings.retry_backoff_s * attempt)

    raise RuntimeError(f"Failed to upload {path} after {settings.max_retries} attempts") from last_err


def upload_json_result(json_path: Path, settings: Optional[S3Settings] = None) -> str:
    """
    Upload exactly one JSON result file to S3.
    Returns the object key.
    """
    if json_path.suffix.lower() != ".json":
        raise ValueError(f"Refusing to upload non-json file as result: {json_path}")
    return upload_file(json_path, settings=settings)


def upload_wav_clip(wav_path: Path, settings: Optional[S3Settings] = None) -> str:
    """
    Upload exactly one WAV clip to S3.
    Returns the object key.
    """
    if wav_path.suffix.lower() != ".wav":
        raise ValueError(f"Refusing to upload non-wav file as clip: {wav_path}")
    return upload_file(wav_path, settings=settings)


def upload_wav_clips_for_chunk(clips_dir: Path, chunk_id: str, settings: Optional[S3Settings] = None) -> int:
    """
    Upload all WAV clips for a chunk_id from clips_dir.
    Expects naming like: <chunk_id>_000.wav, <chunk_id>_001.wav, ...

    Returns number of uploaded clips.
    """
    settings = settings or load_s3_settings_from_env()
    clips_dir = clips_dir.expanduser().resolve()
    if not clips_dir.exists():
        return 0

    count = 0
    for wav in sorted(clips_dir.glob(f"{chunk_id}_*.wav")):
        upload_wav_clip(wav, settings=settings)
        count += 1
    return count


def upload_results_dir(results_dir: Path, settings: Optional[S3Settings] = None) -> int:
    """
    Upload all *.json files in results_dir (non-recursive).
    Returns number of uploaded files.
    """
    settings = settings or load_s3_settings_from_env()
    results_dir = results_dir.expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(str(results_dir))

    count = 0
    for p in sorted(results_dir.glob("*.json")):
        upload_json_result(p, settings=settings)
        count += 1
    return count
