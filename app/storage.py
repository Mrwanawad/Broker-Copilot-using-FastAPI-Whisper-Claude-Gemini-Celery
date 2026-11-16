"""Where uploaded audio lives. Local disk by default; S3-compatible object storage (AWS S3, Cloudflare R2,
MinIO) when BC_STORAGE_BACKEND=s3, so the API and the worker can run on separate machines.

`Call.source_path` holds the reference: a filesystem path, or s3://bucket/key.
"""

import uuid
from functools import lru_cache
from pathlib import Path

from app.config import get_settings


@lru_cache
def _s3():
    import boto3  # `s3` extra

    s = get_settings()
    return boto3.client("s3", endpoint_url=s.s3_endpoint_url or None, region_name=s.s3_region or None)


def put(data: bytes, suffix: str) -> str:
    s = get_settings()
    name = f"{uuid.uuid4().hex}{suffix}"
    if s.storage_backend == "s3":
        key = f"{s.s3_prefix.strip('/')}/{name}".lstrip("/")
        _s3().put_object(Bucket=s.s3_bucket, Key=key, Body=data)
        return f"s3://{s.s3_bucket}/{key}"
    d = Path(s.upload_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    path.write_bytes(data)
    return str(path)


def fetch(ref: str, workdir: Path) -> Path:
    """Local path for a stored file; object-storage files are downloaded into `workdir` first."""
    if not ref.startswith("s3://"):
        return Path(ref)
    bucket, key = ref[5:].split("/", 1)
    workdir.mkdir(parents=True, exist_ok=True)
    local = workdir / Path(key).name
    if not local.exists():
        _s3().download_file(bucket, key, str(local))
    return local
