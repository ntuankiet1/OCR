"""Content-addressed image store: S3 (SeaweedFS) or local paths, with a verified local cache.

Images are immutable and keyed by SHA-256, so a dataset snapshot only needs to list hashes.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from urllib.parse import urlparse


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def object_key(sha: str, suffix: str = ".jpg") -> str:
    return f"{sha[:2]}/{sha}{suffix}"


class ImageStore:
    def __init__(self, cache_dir: Path, s3_endpoint_url: str | None = None) -> None:
        self.cache_dir = cache_dir
        self.s3_endpoint_url = s3_endpoint_url
        self._s3 = None

    @property
    def s3(self):
        if self._s3 is None:
            import boto3

            self._s3 = boto3.client("s3", endpoint_url=self.s3_endpoint_url)
        return self._s3

    def local_path(self, sha: str, uri: str) -> Path:
        """Return a cached local copy, downloading and verifying the hash on first use."""
        suffix = Path(urlparse(uri).path).suffix or ".jpg"
        dst = self.cache_dir / object_key(sha, suffix)
        if dst.exists():
            return dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        parsed = urlparse(uri)
        if parsed.scheme == "s3":
            self.s3.download_file(parsed.netloc, parsed.path.lstrip("/"), str(tmp))
        else:
            shutil.copyfile(parsed.path if parsed.scheme == "file" else uri, tmp)
        actual = sha256_file(tmp)
        if actual != sha:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"hash mismatch for {uri}: expected {sha}, got {actual}")
        tmp.rename(dst)
        return dst

    def put(self, path: Path, bucket: str) -> tuple[str, str]:
        """Upload a local image; returns (sha256, s3 uri). Idempotent."""
        sha = sha256_file(path)
        key = object_key(sha, path.suffix.lower() or ".jpg")
        self.s3.upload_file(str(path), bucket, key)
        return sha, f"s3://{bucket}/{key}"


def ensure_buckets(buckets: tuple[str, ...], s3_endpoint_url: str | None) -> list[str]:
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3", endpoint_url=s3_endpoint_url)
    created = []
    for bucket in buckets:
        try:
            s3.head_bucket(Bucket=bucket)
        except ClientError:
            s3.create_bucket(Bucket=bucket)
            created.append(bucket)
    return created
