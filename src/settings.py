"""Runtime settings (env: PARCEL_OCR_*) and params.yaml loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PARCEL_OCR_", extra="ignore")

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    configs_dir: Path = Path("configs")
    params_file: Path = Path("params.yaml")

    # S3-compatible storage (SeaweedFS in docker-compose). Credentials come from AWS_* env vars.
    s3_endpoint_url: str | None = None
    images_bucket: str = "parcel-images"
    buckets: tuple[str, ...] = ("parcel-images", "dvc-store", "mlflow-artifacts")

    registered_model_name: str = "parcel-ocr-bundle"
    paddleocr_home: Path = Path("/opt/PaddleOCR")

    # Derived locations
    @property
    def labels_dir(self) -> Path:
        return self.data_dir / "labels"

    @property
    def images_index_dir(self) -> Path:
        return self.data_dir / "images_index"

    @property
    def frozen_dir(self) -> Path:
        return self.data_dir / "frozen"

    @property
    def snapshot_dir(self) -> Path:
        return self.data_dir / "snapshots" / "current"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def image_cache_dir(self) -> Path:
        return self.data_dir / "images"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_params(path: Path | None = None) -> dict[str, Any]:
    path = path or get_settings().params_file
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(name: str) -> dict[str, Any]:
    with open(get_settings().configs_dir / name, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
