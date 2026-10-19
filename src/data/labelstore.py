"""Append-only Parquet storage for image records and label records.

Layout:  <root>/ingested=<YYYY-MM-DD>/part-<batch>.parquet
Files are never rewritten. The directory is DVC-tracked, so every snapshot pins exactly
which label batches it saw.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from src.schemas import ImageRecord, LabelRecord


def _append(records: Iterable[BaseModel], root: Path) -> Path | None:
    rows = [r.model_dump(mode="json") for r in records]
    if not rows:
        return None
    part_dir = root / f"ingested={datetime.now(UTC):%Y-%m-%d}"
    part_dir.mkdir(parents=True, exist_ok=True)
    path = part_dir / f"part-{uuid.uuid4().hex[:12]}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def append_labels(records: Iterable[LabelRecord], root: Path) -> Path | None:
    return _append(records, root)


def append_images(records: Iterable[ImageRecord], root: Path) -> Path | None:
    return _append(records, root)


def _read(root: Path, columns: list[str], key: str) -> pd.DataFrame:
    files = sorted(root.glob("ingested=*/part-*.parquet"))
    if not files:
        return pd.DataFrame(columns=columns)
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    # Re-ingesting the same export yields identical deterministic ids → drop them here.
    return df.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)


def read_labels(root: Path) -> pd.DataFrame:
    return _read(root, list(LabelRecord.model_fields), "label_id")


def read_images(root: Path) -> pd.DataFrame:
    return _read(root, list(ImageRecord.model_fields), "image_sha256")
