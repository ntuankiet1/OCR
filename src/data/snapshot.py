"""Build an immutable dataset snapshot from the append-only stores.

Output (DVC-tracked):
  images.parquet  — one row per image: split, group_key, property, capture mode, carrier
  labels.parquet  — one resolved label per (image, field)
  stats.json      — counts, used as DVC metrics and by the retrain trigger
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.labelstore import read_images, read_labels
from src.data.split import assign_splits, check_no_leakage, load_frozen
from src.schemas import FROZEN_SPLITS, SOURCE_PRECEDENCE, LabelField, LabelSource, Split

_FROZEN = [s.value for s in FROZEN_SPLITS]


def resolve_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Pick one label per (image, field): highest source precedence, then most recent."""
    df = labels.assign(
        _rank=labels["source"].map({s.value: p for s, p in SOURCE_PRECEDENCE.items()}),
        _ts=pd.to_datetime(labels["created_at"], utc=True, format="ISO8601"),
    )
    df = df.sort_values(["_rank", "_ts"], ascending=[False, False])
    return (
        df.drop_duplicates(subset=["image_sha256", "field"], keep="first")
        .drop(columns=["_rank", "_ts"])
        .reset_index(drop=True)
    )


def filter_by_split_policy(labels: pd.DataFrame, split_of: pd.Series) -> pd.DataFrame:
    """Frozen sets use only adjudicated (two-reviewer) labels; synthetic labels only in train."""
    split = labels["image_sha256"].map(split_of)
    is_gold = labels["source"] == LabelSource.ADJUDICATED.value
    is_synth = labels["source"] == LabelSource.SYNTHETIC.value
    keep = (~split.isin(_FROZEN) | is_gold) & (~is_synth | (split == Split.TRAIN.value))
    return labels[keep]


def build_snapshot(
    images_root: Path,
    labels_root: Path,
    frozen_dir: Path,
    out_dir: Path,
    split_params: dict[str, Any],
) -> dict[str, Any]:
    images = read_images(images_root)
    raw_labels = read_labels(labels_root)
    frozen = load_frozen(frozen_dir)

    images["split"] = assign_splits(
        images,
        frozen,
        salt=split_params["salt"],
        val_fraction=split_params["val_fraction"],
        calibration_fraction=split_params["calibration_fraction"],
    )
    check_no_leakage(images)
    split_of = images.set_index("image_sha256")["split"]

    labels = resolve_labels(filter_by_split_policy(raw_labels, split_of))
    labels["split"] = labels["image_sha256"].map(split_of)

    # Carrier is the main reporting/gating subgroup; denormalize it onto images.
    carrier = labels.loc[labels["field"] == LabelField.CARRIER.value].set_index("image_sha256")[
        "value"
    ]
    images["carrier"] = images["image_sha256"].map(carrier).fillna("unknown")

    is_frozen = images["split"].isin(_FROZEN)
    gold = labels.loc[labels["source"] == LabelSource.ADJUDICATED.value, "image_sha256"]
    stats = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_images": len(images),
        "n_labels_raw": len(raw_labels),
        "n_labels_resolved": len(labels),
        "n_verified_labels": int((raw_labels["source"] != LabelSource.SYNTHETIC.value).sum()),
        "images_per_split": images["split"].value_counts().to_dict(),
        "labels_per_source": raw_labels["source"].value_counts().to_dict(),
        "frozen_groups_expected": {s.value: len(frozen[s]) for s in FROZEN_SPLITS},
        # Frozen images without two-reviewer labels cannot be scored — fix before gating/UAT.
        "frozen_images_without_gold": int((is_frozen & ~images["image_sha256"].isin(gold)).sum()),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    images.to_parquet(out_dir / "images.parquet", index=False)
    labels.to_parquet(out_dir / "labels.parquet", index=False)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def load_snapshot(snapshot_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(snapshot_dir / "images.parquet"),
        pd.read_parquet(snapshot_dir / "labels.parquet"),
    )
