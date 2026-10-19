"""Export a snapshot into framework-specific training formats.

  rec/        PaddleOCR recognition: crops/*.png + rec_gt_<split>.txt  ("path\\ttext")
  type_size/  CSV per split: image_path,label_index   (label_index = class id - 1)

Only train/val (and calibration, for scoring) can be exported; frozen splits are refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.split import guard_training_splits
from src.data.store import ImageStore
from src.schemas import LabelField, Split

# Line-level printed fields worth training the recognizer on. Address is usually multi-line;
# include it once labels carry per-line polygons.
REC_FIELDS = (LabelField.RECIPIENT_NAME, LabelField.UNIT, LabelField.TRACKING_NUMBER)
EXPORT_SPLITS = [Split.TRAIN.value, Split.VAL.value, Split.CALIBRATION.value]


def crop_polygon(image: np.ndarray, polygon: list[list[float]]) -> np.ndarray:
    """Perspective-crop a 4-point text polygon to an upright line image."""
    pts = np.asarray(polygon, dtype=np.float32)
    if len(pts) != 4:
        x, y, w, h = cv2.boundingRect(pts)
        return image[y : y + h, x : x + w]
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    crop = cv2.warpPerspective(
        image,
        cv2.getPerspectiveTransform(pts, dst),
        (w, h),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if h > 0 and h / max(w, 1) >= 1.5:  # vertical text line
        crop = np.rot90(crop)
    return crop


def export_rec(
    images: pd.DataFrame,
    labels: pd.DataFrame,
    store: ImageStore,
    out_dir: Path,
    splits: list[str] = EXPORT_SPLITS,
) -> dict[str, int]:
    guard_training_splits(splits)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    uri_of = images.set_index("image_sha256")["uri"]

    rec = labels[
        labels["split"].isin(splits)
        & labels["field"].isin([f.value for f in REC_FIELDS])
        & labels["value"].notna()
        & labels["polygon"].notna()
    ]
    counts: dict[str, int] = {}
    lines: dict[str, list[str]] = {s: [] for s in splits}
    for sha, group in rec.groupby("image_sha256"):
        image = cv2.imread(str(store.local_path(sha, uri_of[sha])))
        if image is None:
            continue
        for row in group.itertuples():
            if "\n" in row.value:
                continue
            crop = crop_polygon(image, [list(p) for p in row.polygon])
            if crop.size == 0:
                continue
            name = f"{sha[:16]}_{row.field}.png"
            cv2.imwrite(str(crops_dir / name), crop)
            lines[row.split].append(f"crops/{name}\t{row.value}")
            counts[row.split] = counts.get(row.split, 0) + 1

    for split, rows in lines.items():
        (out_dir / f"rec_gt_{split}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
        )
    return counts


def export_type_size(
    images: pd.DataFrame,
    labels: pd.DataFrame,
    store: ImageStore,
    out_dir: Path,
    splits: list[str] = EXPORT_SPLITS,
) -> dict[str, int]:
    guard_training_splits(splits)
    out_dir.mkdir(parents=True, exist_ok=True)
    uri_of = images.set_index("image_sha256")["uri"]
    ts = labels[
        labels["split"].isin(splits)
        & (labels["field"] == LabelField.TYPE_SIZE.value)
        & labels["value"].notna()
    ]
    counts = {}
    for split in splits:
        part = ts[ts["split"] == split]
        rows = [
            {
                "image_path": str(
                    store.local_path(r.image_sha256, uri_of[r.image_sha256]).resolve()
                ),
                "image_sha256": r.image_sha256,
                "label_index": int(r.value) - 1,
            }
            for r in part.itertuples()
        ]
        pd.DataFrame(rows, columns=["image_path", "image_sha256", "label_index"]).to_csv(
            out_dir / f"type_size_{split}.csv", index=False
        )
        counts[split] = len(rows)
    return counts


def export_all(snapshot_dir: Path, store: ImageStore, out_dir: Path) -> dict[str, dict[str, int]]:
    images = pd.read_parquet(snapshot_dir / "images.parquet")
    labels = pd.read_parquet(snapshot_dir / "labels.parquet")
    stats = {
        "rec": export_rec(images, labels, store, out_dir / "rec"),
        "type_size": export_type_size(images, labels, store, out_dir / "type_size"),
    }
    (out_dir / "export_stats.json").write_text(json.dumps(stats, indent=2))
    return stats
