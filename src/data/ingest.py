"""Turn review-service exports (JSONL of ReviewEvent) into image + label records.

Trust policy:
  * decided_by == "auto"    → images are indexed, but NO labels: nobody verified them.
  * decided_by == "officer" → unchanged value = OFFICER_CONFIRM, changed value = OFFICER_EDIT.
  * decided_by == "audit"   → AUDIT labels for every final field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.data.labelstore import append_images, append_labels
from src.schemas import (
    ImageRecord,
    LabelField,
    LabelRecord,
    LabelSource,
    ReviewEvent,
)
from src.text.normalize import normalize_for_eval, normalize_tracking


@dataclass
class IngestStats:
    events: int = 0
    invalid: int = 0
    auto_skipped: int = 0
    images: int = 0
    labels: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def group_key_for(event: ReviewEvent) -> str:
    """Retakes/duplicates of one parcel share the tracking number → same split."""
    tracking = event.final.get(LabelField.TRACKING_NUMBER) or event.predicted.get(
        LabelField.TRACKING_NUMBER
    )
    if tracking and tracking.value:
        return f"{event.property_id}:{normalize_tracking(tracking.value)}"
    return f"job:{event.job_id}"


def event_to_records(event: ReviewEvent) -> tuple[ImageRecord, list[LabelRecord]]:
    image = ImageRecord(
        image_sha256=event.image_sha256,
        uri=event.image_uri,
        job_id=event.job_id,
        group_key=group_key_for(event),
        property_id=event.property_id,
        capture_mode=event.capture_mode,
        captured_at=event.captured_at,
    )
    if event.decided_by == "auto":
        return image, []

    labels = []
    for fld, final in event.final.items():
        if event.decided_by == "audit":
            source = LabelSource.AUDIT
        else:
            predicted = event.predicted.get(fld)
            unchanged = predicted is not None and normalize_for_eval(
                fld, predicted.value
            ) == normalize_for_eval(fld, final.value)
            source = LabelSource.OFFICER_CONFIRM if unchanged else LabelSource.OFFICER_EDIT
        labels.append(
            LabelRecord(
                label_id=LabelRecord.make_id(event.job_id, fld, source, final.value),
                image_sha256=event.image_sha256,
                field=fld,
                value=final.value,
                polygon=final.polygon,
                source=source,
                author_id=event.reviewer_id or "unknown",
                created_at=event.decided_at,
            )
        )
    return image, labels


def ingest_jsonl(path: Path, images_root: Path, labels_root: Path) -> IngestStats:
    stats = IngestStats()
    images: list[ImageRecord] = []
    labels: list[LabelRecord] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            stats.events += 1
            try:
                event = ReviewEvent.model_validate(json.loads(line))
            except Exception as exc:  # keep going; report bad lines
                stats.invalid += 1
                stats.errors.append(f"line {lineno}: {exc}")
                continue
            image, recs = event_to_records(event)
            images.append(image)
            if event.decided_by == "auto":
                stats.auto_skipped += 1
            labels.extend(recs)
            for r in recs:
                stats.by_source[r.source] = stats.by_source.get(r.source, 0) + 1

    append_images(images, images_root)
    append_labels(labels, labels_root)
    stats.images, stats.labels = len(images), len(labels)
    return stats
