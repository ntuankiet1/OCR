"""Data contracts shared by ingest, snapshots, training, evaluation and the registry."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class LabelField(StrEnum):
    # Printed text on the label — OCR targets, evaluated by exact match (AC01).
    RECIPIENT_NAME = "recipient_name"
    UNIT = "unit"
    ADDRESS = "address"
    CARRIER = "carrier"
    TRACKING_NUMBER = "tracking_number"
    # Image class (BRD §7, IDs 1-10).
    TYPE_SIZE = "type_size"
    # Resolved entities chosen by the officer — matching targets, used for wrong-assignment
    # checks (AC03). Not printed text: never used to train the recognizer.
    UNIT_ID = "unit_id"
    RECIPIENT_ID = "recipient_id"


PRINTED_FIELDS = (
    LabelField.RECIPIENT_NAME,
    LabelField.UNIT,
    LabelField.ADDRESS,
    LabelField.CARRIER,
    LabelField.TRACKING_NUMBER,
)
ENTITY_FIELDS = (LabelField.UNIT_ID, LabelField.RECIPIENT_ID)


class LabelSource(StrEnum):
    """Who produced a label. Auto-registered predictions are deliberately NOT a source:
    training on unverified model output would reinforce the model's own mistakes."""

    ADJUDICATED = "adjudicated"  # two reviewers agreed or resolved (AC02); only gold source
    ANNOTATOR = "annotator"  # labeling team (Label Studio)
    AUDIT = "audit"  # sampled human audit of auto-registered jobs
    OFFICER_EDIT = "officer_edit"  # officer changed the predicted value in the review queue
    OFFICER_CONFIRM = "officer_confirm"  # officer confirmed the predicted value unchanged
    SYNTHETIC = "synthetic"  # rendered training data; never used for calibration or eval


# Higher wins when several labels exist for the same (image, field).
SOURCE_PRECEDENCE: dict[LabelSource, int] = {
    LabelSource.ADJUDICATED: 60,
    LabelSource.ANNOTATOR: 50,
    LabelSource.AUDIT: 40,
    LabelSource.OFFICER_EDIT: 30,
    LabelSource.OFFICER_CONFIRM: 20,
    LabelSource.SYNTHETIC: 10,
}


class CaptureMode(StrEnum):
    OFFICER = "officer"
    AUTO_CAMERA = "auto_camera"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"  # checkpoint selection
    CALIBRATION = "calibration"  # confidence calibration + thresholds
    REGRESSION = "regression"  # frozen; champion/challenger gate
    ACCEPTANCE = "acceptance"  # frozen; UAT only — never training, tuning or gating


FROZEN_SPLITS = (Split.REGRESSION, Split.ACCEPTANCE)
TRAINABLE_SPLITS = (Split.TRAIN, Split.VAL)

Polygon = list[tuple[float, float]]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ImageRecord(BaseModel):
    image_sha256: str
    uri: str  # s3://parcel-images/ab/<sha>.jpg or a local path
    job_id: str
    group_key: str  # retakes of one parcel share it → always land in the same split
    property_id: str
    capture_mode: CaptureMode
    captured_at: datetime


class LabelRecord(BaseModel):
    """Append-only. A correction is a new record; snapshots resolve by precedence then time."""

    label_id: str
    image_sha256: str
    field: LabelField
    value: str | None  # None = marked missing/unreadable by a human
    polygon: Polygon | None = None  # text location (printed fields), image pixel coords
    source: LabelSource
    author_id: str  # pseudonymous officer/annotator id — no personal data
    created_at: datetime = Field(default_factory=_utcnow)
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def make_id(*parts: object) -> str:
        """Deterministic id so re-ingesting the same export is idempotent."""
        return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()[:32]


class FieldValue(BaseModel):
    value: str | None = None
    polygon: Polygon | None = None


class ReviewEvent(BaseModel):
    """One registration job as exported by the review service (one JSON object per line).

    This is the contract between the registration app and the training pipeline.
    `predicted` is what the bundle produced; `final` is what was saved after review.
    """

    job_id: str
    image_sha256: str
    image_uri: str
    property_id: str
    capture_mode: CaptureMode
    captured_at: datetime
    decided_by: Literal["officer", "auto", "audit"]
    reviewer_id: str | None = None
    bundle_id: str | None = None
    predicted: dict[LabelField, FieldValue] = Field(default_factory=dict)
    final: dict[LabelField, FieldValue] = Field(default_factory=dict)
    decided_at: datetime = Field(default_factory=_utcnow)
