"""Score a prediction run against the gold labels of one split.

Prediction run layout (written by `parcel-ocr predict`):
  fields.parquet  image_sha256, field, value, score
  jobs.parquet    image_sha256, decision ("auto"|"review"), reasons, latency_ms, bundle_id

Evaluation output:
  metrics.json         headline + per-field + per-subgroup metrics (DVC metrics)
  correctness.parquet  image_sha256, field, correct — paired input for the gate's McNemar test
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.data.split import frozen_set_hash
from src.evaluation.metrics import (
    error_rate_upper_bound,
    field_accuracy,
    join_field_predictions,
    latency_summary,
    whole_record_accuracy,
)
from src.schemas import ENTITY_FIELDS, FROZEN_SPLITS, LabelField, Split
from src.text.normalize import normalize_for_eval

# AC01 scope: readable recipient, address/unit, carrier and tracking.
OCR_FIELDS = [
    LabelField.RECIPIENT_NAME.value,
    LabelField.UNIT.value,
    LabelField.ADDRESS.value,
    LabelField.CARRIER.value,
    LabelField.TRACKING_NUMBER.value,
]
SCORED_FIELDS = OCR_FIELDS + [LabelField.TYPE_SIZE.value]
SUBGROUPS = ["carrier", "capture_mode", "property_id"]


def wrong_assignments(jobs: pd.DataFrame, pred: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Auto-registered jobs whose unit_id or recipient_id differs from gold."""
    auto = jobs.loc[jobs["decision"] == "auto", ["image_sha256"]]
    entity = [f.value for f in ENTITY_FIELDS]
    g = gold[gold["field"].isin(entity)].pivot(
        index="image_sha256", columns="field", values="value"
    )
    p = pred[pred["field"].isin(entity)].pivot(
        index="image_sha256", columns="field", values="value"
    )
    rows = []
    for sha in auto["image_sha256"]:
        for f in entity:
            gv = g.at[sha, f] if sha in g.index and f in g.columns else None
            pv = p.at[sha, f] if sha in p.index and f in p.columns else None
            gv, pv = (None if pd.isna(v) else v for v in (gv, pv))
            if gv is not None and normalize_for_eval(f, pv) != normalize_for_eval(f, gv):
                rows.append({"image_sha256": sha, "field": f, "pred": pv, "gold": gv})
    return pd.DataFrame(rows, columns=["image_sha256", "field", "pred", "gold"])


def evaluate(
    pred_dir: Path, snapshot_dir: Path, split: str, frozen_dir: Path, out_dir: Path
) -> dict:
    images = pd.read_parquet(snapshot_dir / "images.parquet")
    labels = pd.read_parquet(snapshot_dir / "labels.parquet")
    pred = pd.read_parquet(pred_dir / "fields.parquet")
    jobs = pd.read_parquet(pred_dir / "jobs.parquet")

    images = images[images["split"] == split]
    gold = labels[labels["split"] == split]
    jobs = jobs[jobs["image_sha256"].isin(images["image_sha256"])]
    pred = pred[pred["image_sha256"].isin(images["image_sha256"])]

    joined = join_field_predictions(pred, gold, SCORED_FIELDS)
    joined = joined.merge(images[["image_sha256", *SUBGROUPS]], on="image_sha256", how="left")
    wrong = wrong_assignments(jobs, pred, gold)
    n_jobs = len(images)  # denominator = every image that entered evaluation
    n_auto = int((jobs["decision"] == "auto").sum())

    subgroups: dict[str, dict] = {}
    for col in SUBGROUPS:
        for value, part in joined.groupby(col):
            subgroups[f"{col}={value}"] = field_accuracy(part)

    metrics = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": split,
        "eval_set_hash": frozen_set_hash(frozen_dir, Split(split))
        if Split(split) in FROZEN_SPLITS
        else None,
        "bundle_ids": sorted(jobs["bundle_id"].dropna().unique().tolist()),
        "n_jobs": n_jobs,
        "n_predicted_jobs": len(jobs),
        "auto_rate": n_auto / n_jobs if n_jobs else 0.0,
        "n_auto": n_auto,
        "wrong_assignments": len(wrong),
        "wrong_assignment_rate_ub95": error_rate_upper_bound(len(wrong), max(n_auto, 1)),
        "whole_record_accuracy": whole_record_accuracy(joined, OCR_FIELDS),
        "fields": field_accuracy(joined),
        "subgroups": subgroups,
        "latency_ms": latency_summary(jobs["latency_ms"]),
        "review_reasons": (
            jobs.loc[jobs["decision"] == "review", "reasons"].explode().value_counts().to_dict()
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    joined[["image_sha256", "field", "readable", "correct", *SUBGROUPS]].to_parquet(
        out_dir / "correctness.parquet", index=False
    )
    wrong.to_parquet(out_dir / "wrong_assignments.parquet", index=False)
    return metrics
