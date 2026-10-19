"""Run a bundle over one split of the snapshot → fields.parquet + jobs.parquet (train image).

Rosters: data/rosters/<property_id>.json (DVC-tracked export of units/residents at pilot time).
Without a roster the job cannot be matched and is scored as review.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.split import ForbiddenSplitError
from src.data.store import ImageStore
from src.inference.matching import Resident, Unit
from src.inference.pipeline import InferencePipeline
from src.schemas import Split


def load_roster(path: Path) -> list[Unit]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Unit(
            unit_id=u["unit_id"],
            labels=u["labels"],
            residents=[
                Resident(r["recipient_id"], r["names"], r.get("has_account", True))
                for r in u["residents"]
            ],
        )
        for u in data["units"]
    ]


def predict_split(
    bundle_dir: Path,
    snapshot_dir: Path,
    split: str,
    store: ImageStore,
    rosters_dir: Path,
    out_dir: Path,
    allow_acceptance: bool = False,
) -> dict:
    if split == Split.ACCEPTANCE.value and not allow_acceptance:
        raise ForbiddenSplitError("acceptance set is for UAT only — pass --allow-acceptance at UAT")
    images = pd.read_parquet(snapshot_dir / "images.parquet")
    images = images[images["split"] == split]
    pipeline = InferencePipeline.from_bundle(bundle_dir)
    rosters: dict[str, list[Unit] | None] = {}

    field_rows, job_rows = [], []
    for row in images.itertuples():
        if row.property_id not in rosters:
            p = rosters_dir / f"{row.property_id}.json"
            rosters[row.property_id] = load_roster(p) if p.exists() else None
        result = pipeline.run(store.local_path(row.image_sha256, row.uri), rosters[row.property_id])
        job_rows.append(
            {
                "image_sha256": row.image_sha256,
                "decision": result.decision,
                "reasons": result.reasons,
                "latency_ms": result.latency_ms,
                "bundle_id": result.bundle_id,
            }
        )
        for name, f in result.fields.items():
            field_rows.append(
                {
                    "image_sha256": row.image_sha256,
                    "field": name,
                    "value": f["value"],
                    "score": f["score"],
                    "confidence": f["confidence"],
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        field_rows, columns=["image_sha256", "field", "value", "score", "confidence"]
    ).to_parquet(out_dir / "fields.parquet", index=False)
    pd.DataFrame(
        job_rows, columns=["image_sha256", "decision", "reasons", "latency_ms", "bundle_id"]
    ).to_parquet(out_dir / "jobs.parquet", index=False)
    return {"split": split, "n_jobs": len(job_rows), "bundle_id": pipeline.manifest.bundle_id}
