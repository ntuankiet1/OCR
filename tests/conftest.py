from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


def review_event(
    job: int,
    decided_by: str = "officer",
    tracking: str | None = None,
    predicted: dict | None = None,
    final: dict | None = None,
    property_id: str = "P1",
    minutes: int = 0,
) -> dict:
    tracking = tracking or f"EF{job:09d}TH"
    base = {
        "tracking_number": {"value": tracking, "polygon": [[0, 0], [100, 0], [100, 20], [0, 20]]},
        "recipient_name": {"value": "สมชาย ใจดี"},
        "unit": {"value": "99/123"},
        "carrier": {"value": "thailand_post"},
        "type_size": {"value": "7"},
        "unit_id": {"value": "U-99/123"},
        "recipient_id": {"value": "R-1"},
    }
    ts = datetime(2026, 10, 1, tzinfo=UTC) + timedelta(minutes=minutes)
    return {
        "job_id": f"job-{job}",
        "image_sha256": f"{job:064x}",
        "image_uri": f"/tmp/img-{job}.jpg",
        "property_id": property_id,
        "capture_mode": "officer",
        "captured_at": ts.isoformat(),
        "decided_at": ts.isoformat(),
        "decided_by": decided_by,
        "reviewer_id": "officer-7",
        "bundle_id": "b0",
        "predicted": predicted if predicted is not None else base,
        "final": final if final is not None else base,
    }


@pytest.fixture
def write_jsonl(tmp_path: Path):
    def _write(events: list[dict], name: str = "export.jsonl") -> Path:
        p = tmp_path / name
        p.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events))
        return p

    return _write


@pytest.fixture
def split_params() -> dict:
    return {"salt": "test", "val_fraction": 0.1, "calibration_fraction": 0.2}
