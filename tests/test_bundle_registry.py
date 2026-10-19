from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest

from src.evaluation.gate import GateReport
from src.registry import mlflow_registry as reg
from src.registry.bundle import assemble_bundle, read_manifest, verify_bundle

ROOT = Path(__file__).parents[1]


def make_bundle(tmp: Path, name: str, rec_payload: str) -> Path:
    comp = tmp / f"{name}_rec"
    comp.mkdir()
    (comp / "inference.json").write_text(rec_payload)
    cal = tmp / f"{name}_cal.json"
    cal.write_text(json.dumps({"fields": {"unit": {"threshold": 0.93}}}))
    out = tmp / name
    assemble_bundle(
        out,
        {"rec": (comp, "paddle-inference", "test")},
        cal,
        [ROOT / "configs" / "carriers.yaml"],
        data_snapshot="snap1",
        pipeline_meta={"params": {}},
    )
    return out


def test_bundle_id_is_content_addressed(tmp_path):
    a = read_manifest(make_bundle(tmp_path, "a", "weights-v1"))
    b = read_manifest(make_bundle(tmp_path, "b", "weights-v1"))
    c = read_manifest(make_bundle(tmp_path, "c", "weights-v2"))
    assert a.bundle_id == b.bundle_id != c.bundle_id
    assert a.thresholds == {"unit": 0.93}


def test_tampering_is_detected(tmp_path):
    bundle = make_bundle(tmp_path, "a", "weights")
    assert verify_bundle(bundle) == []
    (bundle / "models" / "rec" / "inference.json").write_text("swapped")
    (bundle / "pipeline.json").write_text("{}")
    problems = verify_bundle(bundle)
    assert any("rec" in p for p in problems) and any("pipeline.json" in p for p in problems)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(reg.EXPERIMENT, artifact_location=(tmp_path / "mlartifacts").as_uri())
    gate = tmp_path / "gate.json"
    gate.write_text(
        GateReport(
            passed=True, mode="first_model", challenger_bundles=[], champion_bundles=[], checks=[]
        ).model_dump_json()
    )
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "auto_rate": 0.9,
                "wrong_assignments": 0,
                "whole_record_accuracy": 0.95,
                "latency_ms": {"p95": 400},
                "fields": {"unit": {"accuracy": 0.99}},
                "eval_set_hash": "h1",
            }
        )
    )
    return tmp_path, gate, metrics


def test_register_promote_rollback(registry):
    tmp, gate, metrics = registry
    name = "parcel-ocr-bundle-test"
    v1 = reg.register_bundle(make_bundle(tmp, "b1", "w1"), gate, metrics, name)
    v2 = reg.register_bundle(make_bundle(tmp, "b2", "w2"), gate, metrics, name)
    assert reg.status(name)["challenger"]["version"] == v2

    with pytest.raises(reg.RegistryError):
        reg.promote(name, v1, approved_by="  ")  # supervised: approver required

    reg.promote(name, v1, approved_by="lead")
    reg.promote(name, v2, approved_by="lead")
    s = reg.status(name)
    assert (
        s["champion"]["version"] == v2
        and s["previous"]["version"] == v1
        and s["challenger"] is None
    )

    out = reg.rollback(name, reason="wrong unit matches in pilot", by="ops")
    s = reg.status(name)
    assert out == {"champion": v1, "rolled_back": v2}
    assert s["champion"]["version"] == v1 and s["previous"] is None
    with pytest.raises(reg.RegistryError):
        reg.rollback(name, reason="again", by="ops")

    # the champion's bundle can be fetched back for re-evaluation
    bundle = reg.download_bundle(name, reg.CHAMPION, tmp / "dl")
    assert read_manifest(bundle).bundle_id == read_manifest(tmp / "b1").bundle_id


def test_failed_gate_is_not_registered(registry):
    tmp, gate, metrics = registry
    gate.write_text(
        GateReport(
            passed=False, mode="first_model", challenger_bundles=[], champion_bundles=[], checks=[]
        ).model_dump_json()
    )
    with pytest.raises(reg.RegistryError):
        reg.register_bundle(make_bundle(tmp, "b1", "w1"), gate, metrics, "m")
