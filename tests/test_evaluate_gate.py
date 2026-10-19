from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.split import write_frozen
from src.evaluation.evaluate import evaluate
from src.evaluation.gate import EvalRun, GateConfig, run_gate
from src.schemas import Split

N = 400
FIELDS = {
    "tracking_number": "EF{i:09d}TH",
    "unit": "99/{i}",
    "recipient_name": "name {i}",
    "carrier": "thailand_post",
    "unit_id": "U{i}",
    "recipient_id": "R{i}",
}


def make_snapshot(root: Path) -> tuple[Path, Path]:
    snap, frozen = root / "snap", root / "frozen"
    snap.mkdir(parents=True)
    shas = [f"{i:064x}" for i in range(N)]
    images = pd.DataFrame(
        {
            "image_sha256": shas,
            "group_key": [f"g{i}" for i in range(N)],
            "split": "regression",
            "carrier": ["thailand_post" if i % 2 else "flash" for i in range(N)],
            "capture_mode": "officer",
            "property_id": "P1",
            "uri": "x",
        }
    )
    labels = pd.DataFrame(
        [
            {
                "image_sha256": shas[i],
                "field": f,
                "value": v.format(i=i),
                "source": "adjudicated",
                "split": "regression",
            }
            for i in range(N)
            for f, v in FIELDS.items()
        ]
    )
    images.to_parquet(snap / "images.parquet")
    labels.to_parquet(snap / "labels.parquet")
    write_frozen(frozen, Split.REGRESSION, images["group_key"].tolist(), "test")
    return snap, frozen


def make_run(
    root: Path,
    name: str,
    wrong_tracking: int = 0,
    wrong_unit_auto: int = 0,
    auto_share: float = 0.9,
) -> Path:
    d = root / name
    d.mkdir()
    rows, jobs = [], []
    for i in range(N):
        sha = f"{i:064x}"
        for f, v in FIELDS.items():
            value = v.format(i=i)
            if f == "tracking_number" and i < wrong_tracking:
                value = "WRONG"
            if f == "unit_id" and i < wrong_unit_auto:
                value = "U-other"
            rows.append(
                {"image_sha256": sha, "field": f, "value": value, "score": 0.99, "confidence": 0.99}
            )
        jobs.append(
            {
                "image_sha256": sha,
                "decision": "auto" if i < auto_share * N else "review",
                "reasons": [] if i < auto_share * N else ["low_confidence_unit"],
                "latency_ms": 300.0,
                "bundle_id": name,
            }
        )
    pd.DataFrame(rows).to_parquet(d / "fields.parquet")
    pd.DataFrame(jobs).to_parquet(d / "jobs.parquet")
    return d


@pytest.fixture
def env(tmp_path):
    snap, frozen = make_snapshot(tmp_path)

    def run_eval(name: str, **kw) -> EvalRun:
        pred = make_run(tmp_path, name, **kw)
        evaluate(pred, snap, "regression", frozen, tmp_path / f"eval_{name}")
        return EvalRun.load(tmp_path / f"eval_{name}")

    return run_eval


def test_auto_rate_denominator_is_all_jobs(env):
    m = env("a", auto_share=0.5).metrics
    assert m["n_jobs"] == N and m["auto_rate"] == pytest.approx(0.5)
    assert m["review_reasons"] == {"low_confidence_unit": N // 2}


def test_wrong_assignment_blocks_even_first_model(env):
    report = run_gate(env("bad", wrong_unit_auto=1), None, GateConfig())
    assert not report.passed and report.mode == "first_model"
    assert [c.name for c in report.checks if not c.passed] == ["wrong_assignments"]


def test_equal_challenger_passes(env):
    report = run_gate(env("new"), env("old"), GateConfig())
    assert report.passed, [c for c in report.checks if not c.passed]


def test_field_regression_fails(env):
    report = run_gate(env("new", wrong_tracking=20), env("old"), GateConfig())
    failed = {c.name for c in report.checks if not c.passed}
    assert "field:tracking_number" in failed
    assert any(n.startswith("subgroup:carrier=") for n in failed)


def test_auto_rate_drop_fails(env):
    report = run_gate(env("new", auto_share=0.8), env("old", auto_share=0.9), GateConfig())
    assert {c.name for c in report.checks if not c.passed} == {"auto_rate"}


def test_different_eval_sets_cannot_be_compared(env):
    old = env("old")
    old.metrics["eval_set_hash"] = "something-else"
    report = run_gate(env("new"), old, GateConfig())
    assert not report.passed and "same_eval_set" in {c.name for c in report.checks if not c.passed}
