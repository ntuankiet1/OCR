from __future__ import annotations

import pandas as pd
import pytest
from conftest import review_event

from src.data.ingest import ingest_jsonl
from src.data.labelstore import read_labels
from src.data.snapshot import build_snapshot, load_snapshot
from src.data.split import (
    ForbiddenSplitError,
    LeakageError,
    assign_splits,
    guard_training_splits,
    load_frozen,
    write_frozen,
)
from src.schemas import Split


@pytest.fixture
def roots(tmp_path):
    return tmp_path / "images_index", tmp_path / "labels", tmp_path / "frozen", tmp_path / "snap"


def test_ingest_trust_policy(roots, write_jsonl):
    images, labels, *_ = roots
    edited = review_event(2)
    edited["final"] = {**edited["final"], "unit": {"value": "99/124"}}
    stats = ingest_jsonl(
        write_jsonl([review_event(1), edited, review_event(3, decided_by="auto")]), images, labels
    )

    assert stats.auto_skipped == 1 and stats.images == 3
    df = read_labels(labels)
    assert set(df["image_sha256"]) == {f"{1:064x}", f"{2:064x}"}  # auto job has no labels
    unit2 = df[(df["image_sha256"] == f"{2:064x}") & (df["field"] == "unit")]
    assert unit2["source"].tolist() == ["officer_edit"] and unit2["value"].tolist() == ["99/124"]
    assert (df[df["image_sha256"] == f"{1:064x}"]["source"] == "officer_confirm").all()


def test_ingest_is_idempotent(roots, write_jsonl):
    images, labels, *_ = roots
    path = write_jsonl([review_event(1), review_event(2)])
    ingest_jsonl(path, images, labels)
    n = len(read_labels(labels))
    ingest_jsonl(path, images, labels)
    assert len(read_labels(labels)) == n


def test_splits_are_deterministic_and_grouped(split_params):
    df = pd.DataFrame({"group_key": [f"g{i % 50}" for i in range(200)]})
    a = assign_splits(df, {}, **split_params)
    b = assign_splits(df, {}, **split_params)
    assert (a == b).all()
    assert df.assign(split=a).groupby("group_key")["split"].nunique().max() == 1
    assert {"train", "val", "calibration"} >= set(a)


def test_frozen_sets_are_immutable_and_disjoint(tmp_path):
    write_frozen(tmp_path, Split.REGRESSION, ["g1", "g2"], "test")
    with pytest.raises(FileExistsError):
        write_frozen(tmp_path, Split.REGRESSION, ["g3"], "again")
    with pytest.raises(LeakageError):
        write_frozen(tmp_path, Split.ACCEPTANCE, ["g2", "g9"], "overlap")
    assert load_frozen(tmp_path)[Split.REGRESSION] == {"g1", "g2"}


def test_frozen_splits_cannot_be_exported_for_training():
    with pytest.raises(ForbiddenSplitError):
        guard_training_splits(["train", "acceptance"])


def test_snapshot_precedence_and_gold_only_frozen(roots, write_jsonl, split_params):
    images, labels, frozen, snap = roots
    # job 1: officer confirm, later an audit correction → audit wins
    e1 = review_event(1)
    fix = review_event(1, decided_by="audit", minutes=5)
    fix["final"] = {"unit": {"value": "99/999"}}
    # job 2 is in the regression set but has only officer labels → excluded (not gold)
    ingest_jsonl(write_jsonl([e1, fix, review_event(2)]), images, labels)
    write_frozen(frozen, Split.REGRESSION, ["P1:EF000000002TH"], "test")

    stats = build_snapshot(images, labels, frozen, snap, split_params)
    imgs, lbls = load_snapshot(snap)

    unit1 = lbls[(lbls["image_sha256"] == f"{1:064x}") & (lbls["field"] == "unit")]
    assert unit1["value"].tolist() == ["99/999"] and unit1["source"].tolist() == ["audit"]
    assert imgs.set_index("image_sha256").loc[f"{2:064x}", "split"] == "regression"
    assert lbls[lbls["image_sha256"] == f"{2:064x}"].empty
    assert stats["frozen_images_without_gold"] == 1


def test_retakes_share_a_split(roots, write_jsonl, split_params):
    images, labels, frozen, snap = roots
    # same tracking number, several jobs (retakes) → same group_key → same split
    ingest_jsonl(
        write_jsonl([review_event(i, tracking="EF999999999TH") for i in range(1, 6)]),
        images,
        labels,
    )
    build_snapshot(images, labels, frozen, snap, split_params)
    imgs, _ = load_snapshot(snap)
    assert imgs["group_key"].nunique() == 1 and imgs["split"].nunique() == 1
