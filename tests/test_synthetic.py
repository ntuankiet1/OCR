import json
import random

import pytest

from src.data.ingest import ingest_jsonl, ingest_label_records
from src.data.labelstore import read_labels
from src.data.synthetic import find_thai_fonts, generate, s10_tracking
from src.schemas import ReviewEvent


def test_s10_check_digit():
    rng = random.Random(0)
    for _ in range(50):
        t = s10_tracking(rng)
        serial, check = t[2:10], int(t[10])
        c = 11 - sum(int(d) * w for d, w in zip(serial, (8, 6, 4, 2, 3, 5, 9, 7), strict=True)) % 11
        assert check == {10: 0, 11: 5}.get(c, c) and t.endswith("TH") and len(t) == 13


def _fonts_available() -> bool:
    try:
        return bool(find_thai_fonts())
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _fonts_available(), reason="no Thai font installed (fonts-thai-tlwg)")
def test_generate_feeds_the_ingest_contract(tmp_path):
    out, rosters = tmp_path / "demo", tmp_path / "rosters"
    summary = generate(out, rosters, n_jobs=12, regression_share=0.5, acceptance_share=0.2)
    assert summary["jobs"] == 12 and (rosters / "P1.json").exists()

    events = [ReviewEvent.model_validate_json(x) for x in (out / "review_events.jsonl").open()]
    assert len(events) == 12
    assert all((out / "images" / f"{e.image_sha256}.jpg").exists() for e in events)
    assert all(e.final["recipient_name"].polygon for e in events)  # text locations recorded

    images, labels = tmp_path / "images_index", tmp_path / "labels"
    ingest_jsonl(out / "review_events.jsonl", images, labels)
    res = ingest_label_records(out / "gold_labels.jsonl", images, labels)
    assert res["rejected"] == {"unknown_image": 0, "wrong_source": 0, "invalid": 0}
    assert "adjudicated" in set(read_labels(labels)["source"])

    keys = (out / "frozen_regression_keys.txt").read_text().split()
    assert len(keys) == summary["frozen_groups"]["regression"]
    assert json.loads((out / "summary.json").read_text())["batch"] == 1
