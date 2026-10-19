from pathlib import Path

import yaml

from src.inference.barcode import Barcode
from src.inference.decision import DecisionInput, decide
from src.inference.kie_rules import RuleExtractor
from src.inference.matching import Resident, Unit, match_recipient, match_unit
from src.inference.ocr import OcrLine

CARRIERS = yaml.safe_load((Path(__file__).parents[1] / "configs" / "carriers.yaml").read_text())
BOX = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)]


def line(text: str, score: float = 0.98) -> OcrLine:
    return OcrLine(text, score, BOX)


# ---------------------------------------------------------------- decision (BRD §7)
def ok_input(**kw) -> DecisionInput:
    base = dict(
        auto_mode_enabled=True,
        values={
            "recipient_name": "a",
            "unit": "99/1",
            "carrier": "thailand_post",
            "tracking_number": "x",
            "type_size": "7",
        },
        scores=dict.fromkeys(
            ["recipient_name", "unit", "carrier", "tracking_number", "type_size"], 0.99
        ),
        thresholds=dict.fromkeys(
            ["recipient_name", "unit", "carrier", "tracking_number", "type_size"], 0.9
        ),
        unit_status="matched",
        recipient_status="matched",
        recipient_has_account=True,
    )
    return DecisionInput(**{**base, **kw})


def test_all_conditions_pass():
    assert decide(ok_input()).auto


def test_every_failing_condition_is_reported():
    d = decide(
        ok_input(
            auto_mode_enabled=False,
            values={**ok_input().values, "type_size": None},
            unit_status="ambiguous",
            conflict_reasons=["conflict_barcode_text_mismatch"],
            duplicate_check_passed=False,
        )
    )
    assert not d.auto
    assert {
        "auto_mode_disabled",
        "missing_type_size",
        "unit_ambiguous",
        "conflict_barcode_text_mismatch",
        "duplicate",
    } <= set(d.reasons)


def test_uncalibrated_field_never_auto():
    d = decide(ok_input(thresholds={**ok_input().thresholds, "unit": None}))
    assert "low_confidence_unit" in d.reasons


def test_recipient_without_account_goes_to_review():
    assert "recipient_no_account" in decide(ok_input(recipient_has_account=False)).reasons


# ---------------------------------------------------------------- matching (BR03/BR04)
UNITS = [
    Unit(
        "U1",
        ["A/1203"],
        [Resident("R1", ["สมชาย ใจดี", "Somchai Jaidee"]), Resident("R2", ["สมหญิง ใจดี"])],
    ),
    Unit("U2", ["A/1204"], [Resident("R3", ["John Smith"])]),
]


def test_unit_requires_exact_code():
    assert match_unit("a / 1203", UNITS, 90, 8).best_id == "U1"
    assert match_unit("A/1205", UNITS, 90, 8).status == "no_match"


def test_recipient_match_and_ambiguity():
    assert match_recipient("นายสมชาย ใจดี", UNITS[0], 90, 8).best_id == "R1"
    assert match_recipient("Somchai Jaidee", UNITS[0], 90, 8).best_id == "R1"
    twins = Unit("U3", ["1"], [Resident("R4", ["Anan Sukjai"]), Resident("R5", ["Anan Sukjai"])])
    assert match_recipient("Anan Sukjai", twins, 90, 8).status == "ambiguous"


# ---------------------------------------------------------------- rule KIE (BR07)
def test_barcode_and_text_agree():
    ex = RuleExtractor(CARRIERS).extract(
        [line("ไปรษณีย์ไทย"), line("EF 123456789 TH"), line("ผู้รับ: สมชาย ใจดี"), line("ห้อง A/1203")],
        [Barcode("EF123456789TH", "Code128", BOX)],
    )
    f = ex.fields
    assert (
        f["tracking_number"].value == "EF123456789TH" and f["tracking_number"].source == "barcode"
    )
    assert f["carrier"].value == "thailand_post"
    assert f["recipient_name"].value == "สมชาย ใจดี"
    assert f["unit"].value == "A/1203"
    assert ex.reasons == []


def test_barcode_text_disagreement_is_a_conflict():
    ex = RuleExtractor(CARRIERS).extract(
        [line("EF123456780TH")], [Barcode("EF123456789TH", "Code128", BOX)]
    )
    assert ex.fields["tracking_number"].value is None
    assert "conflict_barcode_text_mismatch" in ex.reasons


def test_sender_block_unit_is_ignored():
    ex = RuleExtractor(CARRIERS).extract(
        [line("ผู้รับ สมชาย"), line("99/123"), line("ผู้ส่ง ร้านค้า"), line("55/8")], []
    )
    assert ex.fields["unit"].value == "99/123"
