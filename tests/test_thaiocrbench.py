import numpy as np
import pytest

from src.benchmarks.thaiocrbench import crop_normalized, parse_bbox, reading_order
from src.benchmarks.thaiocrbench_metrics import (
    bmfl,
    official_score,
    our_metrics,
    text_recognition_score,
)
from src.inference.ocr import OcrLine


# ---------------------------------------------------------------- official metrics
def test_text_recognition_containment_ignores_case_and_spaces():
    assert text_recognition_score("ป้าย: ร้าน ข้าวมันไก่\nเปิด 7 โมง", "ร้านข้าวมันไก่") == 1.0
    assert text_recognition_score("HELLO world", "hello  world") == 1.0


def test_text_recognition_anls_floor():
    assert text_recognition_score("abcdefghij", "abcdefghik") == pytest.approx(0.9)
    assert text_recognition_score("zzzz", "abcdefgh") == 0.0  # similarity < 0.5 → 0


def test_text_recognition_list_answer():
    assert text_recognition_score("ราคา 50 บาท", "['50 บาท', '฿50']") == 1.0


def test_bmfl_identical_thai_scores_high_and_empty_scores_zero():
    gt = "สำนักมาตรฐานห้องปฏิบัติการ กระทรวงสาธารณสุข ประเทศไทย"
    s = bmfl(gt, gt)
    assert s["nls"] == 1.0 and s["f_measure"] == 1.0 and s["bmfl"] > 0.95
    assert bmfl("", gt)["bmfl"] == 0.0


def test_official_score_rejects_vqa_tasks():
    with pytest.raises(ValueError):
        official_score("Cognition VQA", "x", "y")


def test_our_metrics_use_project_normalization():
    m = our_metrics("ปี ๒๕๖๘", "ปี2568")
    assert m == {"cer": 0.0, "exact": 1.0}
    assert our_metrics("", "abc")["cer"] == 1.0


# ---------------------------------------------------------------- geometry
def test_parse_bbox():
    q = (
        "แบ่งความยาวและความสูงของรูปภาพออกเป็น 1000 ส่วน "
        "แล้วช่วยดึงข้อความที่อยู่ในพิกัด [274, 385, 573, 501] ของรูปภาพ"
    )
    assert parse_bbox(q) == (274.0, 385.0, 573.0, 501.0)
    assert parse_bbox("no box here") is None


def test_crop_normalized_maps_grid_to_pixels():
    img = np.zeros((200, 1000, 3), np.uint8)
    crop = crop_normalized(img, (100, 500, 300, 1000), pad=0.0)
    assert crop.shape[:2] == (100, 200)


def test_reading_order_rows_then_columns():
    def ln(text, x, y):
        return OcrLine(text, 0.9, [(x, y), (x + 50, y), (x + 50, y + 20), (x, y + 20)])

    lines = [ln("world", 100, 2), ln("second", 0, 60), ln("hello", 0, 0)]
    assert reading_order(lines) == "hello world\nsecond"
    assert reading_order([]) == ""
