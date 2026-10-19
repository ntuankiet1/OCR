from src.text.normalize import (
    normalize_common,
    normalize_for_eval,
    normalize_name_for_matching,
    normalize_tracking,
    normalize_unit,
)


def test_sara_am_is_unified_and_not_decomposed():
    # NIKHAHIT + SARA AA → SARA AM; an existing SARA AM must survive (NFKC would split it).
    assert normalize_common("นํา") == "นำ"
    assert normalize_common("นำ") == "นำ"


def test_tone_mark_order_is_canonical():
    # ก + mai ek + sara i  ==  ก + sara i + mai ek
    assert normalize_common("ก่ิ") == normalize_common("กิ่")


def test_thai_digits_and_fullwidth():
    assert normalize_common("๙๙/๑๒๓") == "99/123"
    assert normalize_common("ＡＢ１２") == "AB12"


def test_tracking_normalization():
    assert normalize_tracking(" ef 123-456-789 th ") == "EF123456789TH"


def test_unit_keeps_building_and_slash():
    assert normalize_unit("a / 1203") == "A/1203"


def test_honorifics_only_for_matching():
    assert normalize_name_for_matching("นายสมชาย ใจดี") == "สมชาย ใจดี"
    assert normalize_name_for_matching("Mr. John Smith") == "john smith"
    assert normalize_name_for_matching("Mrinal Sen") == "mrinal sen"  # no false strip
    # exact-match evaluation keeps the honorific: it is printed on the label
    assert normalize_for_eval("recipient_name", "นายสมชาย") == "นายสมชาย"


def test_missing_values():
    assert normalize_for_eval("unit", None) is None
    assert normalize_for_eval("unit", float("nan")) is None
