import numpy as np

from src.extract import LabelExtractor
from src.utils import unrotate


def lines(*texts, score=0.95):
    return [{"text": t, "score": score, "box": [[0, i], [1, i], [1, i + 1], [0, i + 1]]}
            for i, t in enumerate(texts)]


# What the OCR actually returned for the KEX sample label (samples/kex-label.png).
KEX = lines(
    "KEX", "ND", "KEX40001422961", "C", "EKC", "EKC001", "1of 1", "ผู้รับ: OMS PVS TEST",
    "โทร: 08****0720", "ที่อยู่: test บางบอนใด้ เขตบางบอน กทม.", "10150", "[10150]",
    "CW: 13KG", "B13*KS-R2M", "หมายเหตุ: Wt : 13.0 kg", "[BRK2] itbouatpos02 1st 2024-11-22 14:16",
)
KEX_BARCODES = [
    {"text": "KEX40001422961", "format": "Code128", "box": [[0, 0]] * 4},
    {"text": "KEX40001422961 1 1 EKC001", "format": "QRCode", "box": [[0, 0]] * 4},
]


def test_reads_the_kex_label():
    f = LabelExtractor().extract(KEX, KEX_BARCODES)
    assert f["CARRIER"].value == "KEX"
    assert f["TRACKING_TEXT"].value == "KEX40001422961"
    assert f["TRACKING_BARCODE"].value == "KEX40001422961"  # the 1D code, not the QR
    assert f["RECIPIENT"].value == "OMS PVS TEST"
    assert f["ADDRESS"].value == "test บางบอนใด้ เขตบางบอน กทม. 10150"  # stops before "[10150]"
    assert f["PARCEL_TYPE"] is None


def test_nothing_found_returns_none_not_a_guess():
    f = LabelExtractor().extract(lines("hello", "world"), [])
    assert all(v is None for v in f.values())


def test_two_carriers_on_one_label_are_not_chosen():
    f = LabelExtractor().extract(lines("DHL express", "Flash Express"), [])
    assert f["CARRIER"] is None


def test_recipient_on_the_next_line():
    f = LabelExtractor().extract(lines("ผู้รับ:", "สมชาย ใจดี"), [])
    assert f["RECIPIENT"].value == "สมชาย ใจดี"


def test_unrotate_is_the_inverse_of_a_counter_clockwise_turn():
    import cv2

    img = np.zeros((40, 60, 3), np.uint8)
    img[5, 10] = 255  # a marked pixel at x=10, y=5
    for angle, code in ((90, cv2.ROTATE_90_COUNTERCLOCKWISE), (180, cv2.ROTATE_180),
                        (270, cv2.ROTATE_90_CLOCKWISE)):
        turned = cv2.rotate(img, code)
        ys, xs = np.nonzero(turned[:, :, 0])
        x, y = unrotate(np.array([[xs[0] + 0.5, ys[0] + 0.5]]), angle, 60, 40)[0]
        assert abs(x - 10.5) < 1e-3 and abs(y - 5.5) < 1e-3, angle


def boxed(rows):
    """(x0, y0, x1, y1, text, score) -> OCR lines with rectangular boxes."""
    return [
        {"text": t, "score": sc, "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}
        for x0, y0, x1, y1, t, sc in rows
    ]


# The DHL Paket sample (samples/dhl-label.png): a handwritten form, two columns, German captions.
DHL = boxed([
    (148, 186, 912, 281, "DHL PAKET UND PäCKCHEN DEUTSCHLAND + EU", 0.95),
    (148, 248, 385, 293, "Absender:in /Sender", 0.98),
    (985, 261, 1206, 332, "DHL", 0.51),
    (132, 286, 576, 433, "Hadlna (da", 0.62),
    (722, 287, 949, 339, "Empfanger:in Aadress", 0.74),
    (724, 320, 1195, 396, "Dpofbg.556597", 0.72),
    (714, 380, 1186, 463, "Shop sdeaItda", 0.72),
    (136, 401, 529, 485, "Sn dorNee d1", 0.65),
    (716, 428, 1234, 486, "Adresszusatz/ostnr. fur PadTel (our ber EU-Versand oder Sper", 0.85),
    (1272, 364, 1337, 444, "Jeutsc", 0.91),
    (723, 491, 1082, 537, "NBe nd Hausnr. (keiPost/DHL Packstation", 0.77),
    (713, 501, 1041, 588, "Nniadbt.D5", 0.53),
    (716, 574, 1142, 657, "0041Nimbore", 0.59),
    (709, 627, 822, 651, "Land /Destieation", 0.76),
    (713, 675, 753, 716, "7", 0.99),
    (773, 679, 1046, 720, "Inklusive nachhaltigem Versand.", 0.95),
])


def test_dhl_recipient_column_skips_captions_and_the_other_column():
    f = LabelExtractor().extract(DHL, [])
    assert f["RECIPIENT"].value == "Dpofbg.556597"
    assert f["RECIPIENT"].lines == [5]
    assert f["ADDRESS"].value == "Shop sdeaItda Nniadbt.D5 0041Nimbore"  # not captions, not "7"
    # the weakest handwritten line (0.53), lowered because a rule picked it
    assert f["ADDRESS"].confidence == round(0.53 * 0.8, 4)


def test_dhl_carrier_box_is_the_logo_and_confidence_the_best_evidence():
    carrier = LabelExtractor().extract(DHL, [])["CARRIER"]
    assert carrier.value == "DHL"
    assert carrier.lines == [2]  # the line that is only "DHL"
    assert carrier.confidence == round(0.95 * 0.9, 4)  # the title line names it too (0.95)


# --- the popular brands: real OCR output of one sample label each (tests/fixtures/*.json) -------
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402


def fixture(name):
    data = json.loads((Path(__file__).parent / "fixtures" / f"{name}.json").read_text())
    return data["lines"], data["barcodes"]


def values(name):
    found = LabelExtractor().extract(*fixture(name))
    return {k: (v.value if v else None) for k, v in found.items()}


BRANDS = {
    "flash": {
        "CARRIER": "Flash Express",
        "TRACKING_TEXT": "THF1234567890",
        "TRACKING_BARCODE": None,  # the sample's barcode is not decodable
        "RECIPIENT": "Somchai Srisuk",
        "ADDRESS": "789/10 Rama 9 Rd. Huai Khwang, Bangkok 10310",
    },
    "jt": {
        "CARRIER": "J&T Express",
        "TRACKING_TEXT": "611485225",  # digits only: taken because it is the one such line
        "TRACKING_BARCODE": None,
        "RECIPIENT": "สมชาย ใจดี",  # "คุณ" removed
        "ADDRESS": "789/10 หมู่ 5 ตำบลหนองปรือ อำเภอบางละมุง จังหวัดชลบุรี 20150",
    },
    "ninjavan": {
        "CARRIER": "Ninja Van",
        "TRACKING_TEXT": "SPE3984522429",
        "TRACKING_BARCODE": None,
        "RECIPIENT": "Noor aishah harpan",  # the recipient, not the sender "Noramiza bt zulkifli"
        "ADDRESS": "No. 7, jalan cempenai, taman nira, Batu Pahat, Johor",
    },
    "spx": {
        "CARRIER": "SPX",
        "TRACKING_TEXT": "TH259433395917X",
        "TRACKING_BARCODE": "TH259433395917X",  # the QR code, as there is no 1D barcode
        "RECIPIENT": "สุภาวดี แสงทอง",
        "ADDRESS": "456/78 หมู 4 ตำบลบางจาก อำเภอพระประแดง จังหวัดสมุทรปราการ 10130",
    },
    "shopeefood": {
        "CARRIER": "Shopee Food",
        "TRACKING_TEXT": "250826SWW8QG9E",  # its order number: the only number on the label
        "TRACKING_BARCODE": None,
        "RECIPIENT": "ณัฐวุฒิ ใจดี",  # not the zone code "Z1" on the "ผู้รับ (TO)" line
        "ADDRESS": "123/45 หมู่ 6 ตำบลบางพลี อำเภอบางพลี จังหวัดสมุทรปราการ 10540",
    },
}


@pytest.mark.parametrize("brand", BRANDS)
def test_reads_the_brand_label(brand):
    got = values(brand)
    assert {k: got[k] for k in BRANDS[brand]} == BRANDS[brand]
    assert got["PARCEL_TYPE"] is None


def test_the_sender_is_never_read_as_the_recipient():
    for brand in BRANDS:
        sender_words = ("Mr.partner", "Kittipong", "ABC Online", "Noramiza", "shopeefood.bangkok")
        got = values(brand)
        assert not any(w in (got["RECIPIENT"] or "") + (got["ADDRESS"] or "") for w in sender_words)


def test_thai_anchors_survive_dropped_tone_marks():
    # At A6 size the OCR read "ที่อยู่:" as "ที่อยู:" and "ผู้รับ:" as "ผูรับ:".
    f = LabelExtractor().extract(
        lines("ผูรับ: OMS PVS TEST", "ที่อยู: test บางบอน กทม.", "10150", "[10150]"), []
    )
    assert f["RECIPIENT"].value == "OMS PVS TEST"
    assert f["ADDRESS"].value == "test บางบอน กทม. 10150"


def test_a_doubled_thai_mark_is_squashed():
    from src.extract import squash_marks

    assert squash_marks("สมชาย ใจดีี") == "สมชาย ใจดี"
    assert squash_marks("ผู้รับ") == "ผู้รับ"  # a vowel followed by a different mark is untouched


def test_a_photographed_sticky_label_read_upside_down_and_tilted():
    """tests/fixtures/spx_photo.json: the printed SPX label stuck on a wine box, photographed."""
    got = values("spx_photo")
    assert got["CARRIER"] == "SPX"
    assert got["TRACKING_TEXT"] == "TH259433395917X"
    assert got["RECIPIENT"] == "สุภาวดี แสงทอง"  # despite the stray "ฝ" before "ผู้รับ"
    assert "456/78" in got["ADDRESS"] and "10130" in got["ADDRESS"]
    # none of the box's own printed text ("counting down 1926", "RALIA") leaks into the address
    assert "1926" not in got["ADDRESS"] and "counting" not in got["ADDRESS"]
    # ...and it stops before the order number and the tracking number below it
    assert "Order" not in got["ADDRESS"] and "TH2594" not in got["ADDRESS"]


def test_a_photographed_upside_down_jt_label_with_a_misread_recipient_word():
    """tests/fixtures/jt_photo.json: "ผู้รับ (TO)" was read as "ูรับ (TO)"."""
    got = values("jt_photo")
    assert got["CARRIER"] == "J&T Express"
    assert got["RECIPIENT"].endswith("ใจดี") and "คุณ" not in got["RECIPIENT"]
    assert "789/10" in got["ADDRESS"] and "20150" in got["ADDRESS"]
    assert "Order" not in got["ADDRESS"] and "Mrparther" not in got["ADDRESS"]  # not the sender


def test_a_photographed_shopee_food_label():
    got = values("shopeefood_photo")
    assert got["CARRIER"] == "Shopee Food"
    assert got["RECIPIENT"] == "ณัฐวุฒิ ใจดี"
    assert "123/45" in got["ADDRESS"]


def test_digits_and_thai_words_are_separated_in_an_address():
    from src.extract import space_digits_and_thai

    assert space_digits_and_thai("123/45หมู่6 ตำบลบางพลี") == "123/45 หมู่ 6 ตำบลบางพลี"
    assert space_digits_and_thai("10540") == "10540"


def test_rule_based_fields_are_less_certain_than_the_ocr_line_score():
    ex = LabelExtractor()
    f = ex.extract(*fixture("flash"))
    for name in ("CARRIER", "TRACKING_TEXT", "RECIPIENT", "ADDRESS"):
        cap = ex.confidence[name]["max"]
        assert 0 < f[name].confidence <= cap < 0.95, name  # always under Parcel's 0.95 threshold
    f = ex.extract(*fixture("spx"))
    assert f["TRACKING_BARCODE"].confidence == 0.99  # a decoded barcode is exact: not lowered


def test_jt_digits_are_not_a_tracking_number_when_two_lines_could_be_one():
    lines_ = lines("J&T Express", "611485225", "0812345678")
    assert LabelExtractor().extract(lines_, [])["TRACKING_TEXT"] is None


def test_the_order_number_is_a_tracking_number_only_for_shopee_food():
    spx = lines("SPX", "Shopee Order No. 250826SWW8QG9E")
    assert LabelExtractor().extract(spx, [])["TRACKING_TEXT"] is None
    food = lines("ShopeeFood", "Shopee Order No. 250826SWW8QG9E")
    assert LabelExtractor().extract(food, [])["TRACKING_TEXT"].value == "250826SWW8QG9E"


def test_both_photos_read_the_tracking_number_too():
    assert values("jt_photo")["TRACKING_TEXT"] == "611485225"
    assert values("shopeefood_photo")["TRACKING_TEXT"] is not None


def test_a_row_the_detector_cut_into_word_pieces_is_put_back_together():
    """tests/fixtures/shopeefood_iphone.json: an iPhone photo. "ผู้รับ" "(TO)" "Z1" came as three
    lines, the name as two ("คุณ" "ณัฐวุฒิ ใจดี"), and instruction lines sit in the next column."""
    got = values("shopeefood_iphone")
    assert got["CARRIER"] == "Shopee Food"
    assert got["TRACKING_TEXT"] == "250826SWW8QG9E"
    assert got["RECIPIENT"] == "ณัฐวุฒิ ใจดี"
    assert got["ADDRESS"] == "123/45 หมู่ 6 ตำบลบางพลี อำเภอบางพลี จังหวัดสมุทรปราการ 10540"


def test_pieces_of_one_row_are_merged_and_columns_are_not():
    ex = LabelExtractor()
    rows = ex.merge_rows(boxed([
        (10, 10, 60, 40, "ผู้รับ", 0.99),
        (65, 10, 110, 40, "(TO)", 0.97),
        (115, 10, 150, 40, "Z1", 0.76),
        (400, 10, 520, 40, "COD (บาท)", 0.95),  # another column: a wide gutter away
        (10, 60, 200, 90, "123/45 หมู่ 6", 0.98),  # the next row: stacked, not beside
    ]))
    assert [r["text"] for r in rows] == ["ผู้รับ (TO) Z1", "COD (บาท)", "123/45 หมู่ 6"]
    assert rows[0]["parts"] == [0, 1, 2] and rows[0]["score"] == 0.76  # the weakest piece


def test_a_field_read_from_a_merged_row_points_at_its_original_pieces():
    lines_ = boxed([
        (10, 10, 60, 40, "ผู้รับ", 0.99),
        (65, 10, 110, 40, "(TO)", 0.97),
        (115, 10, 250, 40, "คุณสมชาย ใจดี", 0.95),
        (10, 60, 200, 90, "123/45 หมู่ 6", 0.98),
    ])
    found = LabelExtractor().extract(lines_, [])
    assert found["RECIPIENT"].value == "สมชาย ใจดี"
    assert found["RECIPIENT"].lines == [0, 1, 2]  # the three original lines, for the box
    assert found["ADDRESS"].lines == [3]
