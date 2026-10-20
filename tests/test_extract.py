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
    assert f["ADDRESS"].confidence == 0.53  # the weakest handwritten line


def test_dhl_carrier_box_is_the_logo_and_confidence_the_best_evidence():
    carrier = LabelExtractor().extract(DHL, [])["CARRIER"]
    assert carrier.value == "DHL"
    assert carrier.lines == [2]  # the line that is only "DHL"
    assert carrier.confidence == 0.95  # the title line names it too
