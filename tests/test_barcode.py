import cv2
import numpy as np
import zxingcpp

from src.barcode import BarcodeReader


def barcode_photo(text="KEX40001422961"):
    bars = zxingcpp.write_barcode(zxingcpp.BarcodeFormat.Code128, text, width=420, height=140)
    gray = np.array(bars)
    page = np.full((300, 600), 255, np.uint8)
    page[80:80 + gray.shape[0], 90:90 + gray.shape[1]] = gray
    return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


def test_reads_a_code128_and_reports_where_it_is():
    reader = BarcodeReader()
    found = reader.read(barcode_photo())
    assert [b["text"] for b in found] == ["KEX40001422961"]
    assert len(found[0]["box"]) == 4


def test_a_barcode_that_was_read_is_not_reported_as_unread():
    reader = BarcodeReader()
    img = barcode_photo()
    assert reader.find_unread(img, reader.read(img)) == []


def test_without_the_opencv_barcode_module_nothing_is_reported():
    reader = BarcodeReader()
    reader._locator = None
    assert reader.find_unread(barcode_photo(), []) == []


def test_unread_regions_are_mapped_back_when_the_label_is_upside_down():
    reader = BarcodeReader()
    upright = barcode_photo()
    turned = cv2.rotate(upright, cv2.ROTATE_180)
    # a locator stand-in that answers on the upright image: one region at a known place
    seen = {}

    class Fake:
        def detectMulti(self, img):
            seen["shape"] = img.shape
            return True, np.array([[[10, 20], [110, 20], [110, 60], [10, 60]]], np.float32)

    reader._locator = Fake()
    (region,) = reader.find_unread(turned, [], angle=180)
    h, w = turned.shape[:2]
    # (10, 20) in the upright image is (w - 10, h - 20) in the turned one
    assert any(abs(x - (w - 10)) < 1 and abs(y - (h - 20)) < 1 for x, y in region)
