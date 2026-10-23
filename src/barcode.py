import cv2
import numpy as np
import zxingcpp

from .utils import unrotate

_TURN = {
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,
}


class BarcodeReader:
    """Reads every 1D/2D barcode in an image (zxing-cpp), and can also point at barcodes it
    could not read (OpenCV's detector finds them without decoding)."""

    def __init__(self):
        try:
            self._locator = cv2.barcode.BarcodeDetector()
        except AttributeError:  # an OpenCV build without the barcode module
            self._locator = None

    def read(self, img: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found = []
        for r in zxingcpp.read_barcodes(gray):
            p = r.position
            corners = [p.top_left, p.top_right, p.bottom_right, p.bottom_left]
            found.append(
                {
                    "text": r.text,
                    "format": r.format.name,
                    "box": [[pt.x, pt.y] for pt in corners],  # in `img` coordinates
                }
            )
        return found

    def find_unread(
        self, img: np.ndarray, read: list[dict], angle: int = 0
    ) -> list[list[list[float]]]:
        """Corners (in `img` coordinates) of barcode-looking regions that `read` does not cover:
        the code is there but damaged, blurred, tilted too far or not a valid code. Only the
        place is used: OpenCV's own decoding is not trusted (it misread a real EAN-13 as an
        EAN-8). `angle` is the rotation that makes the label upright (see TextReading.angle):
        the locator finds barcodes much more reliably on an upright image."""
        if self._locator is None:
            return []
        h, w = img.shape[:2]
        upright = cv2.rotate(img, _TURN[angle]) if angle in _TURN else img
        try:
            ok, regions = self._locator.detectMulti(upright)
        except cv2.error:
            return []
        if not ok or regions is None:
            return []
        decoded = [np.array(b["box"], np.float32) for b in read]
        unread = []
        for region in regions:
            region = unrotate(region, angle, w, h)  # back into `img` coordinates
            x, y = (float(v) for v in region.mean(axis=0))
            covered = any(
                cv2.pointPolygonTest(box.reshape(-1, 1, 2), (x, y), False) >= 0 for box in decoded
            )
            if not covered:
                unread.append(region.astype(float).tolist())
        return unread
