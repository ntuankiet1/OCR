import cv2
import numpy as np
import zxingcpp


class BarcodeReader:
    """Reads every 1D/2D barcode in an image (zxing-cpp)."""

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
