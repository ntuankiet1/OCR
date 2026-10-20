import cv2
import numpy as np
import zxingcpp


def read_barcodes(img: np.ndarray) -> list[dict]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return [{"text": r.text, "format": str(r.format.name)} for r in zxingcpp.read_barcodes(gray)]
