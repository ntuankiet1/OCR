"""Find the waybill in a photo and flatten it.

Today: the largest bright four-corner shape (a white label on a parcel). If none is found the whole
image is used. A trained detector (scripts/train_waybill_detector.py) plugs in by returning its box
from `find_quad`.
"""

import cv2
import numpy as np


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]],
                    dtype=np.float32)


def find_quad(img: np.ndarray, min_area: float = 0.15) -> np.ndarray | None:
    h, w = img.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        area = cv2.contourArea(c)
        if area < min_area * w * h:
            break
        if area > 0.95 * w * h:  # the frame itself, not a label
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_quad(approx.reshape(4, 2).astype(np.float32))
    return None


def warp(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, dst), (width, height))


def locate(img: np.ndarray) -> tuple[np.ndarray, list[list[float]] | None]:
    """Return (flattened waybill or the original image, corner list or None)."""
    quad = find_quad(img)
    if quad is None:
        return img, None
    return warp(img, quad), quad.tolist()
