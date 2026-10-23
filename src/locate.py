"""Find the shipping label in a photo and flatten it.

With a trained detector the photo is first cropped to the detected box. Inside that crop (or the
whole photo when there is no detector) the largest bright four-corner shape is straightened; if
none is found the crop is used as is.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import ShippingLabelDetector


@dataclass
class LocatedLabel:
    image: np.ndarray  # the straightened (or just cropped) label
    origin: tuple[int, int]  # top-left of the crop in the photo
    quad: list[list[float]] | None  # corners in the photo, None when no corner shape was found
    label_count: int  # how many labels the detector saw (1 when there is no detector)
    inverse: np.ndarray | None = None  # 3x3 matrix: straightened label -> crop
    detected: bool = False  # True when the detector found the label or the image is itself one

    def to_photo(self, points: np.ndarray) -> np.ndarray:
        """Map (N, 2) points from `image` coordinates to photo coordinates."""
        pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
        if self.inverse is not None:
            pts = cv2.perspectiveTransform(pts, self.inverse)
        return pts.reshape(-1, 2) + np.array(self.origin, np.float32)


class LabelLocator:
    def __init__(
        self,
        detector: ShippingLabelDetector | None = None,
        pad: float = 0.03,
        min_area: float = 0.15,
        flat_labels: bool = True,
    ):
        self.detector = detector
        self.pad = pad  # fraction of the box added around it before cropping
        self.min_area = min_area  # smallest label, as a fraction of the (cropped) image
        self.flat_labels = flat_labels  # an image that is itself the label is not cropped

    def locate(self, img: np.ndarray) -> LocatedLabel:
        """Crop to the detected label, then straighten it when its four corners are found."""
        return self.straighten(*self.crop_to_detection(img))

    def straighten(
        self, crop: np.ndarray, origin: tuple[int, int], label_count: int, detected: bool = False
    ) -> LocatedLabel:
        quad = self.find_quad(crop)
        if quad is None:
            return LocatedLabel(crop, origin, None, label_count, detected=detected)
        return self.warp_to(crop, quad, label_count, origin, detected)

    def warp_to(
        self,
        img: np.ndarray,
        quad: np.ndarray,
        label_count: int = 1,
        origin: tuple[int, int] = (0, 0),
        detected: bool = True,
    ) -> LocatedLabel:
        """Flatten `quad` (corners in `img`)."""
        matrix, size = self.warp_matrix(quad)
        flat = cv2.warpPerspective(img, matrix, size)
        corners = (quad + np.array(origin, np.float32)).tolist()
        return LocatedLabel(flat, origin, corners, label_count, np.linalg.inv(matrix), detected)

    def crop_to_detection(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int], int, bool]:
        """Crop to the best detected box: (crop, (x, y) of its top-left in `img`, labels seen,
        whether the label was actually found: by the detector, or because the image is one)."""
        if self.flat_labels and self.is_flat_label(img):
            return img, (0, 0), 1, True
        found = self.detector.detect(img) if self.detector else []
        if not found:
            return img, (0, 0), 1, False
        (bx0, by0, bx1, by1), _ = found[0]
        px, py = self.pad * (bx1 - bx0), self.pad * (by1 - by0)
        x0, y0 = int(max(bx0 - px, 0)), int(max(by0 - py, 0))
        x1, y1 = int(min(bx1 + px, img.shape[1])), int(min(by1 + py, img.shape[0]))
        return img[y0:y1, x0:x1], (x0, y0), len(found), True

    @staticmethod
    def is_flat_label(img: np.ndarray, min_white: float = 0.5, white: int = 215) -> bool:
        """True when the image itself is a label (a scan, PDF or screenshot) and not a photo of
        one: most of it is white paper (flat labels measured 68-85%, photos of parcels near 0%).
        The detector is trained on photos and would crop such a label short, losing its header
        and side columns."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float((gray > white).mean()) >= min_white

    def find_quad(self, img: np.ndarray) -> np.ndarray | None:
        h, w = img.shape[:2]
        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(c)
            if area < self.min_area * w * h:
                break
            if area > 0.95 * w * h:  # the frame itself, not a label
                continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return self.order_quad(approx.reshape(4, 2).astype(np.float32))
        return None

    @staticmethod
    def order_quad(pts: np.ndarray) -> np.ndarray:
        """Corners as top-left, top-right, bottom-right, bottom-left."""
        s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
        return np.array(
            [pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32
        )

    @staticmethod
    def warp_matrix(quad: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Perspective matrix that straightens `quad`, and the output size (width, height)."""
        tl, tr, br, bl = quad
        width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32
        )
        return cv2.getPerspectiveTransform(quad, dst), (width, height)
