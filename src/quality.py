import cv2
import numpy as np

from .utils import resize_max_side


class QualityChecker:
    """Image-quality flags in Parcel's vocabulary. Only BLUR is measured today.

    GLARE and INCOMPLETE_LABEL are not detected yet: a white label looks like glare to a simple
    brightness test, and a label can legitimately fill the whole photo.
    """

    def __init__(self, blur_threshold: float):
        self.blur_threshold = blur_threshold

    def blur_score(self, img: np.ndarray) -> float:
        """Variance of the Laplacian on a fixed-size grayscale copy (higher = sharper)."""
        gray = cv2.cvtColor(resize_max_side(img, 800), cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def flags(self, img: np.ndarray) -> list[str]:
        return ["BLUR"] if self.blur_score(img) < self.blur_threshold else []
