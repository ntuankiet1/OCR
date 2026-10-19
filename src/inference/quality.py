"""Image quality gate (BR02: detect blur, glare). Heuristics first; a learned model can replace
this later, trained on officer "retake" events."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityReport:
    blur_score: float  # variance of Laplacian; low = blurry
    glare_ratio: float  # share of near-saturated pixels
    reasons: list[str]


def assess(image_bgr: np.ndarray, blur_min: float, glare_max: float) -> QualityReport:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Normalise resolution so the blur score is comparable across cameras.
    scale = 1024 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare = float((gray >= 250).mean())
    reasons = []
    if blur < blur_min:
        reasons.append("image_blurry")
    if glare > glare_max:
        reasons.append("image_glare")
    return QualityReport(blur_score=blur, glare_ratio=glare, reasons=reasons)
