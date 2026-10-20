"""Small helpers shared by several modules."""

import cv2
import numpy as np
import onnxruntime as ort


def pick_providers(requested: str | None, device: str) -> list[str]:
    """ONNX Runtime execution providers to try, best first; CPU is always the last resort.

    Auto: CUDA for an NVIDIA device (OCR_DEVICE=gpu:N), CoreML when running natively on macOS.
    `requested` ("A,B") overrides that. Providers this onnxruntime build lacks are skipped.
    """
    if requested:
        wanted = [p.strip() for p in requested.split(",") if p.strip()]
    elif device.startswith("gpu"):
        wanted = ["CUDAExecutionProvider"]
    else:
        wanted = ["CoreMLExecutionProvider"]
    available = ort.get_available_providers()
    return [p for p in wanted if p in available and p != "CPUExecutionProvider"] + [
        "CPUExecutionProvider"
    ]


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink `img` so its long side is at most `max_side` (never enlarges)."""
    scale = max_side / max(img.shape[:2])
    if scale >= 1:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def unrotate(points: np.ndarray, angle: int, width: int, height: int) -> np.ndarray:
    """Map (N, 2) points from an image turned counter-clockwise by `angle` degrees back to the
    original `width` x `height` image. This is how PaddleOCR's document-orientation step turns
    a sideways label (angle 90 = counter-clockwise, checked against its output image)."""
    x, y = np.asarray(points, np.float32).reshape(-1, 2).T
    if angle == 90:
        out = np.stack([width - y, x])
    elif angle == 180:
        out = np.stack([width - x, height - y])
    elif angle == 270:
        out = np.stack([y, height - x])
    else:
        out = np.stack([x, y])
    return out.T
