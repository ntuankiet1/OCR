"""Parcel type/size classifier (ONNX) with temperature scaling and abstention.
BRD §7: if classification is not possible, leave the field blank — never guess or default."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class TypeSizeClassifier:
    def __init__(self, onnx_path: Path, image_size: int, temperature: float = 1.0) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.image_size = image_size
        self.temperature = temperature

    def predict(self, image_bgr: np.ndarray) -> tuple[int, float]:
        """Return (class id 1-10, calibrated max probability)."""
        rgb = cv2.cvtColor(
            cv2.resize(image_bgr, (self.image_size, self.image_size)), cv2.COLOR_BGR2RGB
        )
        x = ((rgb.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
        logits = self.session.run(None, {self.input_name: x})[0][0]
        probs = softmax(logits / self.temperature)
        idx = int(probs.argmax())
        return idx + 1, float(probs[idx])
