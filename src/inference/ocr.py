"""OCR engine adapter for PaddleOCR 3.x.

Production loads every model from the bundle directory, so nothing is downloaded at runtime
(BRD §9). Benchmarks may pass `models_dir=None` to let PaddleOCR fetch models by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class OcrLine:
    text: str
    score: float
    polygon: list[tuple[float, float]]


class OcrEngine(Protocol):
    def read(self, image_bgr: np.ndarray) -> list[OcrLine]: ...


def _model_kwargs(role: str, prefix: str, name: str, models_dir: Path | None) -> dict:
    kw = {f"{prefix}_model_name": name}
    if models_dir is not None:
        kw[f"{prefix}_model_dir"] = str(models_dir / role)
    return kw


class PaddleOcrEngine:
    def __init__(
        self,
        models_dir: Path | None,
        det_name: str,
        rec_name: str,
        doc_ori_name: str,
        textline_ori_name: str,
    ) -> None:
        from paddleocr import PaddleOCR, TextRecognition

        self._ocr = PaddleOCR(
            **_model_kwargs("det", "text_detection", det_name, models_dir),
            **_model_kwargs("rec", "text_recognition", rec_name, models_dir),
            **_model_kwargs(
                "doc_orientation", "doc_orientation_classify", doc_ori_name, models_dir
            ),
            **_model_kwargs(
                "textline_orientation", "textline_orientation", textline_ori_name, models_dir
            ),
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
        rec_kw = {"model_name": rec_name}
        if models_dir is not None:
            rec_kw["model_dir"] = str(models_dir / "rec")
        self._rec = TextRecognition(**rec_kw)

    def read(self, image_bgr: np.ndarray) -> list[OcrLine]:
        lines: list[OcrLine] = []
        for res in self._ocr.predict(image_bgr):
            for text, score, poly in zip(
                res["rec_texts"], res["rec_scores"], res["rec_polys"], strict=True
            ):
                lines.append(
                    OcrLine(
                        text=text, score=float(score), polygon=[tuple(map(float, p)) for p in poly]
                    )
                )
        return lines

    def recognize(self, line_bgr: np.ndarray) -> tuple[str, float]:
        """Recognizer only, no detection: for an image that is already a single text line."""
        for res in self._rec.predict(line_bgr):
            return str(res["rec_text"]), float(res["rec_score"])
        return "", 0.0
