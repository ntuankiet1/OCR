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


def default_device() -> str:
    """GPU when this paddle build has CUDA and a GPU is visible. The CPU wheel (`paddlepaddle`)
    never uses the GPU, even on a GPU machine: install `paddlepaddle-gpu` for that."""
    import paddle

    if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        return "gpu:0"
    return "cpu"


def default_enable_mkldnn() -> bool:
    """oneDNN speeds up CPU inference, but PaddlePaddle 3.3.x fails with
    "ConvertPirAttribute2RuntimeAttribute not support" (Paddle issue #77340); 3.2.2 is fine."""
    import paddle

    return not paddle.__version__.startswith("3.3")


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
        enable_mkldnn: bool | None = None,  # None → default_enable_mkldnn() (CPU only)
        device: str | None = None,  # None → default_device(); e.g. "gpu:0", "cpu"
    ) -> None:
        from paddleocr import PaddleOCR, TextRecognition

        self.device = device or default_device()
        on_cpu = self.device.startswith("cpu")
        mkldnn = on_cpu and (default_enable_mkldnn() if enable_mkldnn is None else enable_mkldnn)
        common = {"device": self.device, "enable_mkldnn": mkldnn}
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
            **common,
        )
        rec_kw = {"model_name": rec_name, **common}
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
