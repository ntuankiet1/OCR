"""photo → find waybill → straighten → orientation → OCR (+ barcodes)."""

import time

import cv2
import numpy as np
import paddle

from . import config
from .barcode import read_barcodes
from .locate import locate


def _resize(img: np.ndarray) -> np.ndarray:
    scale = config.MAX_SIDE / max(img.shape[:2])
    if scale >= 1:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


class OcrPipeline:
    def __init__(self):
        from paddleocr import PaddleOCR

        cpu = config.DEVICE == "cpu"
        kw = {"text_recognition_model_dir": config.REC_DIR} if config.REC_DIR else {}
        self.ocr = PaddleOCR(
            text_detection_model_name=config.DET_MODEL,
            text_recognition_model_name=config.REC_MODEL,
            use_doc_orientation_classify=True,
            use_doc_unwarping=config.UNWARP,
            use_textline_orientation=True,
            device=config.DEVICE,
            enable_mkldnn=cpu and not paddle.__version__.startswith("3.3"),  # oneDNN bug in 3.3
            **kw,
        )

    def run(self, img: np.ndarray) -> dict:
        t0 = time.perf_counter()
        img = _resize(img)
        flat, quad = locate(img)
        t1 = time.perf_counter()
        lines = []
        for res in self.ocr.predict(flat):
            rows = zip(res["rec_texts"], res["rec_scores"], res["rec_polys"], strict=True)
            for text, score, poly in rows:
                if score >= config.MIN_SCORE and text.strip():
                    lines.append({"text": text, "score": round(float(score), 4),
                                  "box": np.asarray(poly).astype(int).tolist()})
        t2 = time.perf_counter()
        barcodes = read_barcodes(flat)
        t3 = time.perf_counter()
        return {
            "waybill": {"found": quad is not None, "quad": quad},
            "lines": lines,
            "text": "\n".join(x["text"] for x in lines),
            "barcodes": barcodes,
            "timing_ms": {"locate": int((t1 - t0) * 1e3), "ocr": int((t2 - t1) * 1e3),
                          "barcode": int((t3 - t2) * 1e3), "total": int((t3 - t0) * 1e3)},
        }
