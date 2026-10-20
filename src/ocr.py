"""Text detection + recognition (PaddleOCR PP-OCRv5), on Paddle or on the ONNX copies."""

from dataclasses import dataclass

import numpy as np
from paddleocr import PaddleOCR

from .config import Settings


@dataclass
class TextReading:
    image: np.ndarray  # the orientation-corrected image every box below refers to
    angle: int  # rotation applied by the document-orientation step (degrees, counter-clockwise)
    detected: list  # every box the text detector found, before recognition
    lines: list[dict]  # every recognized line: text, score, box, flipped (turned 180 degrees)


class TextReader:
    def __init__(self, cfg: Settings, providers: list[str]):
        self.min_score = cfg.min_score
        kw: dict = {}
        if cfg.engine == "onnxruntime":
            d = cfg.onnx_dir
            kw = {
                "engine": "onnxruntime",
                "engine_config": {"onnxruntime": {"providers": providers}},
                "text_detection_model_dir": f"{d}/{cfg.det_model}",
                "text_recognition_model_dir": f"{d}/{cfg.rec_model}",
                "doc_orientation_classify_model_dir": f"{d}/PP-LCNet_x1_0_doc_ori",
                "textline_orientation_model_dir": f"{d}/PP-LCNet_x1_0_textline_ori",
            }
        elif cfg.rec_dir:
            kw = {"text_recognition_model_dir": cfg.rec_dir}
        for name in ("text_det_thresh", "text_det_box_thresh"):
            if getattr(cfg, name) is not None:
                kw[name] = getattr(cfg, name)
        self.ocr = PaddleOCR(
            text_detection_model_name=cfg.det_model,
            text_recognition_model_name=cfg.rec_model,
            use_doc_orientation_classify=True,
            use_doc_unwarping=cfg.unwarp,
            use_textline_orientation=True,
            device=cfg.device,
            enable_mkldnn=cfg.device == "cpu" and cfg.engine == "paddle" and not self._paddle_33(),
            **kw,
        )

    @staticmethod
    def _paddle_33() -> bool:
        import paddle

        return paddle.__version__.startswith("3.3")  # oneDNN is broken in 3.3

    def read_detailed(self, img: np.ndarray) -> TextReading:
        """Everything the OCR step produced, with nothing filtered out (the debug view uses it)."""
        res = next(iter(self.ocr.predict(img)))
        texts, scores, polys = res["rec_texts"], res["rec_scores"], res["rec_polys"]
        turned = list(res.get("textline_orientation_angles") or [])
        if len(turned) != len(texts):
            turned = [0] * len(texts)
        lines = [
            {
                "text": t,
                "score": round(float(sc), 4),
                "box": np.asarray(p).astype(int).tolist(),
                "flipped": angle == 180,
            }
            for t, sc, p, angle in zip(texts, scores, polys, turned, strict=True)
        ]
        return TextReading(
            image=res["doc_preprocessor_res"]["output_img"],
            angle=int(res["doc_preprocessor_res"]["angle"]),
            detected=[np.asarray(p).astype(int).tolist() for p in res["dt_polys"]],
            lines=lines,
        )

    def keep(self, lines: list[dict]) -> list[dict]:
        """The lines worth using: scoring at least min_score and not blank."""
        return [x for x in lines if x["score"] >= self.min_score and x["text"].strip()]

    def read(self, img: np.ndarray) -> tuple[list[dict], int]:
        """Lines (boxes in the orientation-corrected image) and the rotation applied (degrees)."""
        reading = self.read_detailed(img)
        return self.keep(reading.lines), reading.angle
