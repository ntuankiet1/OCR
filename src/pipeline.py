"""photo → find the label → straighten → orientation → OCR (+ barcodes)."""

import os
import time

import numpy as np

from .barcode import BarcodeReader
from .config import Settings, settings
from .detector import ShippingLabelDetector
from .extract import Found, LabelExtractor
from .locate import LabelLocator, LocatedLabel
from .ocr import TextReader
from .quality import QualityChecker
from .utils import pick_providers, resize_max_side, unrotate


class OcrPipeline:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        providers = pick_providers(cfg.providers, cfg.device)
        self.detector = self._load_detector(providers)
        self.locator = LabelLocator(self.detector, cfg.crop_pad, cfg.locator_min_area)
        self.reader = TextReader(cfg, providers)
        self.barcodes = BarcodeReader()
        self.extractor = LabelExtractor()
        self.quality = QualityChecker(cfg.blur_threshold)

    def _load_detector(self, providers: list[str]) -> ShippingLabelDetector | None:
        path = self.cfg.detector
        if path and os.path.exists(path):
            detector = ShippingLabelDetector(
                path, self.cfg.detector_conf, providers, self.cfg.detector_nms_iou
            )
            print(f"shipping-label detector providers: {detector.session.get_providers()}")
            return detector
        if path:
            print(f"shipping-label detector {path} not found, using the OpenCV fallback")
        return None

    def run(self, img: np.ndarray) -> dict:
        t0 = time.perf_counter()
        img = resize_max_side(img, self.cfg.max_side)
        label = self.locator.locate(img)
        t1 = time.perf_counter()
        lines, angle = self.reader.read(label.image)
        t2 = time.perf_counter()
        barcodes = self.barcodes.read(label.image)
        t3 = time.perf_counter()
        return {
            "waybill": {
                "found": label.quad is not None,
                "quad": label.quad,
                "detector": self.detector is not None,
            },
            "lines": lines,
            "text": "\n".join(x["text"] for x in lines),
            "barcodes": barcodes,
            "reading": self.build_reading(img, label, lines, angle, barcodes),
            "timing_ms": {
                "locate": int((t1 - t0) * 1e3),
                "ocr": int((t2 - t1) * 1e3),
                "barcode": int((t3 - t2) * 1e3),
                "total": int((t3 - t0) * 1e3),
            },
        }

    def build_reading(
        self, photo: np.ndarray, label: LocatedLabel, lines: list[dict], angle: int, barcodes: list
    ) -> dict:
        """The label's fields in Parcel's contract: value, confidence and a box on the photo."""
        h, w = label.image.shape[:2]
        ph, pw = photo.shape[:2]

        def box(points: list) -> dict | None:
            if not points:
                return None
            pts = unrotate(np.array(points, np.float32), angle, w, h)
            x, y = label.to_photo(pts).T
            x0, x1 = float(np.clip(x.min() / pw, 0, 1)), float(np.clip(x.max() / pw, 0, 1))
            y0, y1 = float(np.clip(y.min() / ph, 0, 1)), float(np.clip(y.max() / ph, 0, 1))
            return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}

        def field(found: Found | None) -> dict:
            if found is None:
                return {"value": None, "confidence": 0.0, "box": None}
            if found.barcode is not None:  # barcode corners are in the (unrotated) label image
                b = label.to_photo(np.array(barcodes[found.barcode]["box"], np.float32))
                x, y = b.T
                x0, x1 = float(np.clip(x.min() / pw, 0, 1)), float(np.clip(x.max() / pw, 0, 1))
                y0, y1 = float(np.clip(y.min() / ph, 0, 1)), float(np.clip(y.max() / ph, 0, 1))
                where = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
            else:   
                where = box([p for i in found.lines for p in lines[i]["box"]])
            return {"value": found.value, "confidence": round(found.confidence, 4), "box": where}

        found = self.extractor.extract(lines, barcodes)
        return {
            "fields": {name: field(f) for name, f in found.items()},
            "label_count": max(label.label_count, 1 if lines else 0),
            "quality_flags": self.quality.flags(label.image),
            "model_version": f"{self.cfg.model_version}+{self.extractor.version}",
        }
