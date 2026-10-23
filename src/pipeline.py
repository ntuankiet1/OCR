"""photo → find the label → straighten → orientation → OCR (+ barcodes)."""

import os
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .barcode import BarcodeReader
from .config import Settings, settings
from .detector import ShippingLabelDetector
from .extract import Found, LabelExtractor
from .locate import LabelLocator, LocatedLabel
from .ocr import TextReader, TextReading
from .quality import QualityChecker
from .utils import pick_providers, resize_max_side, unrotate


@dataclass
class Outcome:
    """The label and what the OCR read from it."""

    label: LocatedLabel
    reading: TextReading
    ignored: int = 0  # lines outside the label placed by the fallback (the box's own text)
    locate_ms: int = 0
    ocr_ms: int = 0


class OcrPipeline:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        providers = pick_providers(cfg.providers, cfg.device)
        self.detector = self._load_detector(providers)
        self.locator = LabelLocator(
            self.detector, cfg.crop_pad, cfg.locator_min_area, cfg.flat_labels
        )
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

    def read_label(self, img: np.ndarray) -> Outcome:
        """Find the label and read it. When the detector finds nothing and the image is not
        itself a label, the whole photo is read and the lines printed on bright paper (dark text
        on white) place the label; text outside it, such as a box's own printing, is ignored."""
        t0 = time.perf_counter()
        label = self.locator.locate(img)
        t1 = time.perf_counter()
        reading = self.reader.read_detailed(label.image)
        outcome = Outcome(label, reading, 0, int((t1 - t0) * 1e3))
        if not label.detected and self.cfg.fallback_localize:
            quad = self._paper_quad(label, reading)
            if quad is not None:
                inside = [x for x in reading.lines if self._inside(quad, label, reading, x)]
                outcome = Outcome(
                    replace(label, quad=quad.tolist()),
                    replace(reading, lines=inside),
                    len(reading.lines) - len(inside),
                    outcome.locate_ms,
                )
        outcome.ocr_ms = int((time.perf_counter() - t1) * 1e3)
        return outcome

    @staticmethod
    def _inside(quad: np.ndarray, label: LocatedLabel, reading: TextReading, line: dict) -> bool:
        """Whether the centre of an OCR line lies within `quad` (photo coordinates)."""
        h, w = label.image.shape[:2]
        centre = np.mean(np.array(line["box"], np.float32), axis=0, keepdims=True)
        x, y = label.to_photo(unrotate(centre, reading.angle, w, h))[0]
        polygon = quad.reshape(-1, 1, 2).astype(np.float32)
        return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0

    def _paper_quad(self, label: LocatedLabel, reading: TextReading) -> np.ndarray | None:
        """A rectangle around the lines that sit on bright paper (dark text on white), in photo
        coordinates. The box's own printed text (light on a dark colour) is left out."""
        gray = cv2.cvtColor(reading.image, cv2.COLOR_BGR2GRAY)
        points = []
        for line in self.reader.keep(reading.lines):
            xs, ys = zip(*line["box"], strict=True)
            region = gray[max(min(ys), 0) : max(ys) + 1, max(min(xs), 0) : max(xs) + 1]
            if region.size and float((region > 150).mean()) >= 0.55:
                points += line["box"]
        if len(points) < 4 * 5:  # fewer than five lines on paper: not enough to place a label
            return None
        h, w = label.image.shape[:2]
        pts = label.to_photo(unrotate(np.array(points, np.float32), reading.angle, w, h))
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(pts.astype(np.float32))
        grow = 1.12  # the text does not reach the paper's edge
        box = cv2.boxPoints(((cx, cy), (rw * grow + 8, rh * grow + 8), angle))
        return LabelLocator.order_quad(box)

    def run(self, img: np.ndarray) -> dict:
        t0 = time.perf_counter()
        img = resize_max_side(img, self.cfg.max_side)
        outcome = self.read_label(img)
        label, reading = outcome.label, outcome.reading
        lines = self.reader.keep(reading.lines)
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
            "reading": self.build_reading(img, label, lines, reading.angle, barcodes),
            "timing_ms": {
                "locate": outcome.locate_ms,
                "ocr": outcome.ocr_ms,
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
