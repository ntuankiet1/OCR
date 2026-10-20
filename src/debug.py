"""Step-by-step trace of one photo through the pipeline, for the /debug page (demo)."""

import base64
import time

import cv2
import numpy as np

from .pipeline import OcrPipeline
from .utils import resize_max_side

RED, GREEN, BLUE = (0, 0, 230), (0, 170, 0), (230, 120, 0)
ORANGE, PURPLE = (0, 140, 255), (200, 0, 200)
FIELD_COLORS = {
    "CARRIER": ORANGE,
    "TRACKING_BARCODE": BLUE,
    "TRACKING_TEXT": (0, 200, 200),
    "RECIPIENT": GREEN,
    "ADDRESS": PURPLE,
}


def _thickness(img: np.ndarray) -> int:
    return max(2, max(img.shape[:2]) // 400)


def draw_polys(img: np.ndarray, polys: list, color: tuple, thickness: int | None = None):
    out = img.copy()
    for poly in polys:
        pts = np.asarray(poly, np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color, thickness or _thickness(img))
    return out


def draw_label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple) -> None:
    """ASCII text only (OpenCV cannot draw Thai); Thai goes into the page's tables."""
    scale = max(0.5, max(img.shape[:2]) / 1400)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, _thickness(img))


def data_uri(img: np.ndarray, max_side: int = 1100) -> str:
    ok, buf = cv2.imencode(".jpg", resize_max_side(img, max_side), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class PipelineTracer:
    def __init__(self, pipeline: OcrPipeline):
        self.p = pipeline

    def trace(self, photo: np.ndarray) -> dict:
        p, cfg = self.p, self.p.cfg
        steps: list[dict] = []
        t_start = time.perf_counter()

        def add(title, note, images=(), table=None, t0=None):
            steps.append(
                {
                    "title": title,
                    "note": note,
                    "ms": None if t0 is None else int((time.perf_counter() - t0) * 1e3),
                    "images": [{"label": lab, "src": data_uri(im)} for lab, im in images],
                    "table": table,
                }
            )

        img = resize_max_side(photo, cfg.max_side)
        add(
            "1. Input photo",
            f"Shrunk so the long side is at most {cfg.max_side} px (OCR_MAX_SIDE): "
            f"{img.shape[1]}×{img.shape[0]} px.",
            [("photo", img)],
        )

        # 2. Label detector
        t0 = time.perf_counter()
        found = p.detector.detect(img) if p.detector else []
        shown = img.copy()
        for rank, (box, score) in enumerate(found):
            x0, y0, x1, y1 = (int(v) for v in box)
            cv2.rectangle(shown, (x0, y0), (x1, y1), RED, _thickness(img) * (2 if rank == 0 else 1))
            draw_label(shown, f"{score:.2f}", (x0, max(y0 - 8, 20)), RED)
        if p.detector:
            note = (
                f"{len(found)} label(s) found at confidence ≥ {cfg.detector_conf}; the best one "
                "(thick box) is used."
                if found
                else f"No label found at confidence ≥ {cfg.detector_conf}: the whole photo is used."
            )
        else:
            note = "No label detector is loaded: the whole photo is used."
        add("2. Shipping-label detector", note, [("detections", shown)], t0=t0)

        # 3. Crop, 4. straighten
        t0 = time.perf_counter()
        crop, origin, count = p.locator.crop_to_detection(img)
        add(
            "3. Crop to the label",
            f"Best box plus {cfg.crop_pad:.0%} padding: {crop.shape[1]}×{crop.shape[0]} px.",
            [("crop", crop)],
            t0=t0,
        )
        t0 = time.perf_counter()
        label = p.locator.straighten(crop, origin, count)
        if label.quad is not None:
            corners = (np.array(label.quad) - np.array(origin)).tolist()
            images = [
                ("corners found", draw_polys(crop, [corners], GREEN, _thickness(crop) * 2)),
                ("straightened", label.image),
            ]
            note = "Four corners found: the label is warped flat (perspective correction)."
        else:
            images = [("kept as is", label.image)]
            note = (
                "No clean four-corner bright shape (label ≥ "
                f"{cfg.locator_min_area:.0%} of the crop), so the crop is used as is."
            )
        add("4. Straighten", note, images, t0=t0)

        # 5-8. Orientation, text detection, recognition
        t0 = time.perf_counter()
        reading = p.reader.read_detailed(label.image)
        ocr_ms = int((time.perf_counter() - t0) * 1e3)
        add(
            "5. Orientation",
            f"The document-orientation classifier turned the label by {reading.angle}° "
            "(counter-clockwise); 0° means it was already upright.",
            [("before", label.image), ("after", reading.image)],
        )
        add(
            "6. Text detection",
            f"The text detector (PP-OCRv5 server) found {len(reading.detected)} text regions.",
            [("text regions", draw_polys(reading.image, reading.detected, BLUE))],
        )
        kept = p.reader.keep(reading.lines)
        keep_ids = {id(x) for x in kept}
        flipped = sum(x["flipped"] for x in reading.lines)
        shown = draw_polys(
            reading.image, [x["box"] for x in reading.lines if id(x) in keep_ids], GREEN
        )
        shown = draw_polys(shown, [x["box"] for x in reading.lines if id(x) not in keep_ids], RED)
        rows = [
            [
                i,
                x["text"],
                f"{x['score']:.2f}",
                "yes" if id(x) in keep_ids else f"no (< {cfg.min_score})",
                "yes" if x["flipped"] else "",
            ]
            for i, x in enumerate(reading.lines)
        ]
        add(
            "7. Text-line orientation and recognition",
            f"{len(reading.lines)} lines read (steps 5–7 took {ocr_ms} ms together); "
            f"{len(kept)} kept at score ≥ {cfg.min_score} (green), the rest dropped (red). "
            f"{flipped} line(s) were upside down and turned back before reading.",
            [("recognized lines", shown)],
            {"headers": ["#", "text", "score", "kept", "turned 180°"], "rows": rows},
        )

        # 9. Barcodes
        t0 = time.perf_counter()
        barcodes = p.barcodes.read(label.image)
        add(
            "8. Barcodes",
            f"{len(barcodes)} barcode(s) decoded from the straightened label (before rotation).",
            [("barcodes", draw_polys(label.image, [b["box"] for b in barcodes], ORANGE))],
            {
                "headers": ["text", "format"],
                "rows": [[b["text"], b["format"]] for b in barcodes],
            },
            t0=t0,
        )

        # 10. Fields in Parcel's contract
        t0 = time.perf_counter()
        result = p.build_reading(img, label, kept, reading.angle, barcodes)
        h, w = img.shape[:2]
        shown, rows = img.copy(), []
        for name, f in result["fields"].items():
            box = f["box"]
            conf = f"{f['confidence']:.2f}" if f["value"] else ""
            rows.append([name, f["value"], conf, bool(box)])
            if box and name in FIELD_COLORS:
                x0, y0 = int(box["x"] * w), int(box["y"] * h)
                x1, y1 = int((box["x"] + box["w"]) * w), int((box["y"] + box["h"]) * h)
                cv2.rectangle(shown, (x0, y0), (x1, y1), FIELD_COLORS[name], _thickness(img) * 2)
                draw_label(shown, name, (x0, max(y0 - 8, 20)), FIELD_COLORS[name])
        flags = ", ".join(result["quality_flags"]) or "none"
        add(
            "9. Fields sent to Parcel",
            f"Rules {p.extractor.version} turn the lines and barcodes into Parcel's fields; boxes "
            f"are mapped back onto the original photo. Labels: {result['label_count']}. "
            f"Quality flags: {flags}. Model version: {result['model_version']}.",
            [("what Parcel draws", shown)],
            {"headers": ["field", "value", "confidence", "box"], "rows": rows},
            t0=t0,
        )
        return {"steps": steps, "total_ms": int((time.perf_counter() - t_start) * 1e3)}
