"""Barcode / QR reading with zxing-cpp (no training). The barcode is the most reliable source
of the tracking number; OCR text is the cross-check (BR07: disagreement → review)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Barcode:
    text: str
    format: str
    polygon: list[tuple[float, float]]


def read_barcodes(image_bgr: np.ndarray) -> list[Barcode]:
    import zxingcpp

    out = []
    for r in zxingcpp.read_barcodes(image_bgr[:, :, ::-1].copy()):  # zxing expects RGB
        if not getattr(r, "valid", True) or not r.text:
            continue
        p = r.position
        poly = [
            (p.top_left.x, p.top_left.y),
            (p.top_right.x, p.top_right.y),
            (p.bottom_right.x, p.bottom_right.y),
            (p.bottom_left.x, p.bottom_left.y),
        ]
        out.append(Barcode(text=r.text, format=str(r.format), polygon=poly))
    return out
