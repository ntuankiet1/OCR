"""v1 field extraction: carrier rules + anchors + regex over OCR lines and barcodes.

Deliberately conservative: when candidates disagree it reports a conflict instead of choosing
(BR07). Replace with a learned extractor once enough corrected jobs exist; keep this as the
fallback and as a feature source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.inference.barcode import Barcode
from src.inference.ocr import OcrLine
from src.text.normalize import normalize_common, normalize_tracking

# Thai text may be attached directly before a unit ("ห้องA/1203"), so the boundaries only
# exclude Latin letters, digits and "/".
_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9/])("
    r"[A-Z]{1,2}[/-]\d{2,5}"  # building + unit: A/1203, B-805
    r"|[A-Z]?\d{1,4}/\d{1,5}"  # address-style unit numbers: 99/123, A12/34
    r")(?![A-Za-z0-9/])"
)
_SEP_SPACES = re.compile(r"\s*([/-])\s*")


def _starts_with_anchor(text: str, anchor: str) -> bool:
    """Latin anchors need a word boundary ("to" must not match "tom"); Thai anchors are often
    written attached to the name ("ผู้รับสมชาย")."""
    if not text.startswith(anchor):
        return False
    if not anchor.isascii() or len(text) == len(anchor):
        return True
    return not text[len(anchor)].isalnum()


@dataclass
class FieldPrediction:
    value: str | None
    score: float
    polygon: list[tuple[float, float]] | None = None
    source: str = "ocr"  # ocr | barcode | rule


@dataclass
class Extraction:
    fields: dict[str, FieldPrediction] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


class RuleExtractor:
    def __init__(self, carriers_cfg: dict) -> None:
        self.carriers = carriers_cfg["carriers"]
        self.recipient_anchors = [a.casefold() for a in carriers_cfg["recipient_anchors"]]
        self.sender_anchors = [a.casefold() for a in carriers_cfg["sender_anchors"]]
        self.unit_anchors = [a.casefold() for a in carriers_cfg["unit_anchors"]]
        self.patterns = {
            code: [re.compile(p) for p in c.get("tracking_patterns", [])]
            for code, c in self.carriers.items()
        }

    # -- carrier
    def _carrier(self, lines: list[OcrLine], tracking: str | None) -> FieldPrediction:
        hits: dict[str, float] = {}
        for code, pats in self.patterns.items():
            if tracking and any(p.fullmatch(tracking) for p in pats):
                hits[code] = max(hits.get(code, 0.0), 0.99)
        for line in lines:
            text = normalize_common(line.text).casefold()
            for code, c in self.carriers.items():
                if any(re.search(rf"(?<!\w){re.escape(k)}(?!\w)", text) for k in c["keywords"]):
                    hits[code] = max(hits.get(code, 0.0), line.score)
        if len(hits) != 1:
            return FieldPrediction(None, 0.0, source="rule")
        code, score = next(iter(hits.items()))
        return FieldPrediction(code, score, source="rule")

    # -- tracking number
    def _tracking(
        self, lines: list[OcrLine], barcodes: list[Barcode], reasons: list[str]
    ) -> FieldPrediction:
        all_pats = [p for pats in self.patterns.values() for p in pats]

        def plausible(s: str) -> bool:
            return any(p.fullmatch(s) for p in all_pats) if all_pats else len(s) >= 10

        bc = {
            normalize_tracking(b.text): b for b in barcodes if plausible(normalize_tracking(b.text))
        }
        # OCR-only candidates need a known carrier format; free text that merely looks long
        # enough is not trusted until the label survey fills in carriers.yaml.
        ocr = {}
        for line in lines:
            cand = normalize_tracking(line.text)
            if all_pats and any(p.fullmatch(cand) for p in all_pats):
                ocr[cand] = line
        if len(bc) > 1:
            reasons.append("conflict_multiple_tracking_barcodes")
            return FieldPrediction(None, 0.0)
        if bc:
            value, b = next(iter(bc.items()))
            if ocr and value not in ocr:
                reasons.append("conflict_barcode_text_mismatch")
                return FieldPrediction(None, 0.0)
            return FieldPrediction(value, 0.999, b.polygon, source="barcode")
        if len(ocr) == 1:
            value, line = next(iter(ocr.items()))
            return FieldPrediction(value, line.score, line.polygon)
        if len(ocr) > 1:
            reasons.append("conflict_multiple_tracking_numbers")
        return FieldPrediction(None, 0.0)

    # -- recipient block
    def _anchor_index(self, lines: list[OcrLine], anchors: list[str]) -> int | None:
        for i, line in enumerate(lines):
            text = normalize_common(line.text).casefold()
            if any(_starts_with_anchor(text, a) for a in anchors):
                return i
        return None

    def _recipient(self, lines: list[OcrLine]) -> tuple[FieldPrediction, int | None]:
        i = self._anchor_index(lines, self.recipient_anchors)
        if i is None:
            return FieldPrediction(None, 0.0), None
        line = lines[i]
        text = normalize_common(line.text)
        for a in sorted(self.recipient_anchors, key=len, reverse=True):
            if _starts_with_anchor(text.casefold(), a):
                text = text[len(a) :].lstrip(" :.-")
                break
        if text:
            return FieldPrediction(text, line.score, line.polygon), i
        if i + 1 < len(lines):  # anchor on its own line → name on the next line
            nxt = lines[i + 1]
            return FieldPrediction(normalize_common(nxt.text), nxt.score, nxt.polygon), i
        return FieldPrediction(None, 0.0), i

    def _unit(self, lines: list[OcrLine], start: int | None, reasons: list[str]) -> FieldPrediction:
        # Search the recipient block (until a sender anchor) — never the sender block.
        stop = (
            self._anchor_index(lines[start + 1 :], self.sender_anchors)
            if start is not None
            else None
        )
        block = (
            lines[start : start + 1 + stop]
            if start is not None and stop is not None
            else lines[start or 0 :]
        )
        found = {}
        for line in block:
            text = _SEP_SPACES.sub(r"\1", normalize_common(line.text).upper())
            for m in _UNIT_RE.finditer(text):
                found[m.group(1)] = line
        if len(found) == 1:
            value, line = next(iter(found.items()))
            return FieldPrediction(value, line.score, line.polygon)
        if len(found) > 1:
            reasons.append("conflict_multiple_units")
        return FieldPrediction(None, 0.0)

    def extract(self, lines: list[OcrLine], barcodes: list[Barcode]) -> Extraction:
        ex = Extraction()
        tracking = self._tracking(lines, barcodes, ex.reasons)
        recipient, start = self._recipient(lines)
        ex.fields = {
            "tracking_number": tracking,
            "carrier": self._carrier(lines, tracking.value),
            "recipient_name": recipient,
            "unit": self._unit(lines, start, ex.reasons),
        }
        return ex
