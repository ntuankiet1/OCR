"""Regex/anchor based reading of the label fields from OCR lines and barcodes (v1).

Deliberately conservative: when it is not sure it returns nothing, and Parcel then sends the job
to an officer. The rules live in label_rules.yaml; replace this class with a learned extractor
later without touching the API.
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

RULES_PATH = Path(__file__).with_name("label_rules.yaml")
_LINEAR = {"QRCode", "MicroQRCode", "RMQRCode", "DataMatrix", "Aztec", "PDF417", "MaxiCode"}


@dataclass
class Found:
    value: str
    confidence: float
    lines: list[int]  # indexes of the OCR lines it came from (for its box); [] for a barcode
    barcode: int | None = None  # index of the barcode it came from


def fold(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def tracking_key(text: str) -> str:
    return re.sub(r"[\s\-]", "", unicodedata.normalize("NFKC", text)).upper()


def after_anchor(text: str, anchors: list[str]) -> str | None:
    """The text after a leading anchor ("ผู้รับ: OMS" -> "OMS"), None when no anchor starts it.
    Latin anchors need a word boundary; Thai ones are often glued to the value."""
    low = fold(text)
    for anchor in anchors:
        if not low.startswith(anchor):
            continue
        if anchor.isascii() and len(low) > len(anchor) and low[len(anchor)].isalnum():
            continue
        return low_to_original(text, len(anchor)).strip(" :：-\t")
    return None


def low_to_original(text: str, n: int) -> str:
    """Drop the first `n` characters of the folded form from the original text."""
    norm = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    return norm[n:]


class LabelExtractor:
    def __init__(self, rules_path: Path = RULES_PATH):
        rules = yaml.safe_load(Path(rules_path).read_text())
        self.version: str = rules["version"]
        self.carriers: dict = rules["carriers"]
        self.patterns = {
            code: [re.compile(p) for p in c.get("tracking_patterns", [])]
            for code, c in self.carriers.items()
        }
        self.generic = re.compile(rules["generic_tracking_pattern"])
        self.recipient_anchors = [fold(a) for a in rules["recipient_anchors"]]
        self.address_anchors = [fold(a) for a in rules["address_anchors"]]
        self.stop_anchors = [fold(a) for a in rules["stop_anchors"]]
        self.max_address_lines: int = rules["max_address_lines"]
        block = rules["column_blocks"]
        self.block_anchors = [fold(a) for a in block["anchors"]]
        self.block_captions = [fold(a) for a in block["caption_keywords"]]
        self.block_stops = [fold(a) for a in block["stop_keywords"]]
        self.block_max_lines: int = block["max_lines"]

    def extract(self, lines: list[dict], barcodes: list[dict]) -> dict[str, Found | None]:
        barcode = self._barcode(barcodes)
        tracking = self._tracking(lines, barcode)
        recipient, address = self._recipient(lines), self._address(lines)
        if recipient is None and address is None:  # no "ผู้รับ:" style line: try a form column
            recipient, address = self._column_block(lines)
        return {
            "TRACKING_BARCODE": barcode,
            "TRACKING_TEXT": tracking,
            "CARRIER": self._carrier(lines, barcodes, tracking),
            "RECIPIENT": recipient,
            "ADDRESS": address,
            "PARCEL_TYPE": None,  # no size model yet: the officer picks it
        }

    # -- tracking
    @staticmethod
    def _barcode(barcodes: list[dict]) -> Found | None:
        for i, b in enumerate(barcodes):
            if b["format"] not in _LINEAR and b["text"].strip():
                return Found(b["text"].strip(), 0.99, [], barcode=i)
        return None

    def _tracking(self, lines: list[dict], barcode: Found | None) -> Found | None:
        candidates = []
        for i, line in enumerate(lines):
            key = tracking_key(line["text"])
            known = any(p.fullmatch(key) for ps in self.patterns.values() for p in ps)
            if known or self.generic.fullmatch(key):
                candidates.append((key, known, line["score"], i))
        if not candidates:
            return None
        if barcode:
            for key, _, score, i in candidates:
                if key == tracking_key(barcode.value):
                    return Found(key, score, [i])
        key, _, score, i = sorted(candidates, key=lambda c: (not c[1], -c[2]))[0]
        return Found(key, score, [i])

    # -- carrier
    def _carrier(
        self, lines: list[dict], barcodes: list[dict], tracking: Found | None
    ) -> Found | None:
        hits: dict[str, Found] = {}
        if tracking:
            for code, patterns in self.patterns.items():
                if any(p.fullmatch(tracking.value) for p in patterns):
                    hits[code] = Found(self.carriers[code]["display"], 0.99, tracking.lines)
        for code, c in self.carriers.items():
            names = [fold(k) for k in c["keywords"]]
            kws = [re.compile(rf"(?<!\w){re.escape(k)}(?!\w)") for k in names]
            seen = [
                (i, line)
                for i, line in enumerate(lines)
                if any(k.search(fold(line["text"])) for k in kws)
            ]
            if not seen:
                continue
            # Several lines can name the carrier; the surest one sets the confidence and the line
            # that is only the name (the logo) is where the box goes, else the surest line.
            best = max(seen, key=lambda t: t[1]["score"])
            logo = next((t for t in seen if fold(t[1]["text"]) in names), best)
            if code in hits:  # already matched by its tracking pattern
                continue
            hits[code] = Found(c["display"], best[1]["score"], [logo[0]])
        # Two different carriers on one label: do not choose (the officer does).
        return next(iter(hits.values())) if len(hits) == 1 else None

    # -- recipient and address
    def _recipient(self, lines: list[dict]) -> Found | None:
        for i, line in enumerate(lines):
            value = after_anchor(line["text"], self.recipient_anchors)
            if value is None:
                continue
            if value:
                return Found(value, line["score"], [i])
            if i + 1 < len(lines):  # the name is on the next line
                nxt = lines[i + 1]
                return Found(nxt["text"].strip(), min(line["score"], nxt["score"]), [i + 1])
        return None

    @staticmethod
    def _bbox(line: dict) -> tuple[float, float, float, float]:
        xs, ys = [p[0] for p in line["box"]], [p[1] for p in line["box"]]
        return min(xs), min(ys), max(xs), max(ys)

    def _column_block(self, lines: list[dict]) -> tuple[Found | None, Found | None]:
        """Recipient name and address from the column of lines under a header (see rules)."""
        for ai, header in enumerate(lines):
            text = fold(header["text"])
            if not any(text.startswith(a) for a in self.block_anchors):
                continue
            x0, y0, x1, y1 = self._bbox(header)
            tol, yc = 0.4 * (x1 - x0) + 30, (y0 + y1) / 2 + 0.3 * (y1 - y0)
            below = []
            for j, line in enumerate(lines):
                bx0, by0, _, by1 = self._bbox(line)
                if j != ai and abs(bx0 - x0) <= tol and (by0 + by1) / 2 > yc:
                    below.append((by0 + by1, j))
            picked = []
            for _, j in sorted(below):
                low = fold(lines[j]["text"])
                if any(k in low for k in self.block_stops):
                    break
                if any(k in low for k in self.block_captions):
                    continue
                picked.append(j)
                if len(picked) == self.block_max_lines:
                    break
            if not picked:
                return None, None
            name = Found(lines[picked[0]]["text"].strip(), lines[picked[0]]["score"], picked[:1])
            rest = picked[1:]
            address = (
                Found(
                    " ".join(lines[j]["text"].strip() for j in rest),
                    min(lines[j]["score"] for j in rest),
                    rest,
                )
                if rest
                else None
            )
            return name, address
        return None, None

    def _address(self, lines: list[dict]) -> Found | None:
        for i, line in enumerate(lines):
            first = after_anchor(line["text"], self.address_anchors)
            if first is None:
                continue
            parts, used, score = ([first] if first else []), [i], line["score"]
            for j in range(i + 1, min(i + 1 + self.max_address_lines, len(lines))):
                text = lines[j]["text"].strip()
                if any(fold(text).startswith(a) for a in self.stop_anchors):
                    break
                parts.append(text)
                used.append(j)
                score = min(score, lines[j]["score"])
            if parts:
                return Found(" ".join(parts), score, used)
        return None
