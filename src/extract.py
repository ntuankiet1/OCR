"""Regex/anchor/geometry based reading of the label fields from OCR lines and barcodes (v1).

Deliberately conservative: when it is not sure it returns nothing, and Parcel then sends the job
to an officer. All brand knowledge (keywords, anchors, formats) is in label_rules.yaml; replace
this class with a learned extractor later without touching the API.

How it reads one label, field by field:
  tracking barcode  the first 1D barcode; else a QR whose whole text looks like a tracking number
  tracking text     a printed line equal to the barcode, else one matching a carrier's format,
                    else one that looks like a tracking number (letters + digits)
  carrier           the tracking format, else the carrier keywords found in any line
  recipient/address one of three layouts, tried in order: labelled rows (Ninja Van), a recipient
                    anchor (most Thai labels), a column under a header (DHL Paket)
"""

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

RULES_PATH = Path(__file__).with_name("label_rules.yaml")
_LINEAR = {"QRCode", "MicroQRCode", "RMQRCode", "DataMatrix", "Aztec", "PDF417", "MaxiCode"}


@dataclass
class Found:
    value: str
    confidence: float
    lines: list[int]  # indexes of the OCR lines it came from (for its box); [] for a barcode
    barcode: int | None = None  # index of the barcode it came from


def norm(text: str) -> str:
    """NFKC, single spaces, no space before a colon ("TO :" -> "TO:")."""
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    return re.sub(r"\s+([:：])", r"\1", text)


def fold(text: str) -> str:
    return norm(text).casefold()


def tracking_key(text: str) -> str:
    return re.sub(r"[\s\-]", "", unicodedata.normalize("NFKC", text)).upper()


_THAI_TONES = "\u0e48-\u0e4b"  # ่ ้ ๊ ๋: the OCR often drops them, especially on small or soft labels


def compile_anchor(anchor: str) -> re.Pattern:
    """A pattern for a line-leading anchor. Thai tone marks are optional ("ที่อยู่" also matches
    "ที่อยู"), and a Latin anchor must not be glued to more letters or digits ("to:" vs "tom")."""
    base = re.sub(f"[{_THAI_TONES}]", "", fold(anchor))
    # A Thai anchor may have one or two stray characters in front: a photographed label read as
    # "ฝผู้รับ (TO) ..." (a sticker edge or crease read as a letter).
    noise = "" if anchor.isascii() else r"[^\s\d:]{0,2}"
    body = noise + "".join(
        re.escape(c) + (f"[{_THAI_TONES}]?" if "\u0e01" <= c <= "\u0e4e" else "") for c in base
    )
    return re.compile(body + (r"(?![A-Za-z0-9])" if anchor.isascii() else ""), re.IGNORECASE)


def after_anchor(text: str, anchors: list[re.Pattern]) -> str | None:
    """The text after a leading anchor ("ผู้รับ: OMS" -> "OMS"), None when no anchor starts it.
    Thai anchors are often glued to the value."""
    clean = norm(text)
    for anchor in anchors:
        match = anchor.match(clean)
        if match:
            return clean[match.end() :].strip(" :：-\t")
    return None


def space_digits_and_thai(text: str) -> str:
    """"123/45หมู่6" -> "123/45 หมู่ 6": the OCR often drops the space between a number and a Thai
    word, and a unit number glued to Thai letters is not recognised as a unit."""
    text = re.sub("([0-9])([\u0e01-\u0e4e])", "\\1 \\2", text)
    return re.sub("([\u0e01-\u0e4e])([0-9])", "\\1 \\2", text)


def squash_marks(text: str) -> str:
    """Drop a repeated Thai vowel/tone mark ("ดีี" -> "ดี"): the OCR doubles them on soft images and
    Thai never writes the same mark twice in a row."""
    return re.sub("([\u0e31\u0e34-\u0e3a\u0e47-\u0e4e])\\1+", "\\1", text)


def after_marker(text: str, markers: list[re.Pattern]) -> str | None:
    """The text after a marker such as "(TO)" found anywhere in the line, else None."""
    clean = norm(text)
    for marker in markers:
        match = marker.search(clean)
        if match:
            return clean[match.end() :].strip(" :：-\t")
    return None


def line_height(line: dict) -> float:
    """The text height from the box shape (its shorter side), so a tilted line does not look taller
    than it is, as its axis-aligned bounding box does."""
    (x0, y0), (x1, y1), (x2, y2) = line["box"][:3]
    return min(math.hypot(x1 - x0, y1 - y0), math.hypot(x2 - x1, y2 - y1))


def bbox(line: dict) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in line["box"]], [p[1] for p in line["box"]]
    return min(xs), min(ys), max(xs), max(ys)


class LabelExtractor:
    def __init__(self, rules_path: Path = RULES_PATH):
        r = yaml.safe_load(Path(rules_path).read_text())
        self.version: str = r["version"]
        self.carriers: dict = r["carriers"]
        self.patterns = {
            code: [re.compile(p) for p in c.get("tracking_patterns", [])]
            for code, c in self.carriers.items()
        }
        self.generic = re.compile(r["generic_tracking_pattern"])
        self.by_display = {c["display"]: code for code, c in self.carriers.items()}
        self.tracking_from = {
            code: [compile_anchor(a) for a in c["tracking_from"]]
            for code, c in self.carriers.items()
            if "tracking_from" in c
        }
        self.tracking_digits = {
            code: re.compile(c["tracking_digits"])
            for code, c in self.carriers.items()
            if "tracking_digits" in c
        }
        self.confidence: dict = r["confidence"]
        lower = lambda key: [fold(a) for a in r[key]]  # noqa: E731
        self.recipient_anchors = [compile_anchor(a) for a in r["recipient_anchors"]]
        self.address_anchors = [compile_anchor(a) for a in r["address_anchors"]]
        self.recipient_markers = [
            re.compile(rf"(?<![A-Za-z]){re.escape(fold(m))}(?![A-Za-z])", re.IGNORECASE)
            for m in r["recipient_markers"]
        ]
        self.stop_anchors = lower("stop_anchors")
        self.labelled_headers = lower("labelled_headers")
        self.labelled_stops = lower("labelled_stops")
        self.name_labels = lower("name_labels")
        self.address_labels = lower("address_labels")
        self.name_prefixes = lower("name_prefixes")
        self.stop_contains = lower("stop_contains")
        self.max_address_lines: int = r["max_address_lines"]
        self.gap_lines: float = r["gap_lines"]
        self.ignore_inline = re.compile(r["ignore_inline_value"])
        block = r["column_blocks"]
        self.block_anchors = [fold(a) for a in block["anchors"]]
        self.block_captions = [fold(a) for a in block["caption_keywords"]]
        self.block_stops = [fold(a) for a in block["stop_keywords"]]
        self.block_max_lines: int = block["max_lines"]

    def extract(self, lines: list[dict], barcodes: list[dict]) -> dict[str, Found | None]:
        rows = self.merge_rows(lines)  # the layout rules read rows, not detector pieces
        barcode = self._barcode(barcodes)
        tracking = self._tracking(lines, barcode)
        carrier = self._carrier(lines, barcodes, tracking)
        if tracking is None and carrier is not None:  # a number only this carrier prints
            tracking = self._carrier_tracking(lines, rows, self.by_display[carrier.value])
        # Labelled rows (Ninja Van) and a column under a header (DHL) are written around separate
        # pieces ("Address:" beside its value), so they read the original lines. The anchor rules
        # read merged rows.
        recipient, address = self._labelled_block(lines)
        if recipient is None:
            name, where = self._anchored_block(rows)
            recipient, address = self._to_original(name, rows), self._to_original(where, rows)
        if recipient is None and address is None:
            recipient, address = self._column_block(lines)
        if address is None:  # "ที่อยู่: ..." on its own
            address = self._to_original(self._address(rows), rows)
        found = {
            "TRACKING_BARCODE": barcode,
            "TRACKING_TEXT": tracking,
            "CARRIER": carrier,
            "RECIPIENT": recipient,
            "ADDRESS": address,
            "PARCEL_TYPE": None,  # no size model yet: the officer picks it
        }
        return {name: self._calibrate(name, f) for name, f in found.items()}

    @staticmethod
    def _to_original(found: Found | None, rows: list[dict]) -> Found | None:
        """Point a field read from merged rows back at the original lines it came from."""
        if found is not None and found.barcode is None:
            found.lines = sorted({p for r in found.lines for p in rows[r]["parts"]})
        return found

    def _calibrate(self, name: str, found: Found | None) -> Found | None:
        """Lower the confidence of a field read by rules (see `confidence` in the rules file)."""
        if found is None or name not in self.confidence:
            return found
        c = self.confidence[name]
        found.confidence = round(min(found.confidence * c["factor"], c["max"]), 4)
        return found

    # -- tracking
    def _barcode(self, barcodes: list[dict]) -> Found | None:
        usable = [(i, b) for i, b in enumerate(barcodes) if b["text"].strip()]
        for i, b in usable:
            if b["format"] not in _LINEAR:
                return Found(b["text"].strip(), 0.99, [], barcode=i)
        for i, b in usable:  # a QR that is only a tracking number
            if self._looks_like_tracking(tracking_key(b["text"])):
                return Found(b["text"].strip(), 0.99, [], barcode=i)
        return None

    def _looks_like_tracking(self, key: str) -> bool:
        known = any(p.fullmatch(key) for ps in self.patterns.values() for p in ps)
        return known or bool(self.generic.fullmatch(key))

    def _tracking(self, lines: list[dict], barcode: Found | None) -> Found | None:
        key_of_barcode = tracking_key(barcode.value) if barcode else None
        candidates = []
        for i, line in enumerate(lines):
            key = tracking_key(line["text"])
            known = any(p.fullmatch(key) for ps in self.patterns.values() for p in ps)
            printed_barcode = key == key_of_barcode  # the digits printed under the barcode
            if known or printed_barcode or self.generic.fullmatch(key):
                candidates.append((key, printed_barcode or known, line["score"], i))
        if not candidates:
            return None
        if key_of_barcode:
            for key, _, score, i in candidates:
                if key == key_of_barcode:
                    return Found(key, score, [i])
        key, _, score, i = sorted(candidates, key=lambda c: (not c[1], -c[2]))[0]
        return Found(key, score, [i])

    def _carrier_tracking(self, lines: list[dict], rows: list[dict], code: str) -> Found | None:
        """A tracking number that only this carrier prints: its order number (Shopee Food), or the
        one digits-only line under the barcode (J&T)."""
        for i, row in enumerate(rows):
            value = after_anchor(row["text"], self.tracking_from.get(code, []))
            key = tracking_key((value or "").lstrip(" .:：-"))
            if value is not None and re.fullmatch(r"[A-Z0-9]{8,20}", key):
                return self._to_original(Found(key, row["score"], [i]), rows)
        pattern = self.tracking_digits.get(code)
        if pattern:
            fits = [(tracking_key(x["text"]), i) for i, x in enumerate(lines)]
            fits = [(key, i) for key, i in fits if pattern.fullmatch(key)]
            if len(fits) == 1:  # two such lines: cannot tell which is the number
                key, i = fits[0]
                return Found(key, lines[i]["score"], [i])
        return None

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
            if code in hits:  # already matched by its tracking pattern: keep that, box the logo
                hits[code] = Found(c["display"], hits[code].confidence, [logo[0]])
                continue
            hits[code] = Found(c["display"], best[1]["score"], [logo[0]])
        # Two different carriers on one label: do not choose (the officer does).
        return next(iter(hits.values())) if len(hits) == 1 else None

    # -- recipient and address: shared helpers
    def _clean_name(self, text: str) -> str:
        text = re.sub(r"^\([^)]{1,12}\)\s*", "", text.strip())  # "(TO) คุณ..." -> "คุณ..."
        low = text.casefold()
        for prefix in sorted(self.name_prefixes, key=len, reverse=True):
            if low.startswith(prefix) and len(text) > len(prefix):
                text, low = text[len(prefix) :].lstrip(), low[len(prefix) :].lstrip()
                break
        return "" if self.ignore_inline.match(text) else squash_marks(text.strip())

    def _stops(self, text: str) -> bool:
        low = fold(text)
        return (
            any(low.startswith(a) for a in self.stop_anchors)
            or any(w in low for w in self.stop_contains)
            or bool(self.generic.fullmatch(tracking_key(text)))  # a tracking number ends it
        )

    def _below_aligned(
        self, lines: list[dict], skip: set[int], left: float, top_y: float, bottom: float,
        height: float, tol: float,
    ) -> list[int]:
        """Lines under (top_y) that start at `left` (+-tol), one under the other, in reading order.
        `bottom` is where the previous line ended, `height` its line height."""
        picked = []
        below = []
        for j, line in enumerate(lines):
            x0, y0, _, y1 = bbox(line)
            if j not in skip and abs(x0 - left) <= tol and (y0 + y1) / 2 > top_y:
                below.append((y0, j))
        for _, j in sorted(below):
            x0, y0, _, y1 = bbox(lines[j])
            if self._stops(lines[j]["text"]) or y0 - bottom > self.gap_lines * height:
                break
            picked.append(j)
            bottom, height = max(bottom, y1), y1 - y0
            if len(picked) == self.max_address_lines:
                break
        return picked

    def _join(self, lines: list[dict], idx: list[int]) -> Found | None:
        """The lines as one text; a leading "ที่อยู่:" style anchor is dropped from a line."""
        parts = []
        for j in idx:
            text = lines[j]["text"].strip()
            rest = after_anchor(text, self.address_anchors)
            text = text if rest is None else rest
            if text:
                parts.append(space_digits_and_thai(squash_marks(text)))
        if not parts:
            return None
        return Found(" ".join(parts), min(lines[j]["score"] for j in idx), idx)

    # -- layout 1: "Name:" / "Address:" rows under a header (Ninja Van)
    def _labelled_block(self, lines: list[dict]) -> tuple[Found | None, Found | None]:
        for hi, header in enumerate(lines):
            if not any(fold(header["text"]).startswith(h) for h in self.labelled_headers):
                continue
            _, hy0, _, hy1 = bbox(header)
            region = [
                j for j, line in enumerate(lines)
                if j != hi and bbox(line)[1] >= hy1 - 2
            ]
            region.sort(key=lambda j: bbox(lines[j])[1])
            cut = next(
                (k for k, j in enumerate(region)
                 if any(fold(lines[j]["text"]).startswith(s) for s in self.labelled_stops)),
                len(region),
            )
            region = region[:cut]
            name = self._labelled_value(lines, region, self.name_labels)
            address_idx = self._labelled_value(lines, region, self.address_labels, rows=True)
            if name:
                clean = self._clean_name(lines[name[0]]["text"])
                found = Found(clean, lines[name[0]]["score"], name[:1]) if clean else None
                return found, self._join(lines, address_idx or [])
        return None, None

    def _labelled_value(
        self, lines: list[dict], region: list[int], labels: list[str], rows: bool = False
    ) -> list[int] | None:
        """The line to the right of a label on the same row (and, for an address, the lines
        under that value)."""
        for j in region:
            if not any(fold(lines[j]["text"]).startswith(lab) for lab in labels):
                continue
            lx0, ly0, lx1, ly1 = bbox(lines[j])
            h = ly1 - ly0
            same_row = [
                k for k in region
                if k != j and lx1 - 5 <= bbox(lines[k])[0] <= lx1 + 4 * h
                and abs((bbox(lines[k])[1] + bbox(lines[k])[3]) / 2 - (ly0 + ly1) / 2) <= 0.6 * h
            ]
            if not same_row:
                continue
            first = min(same_row, key=lambda k: bbox(lines[k])[0])
            if not rows:
                return [first]
            vx0, vy0, _, vy1 = bbox(lines[first])
            more = self._below_aligned(
                [lines[k] for k in region], set(), vx0, vy1 - 1, vy1, vy1 - vy0, 1.5 * h + 10
            )
            return [first] + [region[m] for m in more if region[m] != first]
        return None

    def _row_of(
        self, lines: list[dict], i: int, skip: set[int] = frozenset(), gap: float = 3.0
    ) -> list[int]:
        """Line `i` with the lines on its row to the left and right, chained while the gap between
        neighbours stays within `gap` text heights, in reading order. The detector cuts a row into
        one box per word or per group of words, depending on the photo, so one field can arrive in
        several pieces (whose boxes are not perfectly level)."""
        x0, y0, x1, y1 = bbox(lines[i])
        h = line_height(lines[i])
        mid = (y0 + y1) / 2
        def beside(k: int) -> bool:
            """On the same row, side by side: close in height and not stacked (overlapping in x)."""
            kx0, ky0, kx1, ky1 = bbox(lines[k])
            level = abs((ky0 + ky1) / 2 - mid) <= 0.4 * max(h, line_height(lines[k]))
            overlap = min(x1, kx1) - max(x0, kx0)
            return level and overlap <= 0.3 * min(x1 - x0, kx1 - kx0)

        row = [
            (bbox(ln)[0], k)
            for k, ln in enumerate(lines)
            if k == i or (k not in skip and beside(k))
        ]
        row.sort()
        at = [k for _, k in row].index(i)
        lo = hi = at
        while hi + 1 < len(row) and bbox(lines[row[hi + 1][1]])[0] - max(
            bbox(lines[row[j][1]])[2] for j in range(lo, hi + 1)
        ) <= gap * h:
            hi += 1
        while lo > 0 and bbox(lines[row[lo][1]])[0] - max(
            bbox(lines[row[j][1]])[2] for j in range(lo - 1, lo)
        ) <= gap * h:
            lo -= 1
        return [k for _, k in row[lo : hi + 1]]

    def merge_rows(self, lines: list[dict], gap: float = 1.5) -> list[dict]:
        """One line per row: the pieces the detector cut a row into (a word, a group of words)
        are joined left to right when the gap between them is within `gap` text heights. Each
        merged line has the lowest score of its pieces and `parts`, the indexes of the original
        lines, so a field read from it still points at the right boxes. A lone line is unchanged
        (apart from `parts`). Columns stay apart: their gutter is wider than a word gap."""
        used: set[int] = set()
        rows = []
        for i in range(len(lines)):
            if i in used:
                continue
            group = self._row_of(lines, i, used, gap)
            used.update(group)
            if len(group) == 1:
                rows.append({**lines[i], "parts": [i]})
                continue
            points = np.array([pt for k in group for pt in lines[k]["box"]], np.float32)
            rect = cv2.boxPoints(cv2.minAreaRect(points))
            rows.append(
                {
                    "text": " ".join(lines[k]["text"].strip() for k in group),
                    "score": min(lines[k]["score"] for k in group),
                    "box": [[int(x), int(y)] for x, y in rect],
                    "parts": group,
                }
            )
        return rows

    def _row_after(self, lines: list[dict], i: int, skip: set[int] = frozenset()) -> list[int]:
        """The pieces of line `i`'s row that lie to its right (see `_row_of`)."""
        right = bbox(lines[i])[0]
        return [
            k for k in self._row_of(lines, i, skip) if k != i and bbox(lines[k])[0] >= right
        ]

    # -- layout 2: a recipient anchor (KEX, Flash, J&T, SPX, Shopee Food, ...)
    def _anchored_block(self, lines: list[dict]) -> tuple[Found | None, Found | None]:
        for ai, header in enumerate(lines):
            value = after_anchor(header["text"], self.recipient_anchors)
            if value is None:
                value = after_marker(header["text"], self.recipient_markers)
            if value is None:
                continue
            x0, y0, x1, y1 = bbox(header)
            h = y1 - y0
            # The name on the anchor's row: after the anchor in the same line and/or in the pieces
            # to its right ("ผู้รับ" "(TO)" "Z1" can be three lines).
            row = self._row_after(lines, ai)
            name = self._clean_name(" ".join([value] + [lines[k]["text"].strip() for k in row]))
            name_idx = [ai, *row] if self._clean_name(value) else row
            if not name:  # else the first line below it, with the pieces beside it ("คุณ" "ณัฐวุฒิ")
                below = self._below_aligned(
                    lines, {ai, *row}, x0, (y0 + y1) / 2 + 0.3 * h, y1, h, 1.5 * h + 20
                )[:1]
                if below:
                    name_idx = self._row_of(lines, below[0], {ai, *row})
                    name = self._clean_name(" ".join(lines[k]["text"].strip() for k in name_idx))
            if not name:
                continue
            score = min(lines[k]["score"] for k in name_idx)
            boxes = [bbox(lines[k]) for k in name_idx]
            bottom = max(y1, max(b[3] for b in boxes)) if name_idx != [ai] else y1
            line_h = max([h] + [b[3] - b[1] for b in boxes])
            taken = {ai, *row, *name_idx}
            picked = self._below_aligned(
                lines, taken, x0, (y0 + y1) / 2 + 0.3 * h, bottom, line_h, 1.5 * h + 20
            )
            address_idx: list[int] = []  # each address row with the pieces beside it
            for j in picked:
                address_idx += [j] + [
                    k for k in self._row_after(lines, j, taken | set(address_idx)) if k != j
                ]
            return Found(name, score, name_idx), self._join(lines, address_idx)
        return None, None

    # -- layout 3: a column under a header (DHL Paket)
    def _column_block(self, lines: list[dict]) -> tuple[Found | None, Found | None]:
        for ai, header in enumerate(lines):
            text = fold(header["text"])
            if not any(text.startswith(a) for a in self.block_anchors):
                continue
            x0, y0, x1, y1 = bbox(header)
            tol, yc = 0.4 * (x1 - x0) + 30, (y0 + y1) / 2 + 0.3 * (y1 - y0)
            below = []
            for j, line in enumerate(lines):
                bx0, by0, _, by1 = bbox(line)
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
            return name, self._join(lines, picked[1:])
        return None, None

    # -- layout 0: "ที่อยู่: ..." on its own line plus the lines after it (KEX)
    def _address(self, lines: list[dict]) -> Found | None:
        for i, line in enumerate(lines):
            first = after_anchor(line["text"], self.address_anchors)
            if first is None:
                continue
            parts, used, score = ([first] if first else []), [i], line["score"]
            for j in range(i + 1, min(i + 1 + self.max_address_lines, len(lines))):
                text = lines[j]["text"].strip()
                if self._stops(text):
                    break
                parts.append(text)
                used.append(j)
                score = min(score, lines[j]["score"])
            if parts:
                return Found(" ".join(parts), score, used)
        return None
