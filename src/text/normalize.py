"""Text normalization for exact-match evaluation and matching.

AC02 requires the normalization rules to be agreed with the business BEFORE testing. Everything
here is a proposal until signed off; the version string is stamped into every bundle so a rule
change is visible in model lineage and job audit records (BR16).
"""

from __future__ import annotations

import re
import unicodedata

NORMALIZATION_VERSION = "0.1-proposal"

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_FULLWIDTH = str.maketrans({chr(c): chr(c - 0xFEE0) for c in range(0xFF01, 0xFF5F)} | {"　": " "})
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−－"})

_TONE_MARKS = "่้๊๋"  # mai ek, mai tho, mai tri, mai chattawa
# Above/below vowels and marks that must precede a tone mark in canonical typing order.
_PRE_TONE = "ัิีึืฺุู็"
_TONE_BEFORE_VOWEL = re.compile(f"([{_TONE_MARKS}])([{_PRE_TONE}])")
_DUP_COMBINING = re.compile(f"([{_TONE_MARKS}{_PRE_TONE}์ํ])\\1+")
_WS = re.compile(r"\s+")

# Thai honorifics are usually written attached to the name ("นายสมชาย"); longest first.
_THAI_HONORIFICS = ("นางสาว", "น.ส.", "ด.ช.", "ด.ญ.", "นาง", "นาย", "คุณ")
_LATIN_HONORIFIC = re.compile(r"^(mrs|mr|ms|miss|dr)\.?\s+")


def normalize_common(text: str) -> str:
    """Script-safe cleanup applied to every field.

    NFKC is intentionally NOT used: it decomposes Thai SARA AM (U+0E33) into NIKHAHIT + SARA AA.
    """
    s = unicodedata.normalize("NFC", text)
    s = s.translate(_FULLWIDTH).translate(_DASHES).translate(_THAI_DIGITS)
    # NIKHAHIT + SARA AA renders identically to SARA AM; OCR and typists produce both.
    s = s.replace("ํา", "ำ")
    # Tone mark typed before an above/below vowel renders the same; canonical order is vowel first.
    s = _TONE_BEFORE_VOWEL.sub(r"\2\1", s)
    s = _DUP_COMBINING.sub(r"\1", s)
    return _WS.sub(" ", s).strip()


def normalize_tracking(text: str) -> str:
    # NOTE: BRD §12 keeps tracking-number normalization open; it must not change the agreed
    # duplicate definition (same tracking number + same property + outstanding status).
    return re.sub(r"[\s\-]", "", normalize_common(text)).upper()


def normalize_unit(text: str) -> str:
    # Keep building identifiers and "/" (BRD §7); drop spaces only.
    return normalize_common(text).replace(" ", "").upper()


def normalize_name(text: str) -> str:
    """Exact-match form of a printed name: what is printed, minus formatting noise."""
    return normalize_common(text).casefold()


def normalize_name_for_matching(text: str) -> str:
    """Looser form used only to match against the resident roster (never for OCR accuracy)."""
    s = _LATIN_HONORIFIC.sub("", normalize_name(text))
    for h in _THAI_HONORIFICS:
        if s.startswith(h):
            s = s[len(h) :].lstrip(" .")
            break
    # \w drops Thai combining marks (category Mn), so keep the whole Thai block explicitly.
    return _WS.sub(" ", re.sub(r"[^\w\s฀-๿]", " ", s)).strip()


def normalize_for_eval(field: str, value: str | None) -> str | None:
    if not isinstance(value, str):  # None, NaN / pd.NA from DataFrame joins
        return None
    match field:
        case "tracking_number":
            return normalize_tracking(value)
        case "unit" | "unit_id":
            return normalize_unit(value)
        case "recipient_name":
            return normalize_name(value)
        case "carrier" | "type_size" | "recipient_id":
            return normalize_common(value).casefold()
        case _:
            return normalize_common(value)
