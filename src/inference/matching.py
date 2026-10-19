"""Unit and recipient matching against the property's resident roster.

Closed-set matching is what makes ≥99% auto-registration plausible: the OCR text only has to be
close enough to pick ONE roster entry with a clear margin. Rules from the BRD:
  * search units only in the current property (BR03); several candidates → review
  * match recipients only within the unit (BR04); similar names or a partial phone number
    alone must not trigger automatic assignment
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from src.text.normalize import normalize_name_for_matching, normalize_unit


@dataclass
class Resident:
    recipient_id: str
    names: list[str]  # every known spelling: Thai, Latin, nickname
    has_account: bool = True


@dataclass
class Unit:
    unit_id: str
    labels: list[str]  # e.g. ["A/1203", "1203"] — building prefix kept when known
    residents: list[Resident] = field(default_factory=list)


@dataclass
class MatchResult:
    status: str  # matched | ambiguous | no_match
    best_id: str | None
    best_score: float
    margin: float
    candidates: list[tuple[str, float]]


def _decide(scored: list[tuple[str, float]], min_score: float, min_margin: float) -> MatchResult:
    scored = sorted(scored, key=lambda x: -x[1])
    above = [s for s in scored if s[1] >= min_score]
    if not above:
        return MatchResult("no_match", None, scored[0][1] if scored else 0.0, 0.0, scored[:5])
    best = above[0]
    second = scored[1][1] if len(scored) > 1 else 0.0
    margin = best[1] - second
    status = "matched" if margin >= min_margin else "ambiguous"
    return MatchResult(
        status, best[0] if status == "matched" else None, best[1], margin, scored[:5]
    )


def match_unit(
    text: str | None, units: list[Unit], min_score: float, min_margin: float
) -> MatchResult:
    if not text:
        return MatchResult("no_match", None, 0.0, 0.0, [])
    q = normalize_unit(text)
    scored = []
    for u in units:
        # Unit numbers are short codes: exact match is required for a full score.
        best = max(
            (100.0 if normalize_unit(lbl) == q else fuzz.ratio(normalize_unit(lbl), q) * 0.8)
            for lbl in u.labels
        )
        scored.append((u.unit_id, best))
    return _decide(scored, min_score, min_margin)


def match_recipient(
    text: str | None, unit: Unit, min_score: float, min_margin: float
) -> MatchResult:
    if not text:
        return MatchResult("no_match", None, 0.0, 0.0, [])
    q = normalize_name_for_matching(text)
    scored = [
        (
            r.recipient_id,
            max(fuzz.token_sort_ratio(normalize_name_for_matching(n), q) for n in r.names),
        )
        for r in unit.residents
    ]
    return _decide(scored, min_score, min_margin)
