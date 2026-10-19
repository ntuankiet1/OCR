"""Automatic-registration decision — BRD §7 "Automatic registration conditions".

Pure rules over calibrated outputs; never learned. Register automatically only when ALL hold:
  automatic mode enabled · exactly one valid unit and recipient · mandatory data complete ·
  carrier and tracking present · type/size classified · no conflicts · duplicate check passed.
Otherwise the job goes to review with every reason listed (BR15).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

MANDATORY = ("recipient_name", "unit", "carrier", "tracking_number", "type_size")


@dataclass
class DecisionInput:
    auto_mode_enabled: bool
    values: dict[str, str | None]
    scores: dict[str, float]
    thresholds: dict[str, float | None]  # None → field never auto-accepted (not calibrated)
    unit_status: str  # matched | ambiguous | no_match
    recipient_status: str
    recipient_has_account: bool
    quality_reasons: list[str] = field(default_factory=list)
    conflict_reasons: list[str] = field(default_factory=list)
    duplicate_check_passed: bool = True  # checked by the registration service (BR09)


@dataclass
class Decision:
    auto: bool
    reasons: list[str]


def decide(d: DecisionInput) -> Decision:
    reasons: list[str] = []
    if not d.auto_mode_enabled:
        reasons.append("auto_mode_disabled")
    reasons += d.quality_reasons
    reasons += d.conflict_reasons
    for f in MANDATORY:
        if not d.values.get(f):
            reasons.append(f"missing_{f}")
            continue
        t = d.thresholds.get(f)
        if t is None or (isinstance(t, float) and math.isinf(t)) or d.scores.get(f, 0.0) < t:
            reasons.append(f"low_confidence_{f}")
    if d.unit_status != "matched":
        reasons.append(f"unit_{d.unit_status}")
    if d.recipient_status != "matched":
        reasons.append(f"recipient_{d.recipient_status}")
    elif not d.recipient_has_account:
        reasons.append("recipient_no_account")  # BR08: officer confirms unit, notify all members
    if not d.duplicate_check_passed:
        reasons.append("duplicate")
    return Decision(auto=not reasons, reasons=reasons)
