"""Champion/challenger gate. A challenger is registered only if every check passes.

Both evaluations must come from the same frozen regression set version; otherwise the champion
has to be re-evaluated first (`parcel-ocr evaluate-champion`).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from src.evaluation.metrics import mcnemar_worse_pvalue


class GateConfig(BaseModel):
    max_wrong_assignments: int = 0
    max_field_drop_pp: float = 0.2
    mcnemar_alpha: float = 0.05
    subgroup_min_n: int = 100
    max_subgroup_drop_pp: float = 1.0
    max_auto_rate_drop_pp: float = 0.2
    max_latency_p95_ms: float = 1500
    min_field_accuracy: dict[str, float] = Field(default_factory=dict)


class GateCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class GateReport(BaseModel):
    passed: bool
    mode: str  # "vs_champion" | "first_model"
    challenger_bundles: list[str]
    champion_bundles: list[str]
    checks: list[GateCheck]
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class EvalRun(BaseModel):
    metrics: dict
    correctness: pd.DataFrame

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def load(cls, eval_dir: Path) -> EvalRun | None:
        metrics_path = eval_dir / "metrics.json"
        if not metrics_path.exists():
            return None
        return cls(
            metrics=json.loads(metrics_path.read_text()),
            correctness=pd.read_parquet(eval_dir / "correctness.parquet"),
        )


def _pp(x: float) -> str:
    return f"{100 * x:.2f}%"


def run_gate(challenger: EvalRun, champion: EvalRun | None, cfg: GateConfig) -> GateReport:
    cm = challenger.metrics
    checks: list[GateCheck] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append(GateCheck(name=name, passed=bool(passed), detail=detail))

    # --- absolute checks (always)
    check(
        "wrong_assignments",
        cm["wrong_assignments"] <= cfg.max_wrong_assignments,
        f"{cm['wrong_assignments']} wrong (max {cfg.max_wrong_assignments}), "
        f"95% UB {_pp(cm['wrong_assignment_rate_ub95'])} over {cm['n_auto']} auto jobs",
    )
    check(
        "latency_p95",
        cm["latency_ms"]["p95"] <= cfg.max_latency_p95_ms,
        f"p95 {cm['latency_ms']['p95']:.0f} ms (max {cfg.max_latency_p95_ms:.0f})",
    )
    for field, floor in cfg.min_field_accuracy.items():
        acc = cm["fields"].get(field, {}).get("accuracy", 0.0)
        check(f"floor:{field}", acc >= floor, f"{_pp(acc)} (floor {_pp(floor)})")

    if champion is None:
        return GateReport(
            passed=all(c.passed for c in checks),
            mode="first_model",
            challenger_bundles=cm.get("bundle_ids", []),
            champion_bundles=[],
            checks=checks,
        )

    pm = champion.metrics
    same_set = cm.get("eval_set_hash") == pm.get("eval_set_hash") and cm.get("eval_set_hash")
    check(
        "same_eval_set",
        same_set,
        f"challenger {cm.get('eval_set_hash')} vs champion {pm.get('eval_set_hash')}",
    )

    # --- per-field: absolute drop and paired significance
    paired = challenger.correctness.merge(
        champion.correctness[["image_sha256", "field", "correct"]],
        on=["image_sha256", "field"],
        suffixes=("", "_champion"),
    )
    paired = paired[paired["readable"]]
    for field, part in paired.groupby("field"):
        acc_new, acc_old = part["correct"].mean(), part["correct_champion"].mean()
        drop_pp = 100 * (acc_old - acc_new)
        p = mcnemar_worse_pvalue(part["correct_champion"].to_numpy(), part["correct"].to_numpy())
        check(
            f"field:{field}",
            drop_pp <= cfg.max_field_drop_pp and p >= cfg.mcnemar_alpha,
            f"{_pp(acc_old)} → {_pp(acc_new)} "
            f"(drop {drop_pp:.2f}pp, McNemar p={p:.3f}, n={len(part)})",
        )

    # --- subgroups: no carrier / capture mode / property may regress materially
    for col in ("carrier", "capture_mode", "property_id"):
        if col not in paired:
            continue
        for value, part in paired.groupby(col):
            if len(part) < cfg.subgroup_min_n:
                continue
            drop_pp = 100 * (part["correct_champion"].mean() - part["correct"].mean())
            check(
                f"subgroup:{col}={value}",
                drop_pp <= cfg.max_subgroup_drop_pp,
                f"drop {drop_pp:.2f}pp over {len(part)} field samples",
            )

    auto_drop_pp = 100 * (pm["auto_rate"] - cm["auto_rate"])
    check(
        "auto_rate",
        auto_drop_pp <= cfg.max_auto_rate_drop_pp,
        f"{_pp(pm['auto_rate'])} → {_pp(cm['auto_rate'])} (drop {auto_drop_pp:.2f}pp)",
    )

    return GateReport(
        passed=all(c.passed for c in checks),
        mode="vs_champion",
        challenger_bundles=cm.get("bundle_ids", []),
        champion_bundles=pm.get("bundle_ids", []),
        checks=checks,
    )
