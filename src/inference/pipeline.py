"""End-to-end inference over one image, driven entirely by a bundle directory.

quality → barcode → OCR (orientation + det + rec) → rule KIE → roster matching → type/size
→ decision. Every result carries the bundle_id for the audit trail (BR16).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import yaml

from src.calibration.calibrate import apply_isotonic
from src.inference.barcode import read_barcodes
from src.inference.decision import DecisionInput, decide
from src.inference.kie_rules import FieldPrediction, RuleExtractor
from src.inference.matching import MatchResult, Unit, match_recipient, match_unit
from src.inference.ocr import OcrEngine, PaddleOcrEngine
from src.inference.quality import assess
from src.inference.type_size import TypeSizeClassifier
from src.registry.bundle import read_manifest


@dataclass
class JobResult:
    bundle_id: str
    decision: str  # auto | review
    reasons: list[str]
    fields: dict[str, dict]  # field -> {value, score, confidence, polygon, source}
    unit_match: dict | None
    recipient_match: dict | None
    latency_ms: float
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_record(self) -> dict:
        d = asdict(self)
        d["fields"] = json.dumps(d["fields"], ensure_ascii=False)
        return d


class InferencePipeline:
    def __init__(
        self, bundle_dir: Path, ocr: OcrEngine, type_size: TypeSizeClassifier | None, params: dict
    ) -> None:
        self.bundle_dir = bundle_dir
        self.manifest = read_manifest(bundle_dir)
        self.calibration = json.loads((bundle_dir / "calibration.json").read_text())
        rules = yaml.safe_load((bundle_dir / "rules" / "carriers.yaml").read_text())
        self.extractor = RuleExtractor(rules)
        self.ocr = ocr
        self.type_size = type_size
        self.params = params

    @classmethod
    def from_bundle(cls, bundle_dir: Path) -> InferencePipeline:
        meta = json.loads((bundle_dir / "pipeline.json").read_text())
        models = bundle_dir / "models"
        ocr = PaddleOcrEngine(
            models,
            meta["det_name"],
            meta["rec_name"],
            meta["doc_orientation_name"],
            meta["textline_orientation_name"],
        )
        ts_path = models / "type_size" / "model.onnx"
        cal = json.loads((bundle_dir / "calibration.json").read_text())
        ts = (
            TypeSizeClassifier(
                ts_path,
                meta["type_size_image_size"],
                cal.get("fields", {}).get("type_size", {}).get("temperature", 1.0),
            )
            if ts_path.exists()
            else None
        )
        return cls(bundle_dir, ocr, ts, meta["params"])

    def _confidence(self, field_name: str, score: float) -> float:
        curve = self.calibration.get("fields", {}).get(field_name, {}).get("isotonic")
        return float(apply_isotonic(curve, [score])[0]) if curve else score

    def run(
        self,
        image_path: str | Path,
        units: list[Unit] | None = None,
        auto_mode_enabled: bool = True,
        duplicate_check_passed: bool = True,
    ) -> JobResult:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        def lap(name: str, start: float) -> float:
            now = time.perf_counter()
            timings[name] = 1000 * (now - start)
            return now

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot read image {image_path}")
        q = self.params["quality"]
        quality = assess(image, q["blur_laplacian_min"], q["glare_ratio_max"])
        t = lap("quality", t0)
        barcodes = read_barcodes(image)
        t = lap("barcode", t)
        lines = self.ocr.read(image)
        t = lap("ocr", t)
        extraction = self.extractor.extract(lines, barcodes)
        fields: dict[str, FieldPrediction] = dict(extraction.fields)
        t = lap("kie", t)

        if self.type_size is not None:
            cls_id, prob = self.type_size.predict(image)
            fields["type_size"] = FieldPrediction(str(cls_id), prob, source="classifier")
        else:
            fields["type_size"] = FieldPrediction(None, 0.0, source="classifier")
        t = lap("type_size", t)

        m = self.params["matching"]
        unit_res: MatchResult | None = None
        rec_res: MatchResult | None = None
        has_account = False
        if units is not None:
            unit_res = match_unit(fields["unit"].value, units, m["min_score"], m["min_margin"])
            if unit_res.status == "matched":
                unit = next(u for u in units if u.unit_id == unit_res.best_id)
                rec_res = match_recipient(
                    fields["recipient_name"].value, unit, m["min_score"], m["min_margin"]
                )
                if rec_res.status == "matched":
                    has_account = next(
                        r for r in unit.residents if r.recipient_id == rec_res.best_id
                    ).has_account
                    fields["recipient_id"] = FieldPrediction(
                        rec_res.best_id, rec_res.best_score / 100, source="match"
                    )
                fields["unit_id"] = FieldPrediction(
                    unit_res.best_id, unit_res.best_score / 100, source="match"
                )
        lap("matching", t)

        thresholds = self.manifest.thresholds
        decision = decide(
            DecisionInput(
                auto_mode_enabled=auto_mode_enabled and units is not None,
                values={k: v.value for k, v in fields.items()},
                scores={k: v.score for k, v in fields.items()},
                thresholds=thresholds,
                unit_status=unit_res.status if unit_res else "no_match",
                recipient_status=rec_res.status if rec_res else "no_match",
                recipient_has_account=has_account,
                quality_reasons=quality.reasons,
                conflict_reasons=extraction.reasons,
                duplicate_check_passed=duplicate_check_passed,
            )
        )
        return JobResult(
            bundle_id=self.manifest.bundle_id,
            decision="auto" if decision.auto else "review",
            reasons=decision.reasons,
            fields={
                k: {**asdict(v), "confidence": self._confidence(k, v.score)}
                for k, v in fields.items()
            },
            unit_match=asdict(unit_res) if unit_res else None,
            recipient_match=asdict(rec_res) if rec_res else None,
            latency_ms=1000 * (time.perf_counter() - t0),
            timings_ms=timings,
        )
