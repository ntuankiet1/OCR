"""Calibration stage: provisional-bundle predictions on the calibration split → calibration.json.

OCR fields  : isotonic curve (raw score → P(correct)) + auto-accept threshold on raw score
type_size   : temperature from classifier logits + threshold on calibrated max probability
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.calibrate import fit_isotonic, fit_temperature, select_threshold
from src.evaluation.metrics import join_field_predictions
from src.inference.type_size import softmax


def calibrate(
    pred_dir: Path, snapshot_dir: Path, type_size_logits: Path | None, params: dict, out_file: Path
) -> dict:
    labels = pd.read_parquet(snapshot_dir / "labels.parquet")
    gold = labels[labels["split"] == "calibration"]
    pred = pd.read_parquet(pred_dir / "fields.parquet")
    target, alpha, min_n = params["target_precision"], params["alpha"], params["min_accepted"]

    out: dict = {"target_precision": target, "alpha": alpha, "fields": {}}
    ocr_fields = [f for f in params["fields"] if f != "type_size"]
    joined = join_field_predictions(pred, gold, ocr_fields)
    for field in ocr_fields:
        part = joined[(joined["field"] == field) & joined["readable"]]
        res = select_threshold(
            part["score"].to_numpy(), part["correct"].to_numpy(), target, alpha, min_n
        )
        out["fields"][field] = {
            **res.to_dict(),
            "isotonic": fit_isotonic(part["score"], part["correct"]) if len(part) >= 2 else None,
        }

    if "type_size" in params["fields"] and type_size_logits and type_size_logits.exists():
        npz = np.load(type_size_logits)
        logits, y = npz["logits"], npz["labels"]
        temperature = fit_temperature(logits, y) if len(y) else 1.0
        probs = softmax(logits / temperature) if len(y) else np.zeros((0, 10))
        res = select_threshold(probs.max(axis=1), probs.argmax(axis=1) == y, target, alpha, min_n)
        out["fields"]["type_size"] = {**res.to_dict(), "temperature": temperature}

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2))
    return out


def empty_calibration(out_file: Path, fields: list[str]) -> dict:
    """Provisional calibration: no thresholds → every field routes to review."""
    out = {"provisional": True, "fields": {f: {"threshold": None} for f in fields}}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2))
    return out


def threshold_summary(calibration: dict) -> dict[str, str]:
    return {
        f: (
            "never auto"
            if c.get("threshold") is None or math.isinf(c["threshold"])
            else f"≥{c['threshold']:.4f} (coverage {100 * c.get('coverage', 0):.1f}%)"
        )
        for f, c in calibration.get("fields", {}).items()
    }
