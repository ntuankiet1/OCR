"""Metric definitions that follow BRD §9 and AC01-AC04 literally.

* Per-field accuracy: exact match after the agreed normalization, on READABLE gold values only
  (gold value present). Uncorrected OCR output — what the model produced, not what was saved.
* Auto rate: auto-registered jobs / ALL jobs that entered evaluation. Missing-data, unreadable
  and review cases stay in the denominator.
* Wrong assignment: an auto-registered job whose unit or recipient differs from gold. Any one
  blocks rollout.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest, norm

from src.text.normalize import normalize_for_eval


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def error_rate_upper_bound(errors: int, n: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound. With 0 errors this is 1 - alpha**(1/n) ≈ 3/n
    ("rule of three"): 0 wrong assignments in 5,000 tests → true rate < ~0.06%."""
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, errors + 1, n - errors))


def mcnemar_worse_pvalue(champion_correct: np.ndarray, challenger_correct: np.ndarray) -> float:
    """Exact one-sided McNemar on paired samples. Small p → challenger significantly worse."""
    a, b = np.asarray(champion_correct, bool), np.asarray(challenger_correct, bool)
    lost = int((a & ~b).sum())  # champion right, challenger wrong
    gained = int((~a & b).sum())
    if lost + gained == 0:
        return 1.0
    return float(binomtest(lost, lost + gained, 0.5, alternative="greater").pvalue)


def join_field_predictions(
    pred: pd.DataFrame, gold: pd.DataFrame, fields: list[str]
) -> pd.DataFrame:
    """One row per (image, field) with gold present. Columns: pred_value, score, gold_value,
    readable, correct. A missing prediction counts as wrong when gold is readable."""
    g = gold.loc[gold["field"].isin(fields), ["image_sha256", "field", "value"]]
    p = pred.loc[pred["field"].isin(fields), ["image_sha256", "field", "value", "score"]]
    df = g.rename(columns={"value": "gold_value"}).merge(
        p.rename(columns={"value": "pred_value"}), on=["image_sha256", "field"], how="left"
    )
    df["readable"] = df["gold_value"].notna()
    df["score"] = df["score"].astype(float).fillna(0.0)
    df["correct"] = [
        bool(r and normalize_for_eval(f, pv) == normalize_for_eval(f, gv))
        for f, pv, gv, r in zip(
            df["field"], df["pred_value"], df["gold_value"], df["readable"], strict=True
        )
    ]
    return df


def field_accuracy(joined: pd.DataFrame, alpha: float = 0.05) -> dict[str, dict]:
    out = {}
    for field, part in joined[joined["readable"]].groupby("field"):
        k, n = int(part["correct"].sum()), len(part)
        lo, hi = wilson_interval(k, n, alpha)
        out[str(field)] = {"n": n, "correct": k, "accuracy": k / n, "ci_low": lo, "ci_high": hi}
    return out


def whole_record_accuracy(joined: pd.DataFrame, fields: list[str]) -> float:
    """Share of images where every readable gold field in `fields` is correct."""
    part = joined[joined["readable"] & joined["field"].isin(fields)]
    if part.empty:
        return 0.0
    return float(part.groupby("image_sha256")["correct"].all().mean())


def latency_summary(latency_ms: pd.Series) -> dict[str, float]:
    if latency_ms.empty:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "p50": float(latency_ms.quantile(0.5)),
        "p95": float(latency_ms.quantile(0.95)),
        "max": float(latency_ms.max()),
    }
