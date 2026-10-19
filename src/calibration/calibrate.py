"""Confidence calibration and auto-accept threshold selection.

BR05: "Validate confidence thresholds using real tests; confidence 0.99 does not establish 99%
accuracy." So a threshold is chosen from labelled calibration data: the lowest raw score whose
accepted subset still has a Clopper-Pearson LOWER bound >= target precision. Scanning many
thresholds is optimistic (multiple comparisons); the frozen regression set is the honest check.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression


def clopper_pearson_lower(k: int | np.ndarray, n: int | np.ndarray, alpha: float) -> np.ndarray:
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore"):
        lo = beta.ppf(alpha / 2, k, n - k + 1)
    return np.where(k <= 0, 0.0, np.nan_to_num(lo, nan=0.0))


@dataclass
class ThresholdResult:
    threshold: float  # math.inf → never auto-accept this field
    coverage: float  # share of calibration samples auto-accepted
    precision: float
    precision_lcb: float
    n_accepted: int
    n_total: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["threshold"] = None if math.isinf(self.threshold) else self.threshold
        return d


def select_threshold(
    scores: np.ndarray,
    correct: np.ndarray,
    target_precision: float,
    alpha: float = 0.05,
    min_accepted: int = 1,
) -> ThresholdResult:
    scores, correct = np.asarray(scores, float), np.asarray(correct, bool)
    n = len(scores)
    if n == 0:
        return ThresholdResult(math.inf, 0.0, 0.0, 0.0, 0, 0)
    order = np.argsort(-scores, kind="stable")
    s, c = scores[order], correct[order]
    k_cum = np.cumsum(c)
    n_cum = np.arange(1, n + 1)
    # A threshold t accepts every sample with score >= t, so only cut after the last tie.
    last_of_tie = np.r_[s[1:] != s[:-1], True]
    lcb = clopper_pearson_lower(k_cum, n_cum, alpha)
    ok = last_of_tie & (lcb >= target_precision) & (n_cum >= min_accepted)
    if not ok.any():
        return ThresholdResult(math.inf, 0.0, 0.0, 0.0, 0, n)
    i = int(np.flatnonzero(ok)[-1])  # largest accepted set that still meets the bound
    return ThresholdResult(
        threshold=float(s[i]),
        coverage=float(n_cum[i] / n),
        precision=float(k_cum[i] / n_cum[i]),
        precision_lcb=float(lcb[i]),
        n_accepted=int(n_cum[i]),
        n_total=n,
    )


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling for the type/size classifier (minimise NLL on calibration data)."""
    logits, labels = np.asarray(logits, float), np.asarray(labels, int)

    def nll(log_t: float) -> float:
        lp = log_softmax(logits / math.exp(log_t), axis=1)
        return float(-lp[np.arange(len(labels)), labels].mean())

    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(math.exp(res.x))


def fit_isotonic(scores: np.ndarray, correct: np.ndarray) -> dict[str, list[float]]:
    """Map a raw OCR score to P(correct). Stored as breakpoints (JSON), not a pickle."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(scores, float), np.asarray(correct, float))
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def apply_isotonic(curve: dict[str, list[float]], scores: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(scores, float), curve["x"], curve["y"])
