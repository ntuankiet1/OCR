import math

import numpy as np
import pytest

from src.calibration.calibrate import (
    apply_isotonic,
    fit_isotonic,
    fit_temperature,
    select_threshold,
)
from src.evaluation.metrics import error_rate_upper_bound, mcnemar_worse_pvalue, wilson_interval


def test_threshold_meets_lower_bound_not_point_estimate():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 5000)
    correct = rng.uniform(0, 1, 5000) < scores  # P(correct) == score
    res = select_threshold(scores, correct, target_precision=0.95, alpha=0.05)
    assert res.precision_lcb >= 0.95
    assert res.precision >= res.precision_lcb
    assert 0 < res.coverage < 0.2  # only the top of the score range is that reliable


def test_unreachable_target_means_never_auto():
    res = select_threshold(np.full(100, 0.99), np.r_[np.ones(90), np.zeros(10)].astype(bool), 0.995)
    assert math.isinf(res.threshold) and res.to_dict()["threshold"] is None


def test_min_accepted_blocks_tiny_samples():
    res = select_threshold(np.array([0.9, 0.8]), np.array([True, True]), 0.5, min_accepted=10)
    assert math.isinf(res.threshold)


def test_ties_are_never_split():
    res = select_threshold(
        np.array([0.9, 0.9, 0.9, 0.1]), np.array([True, True, False, False]), 0.0
    )
    assert res.n_accepted in (3, 4)


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, (4000, 10))  # calibrated logits: labels are sampled from softmax(z)
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(10, p=row) for row in p])
    t = fit_temperature(z * 3.0, y)  # model 3x overconfident
    assert t == pytest.approx(3.0, rel=0.15)


def test_isotonic_is_monotone_json():
    curve = fit_isotonic(np.array([0.1, 0.5, 0.9, 0.95]), np.array([0, 0, 1, 1]))
    out = apply_isotonic(curve, np.array([0.0, 0.6, 1.0]))
    assert list(out) == sorted(out) and set(curve) == {"x", "y"}


def test_rule_of_three():
    assert error_rate_upper_bound(0, 5000) == pytest.approx(3 / 5000, rel=0.01)


def test_wilson_contains_point():
    lo, hi = wilson_interval(990, 1000)
    assert lo < 0.99 < hi


def test_mcnemar_detects_worse_challenger():
    champ = np.ones(1000, bool)
    chal = champ.copy()
    chal[:30] = False
    assert mcnemar_worse_pvalue(champ, chal) < 0.001
    assert mcnemar_worse_pvalue(champ, champ) == 1.0
