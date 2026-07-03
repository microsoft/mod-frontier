"""Domain calibration: math, packaging, and recorded per-row decisions.

Runs entirely on CPU with no model download. The fixture
(``domain_calibration_cases.json``) is 50 recorded experiment-#11 test-split
rows: each row's eight one-vs-rest domain probabilities, its gpt-5-mini domain
label, and the domain the packaged routing path is expected to pick under both
the calibrated and the raw-argmax settings.
"""
import json
import os

import numpy as np
import pytest

from rewriter.routing_probe import (
    DEFAULT_DOMAIN_CALIBRATION,
    DOMAINS,
    DomainCalibration,
    ProbeScores,
    load_domain_calibration,
    route_scores,
)
from rewriter.routing_probe.calibration import logit

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
CASES = os.path.join(FIXTURES, "domain_calibration_cases.json")


def test_default_calibration_shape_and_domains():
    cal = DEFAULT_DOMAIN_CALIBRATION
    assert cal.k == 8
    assert list(cal.domains) == DOMAINS
    assert cal.temp.shape == (8,) and cal.bias.shape == (8,)
    assert np.all(cal.temp > 0)


def test_default_calibration_loads_from_package_data():
    # the shipped vector is loadable as package data (not hardcoded in .py)
    reloaded = load_domain_calibration()
    np.testing.assert_array_equal(reloaded.temp, DEFAULT_DOMAIN_CALIBRATION.temp)
    np.testing.assert_array_equal(reloaded.bias, DEFAULT_DOMAIN_CALIBRATION.bias)


def test_scaled_logits_matches_formula():
    cal = DEFAULT_DOMAIN_CALIBRATION
    rng = np.random.default_rng(0)
    P = rng.random((4, 8))
    expected = logit(P) / cal.temp + cal.bias
    np.testing.assert_allclose(cal.scaled_logits(P), expected, rtol=0, atol=0)


def test_class_proba_argmax_equals_scaled_logits_argmax():
    # softmax is monotonic per row -> argmax is invariant
    cal = DEFAULT_DOMAIN_CALIBRATION
    rng = np.random.default_rng(1)
    P = rng.random((32, 8))
    np.testing.assert_array_equal(
        cal.class_proba(P).argmax(1), cal.scaled_logits(P).argmax(1)
    )


def test_marginal_proba_is_sigmoid_of_scaled_logits():
    cal = DEFAULT_DOMAIN_CALIBRATION
    rng = np.random.default_rng(2)
    P = rng.random((8, 8))
    zp = cal.scaled_logits(P)
    np.testing.assert_allclose(cal.marginal_proba(P), 1.0 / (1.0 + np.exp(-zp)))


def test_logit_clips_zero_and_one():
    # 0 and 1 must stay finite (heads can saturate)
    out = logit(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(out))


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        DomainCalibration(temp=np.ones(3), bias=np.ones(8), domains=DOMAINS)
    with pytest.raises(ValueError):
        DomainCalibration(temp=np.array([1.0, -1.0]), bias=np.zeros(2),
                          domains=["a", "b"])
    with pytest.raises(ValueError):
        DEFAULT_DOMAIN_CALIBRATION.scaled_logits(np.ones((4, 7)))  # wrong K


def test_roundtrip_to_from_dict():
    cal = DEFAULT_DOMAIN_CALIBRATION
    rebuilt = DomainCalibration.from_dict(cal.to_dict())
    np.testing.assert_array_equal(rebuilt.temp, cal.temp)
    np.testing.assert_array_equal(rebuilt.bias, cal.bias)
    assert rebuilt.domains == cal.domains


def _load_cases():
    blob = json.load(open(CASES))
    assert blob["domains"] == DOMAINS
    return blob["cases"]


def test_route_scores_calibrated_reproduces_recorded_decisions():
    """The shipped calibrated path reproduces the recorded per-row domain."""
    cases = _load_cases()
    dom = np.array([c["domain"] for c in cases], dtype=np.float64)
    scores = ProbeScores(refuse=np.zeros(len(cases)), domain=dom)
    out = route_scores(scores)  # default: DEFAULT_DOMAIN_CALIBRATION
    for c, d in zip(cases, out):
        assert d.domain == c["expected_domain_calibrated"]


def test_route_scores_none_reproduces_raw_argmax():
    """domain_calibration=None recovers the original raw-argmax behavior."""
    cases = _load_cases()
    dom = np.array([c["domain"] for c in cases], dtype=np.float64)
    scores = ProbeScores(refuse=np.zeros(len(cases)), domain=dom)
    out = route_scores(scores, domain_calibration=None)
    for c, d in zip(cases, out):
        assert d.domain == c["expected_domain_raw"]
        # uncalibrated confidence is the raw max OvR probability
        assert d.domain_confidence == pytest.approx(max(c["domain"]))


def test_calibration_actually_reranks_some_rows():
    """The fixture exercises rows where calibration changes the decision."""
    cases = _load_cases()
    changed = [c for c in cases
               if c["expected_domain_raw"] != c["expected_domain_calibrated"]]
    assert changed, "fixture should include rows where calibration re-ranks argmax"
