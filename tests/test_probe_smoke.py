"""Smoke test: clean import, load bundled weights, score + route committed acts.

Runs entirely on CPU with no model download: it scores a small committed
activation fixture (per-token L18 residual for a few benign prompts) through the
nine bundled probe heads and checks the outputs reproduce the frozen values and
yield valid routing decisions. This is the test that must pass on a bare
off-cluster checkout.
"""
import json
import os

import numpy as np
import pytest
import torch

from rewriter.routing_probe import DOMAINS, Router, route_scores

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
ACTS = os.path.join(FIXTURES, "activations.pt")
EXPECTED = os.path.join(FIXTURES, "expected_scores.json")


def test_import():
    import rewriter.routing_probe as srp

    assert srp.__version__
    assert len(srp.DOMAINS) == 8


def test_router_loads_bundled_weights():
    router = Router(device="cpu")
    assert router.refusal_probe is not None
    assert set(router.domain_probes) == set(DOMAINS)


@pytest.mark.skipif(not os.path.exists(ACTS), reason="activation fixture missing")
def test_score_and_route_committed_activations():
    # weights_only=True with NO allowlist: the fixture stores plain tensors
    # (offsets as a LongTensor, not an ndarray), so the safe load works on
    # the pinned requirement floors (torch>=2.3, numpy>=1.26) — the
    # allowlist APIs (torch.serialization.safe_globals, np._core paths) do
    # not exist there (round-3 review finding #14).
    blob = torch.load(ACTS, map_location="cpu", weights_only=True)
    tokens, offsets = blob["tokens"], np.asarray(blob["offsets"])
    n = len(offsets) - 1

    router = Router(device="cpu")
    scores = router.score_packed(tokens, offsets, batch_size=8)

    assert scores.refuse.shape == (n,)
    assert scores.domain.shape == (n, 8)
    assert np.all((scores.refuse >= 0) & (scores.refuse <= 1))

    decisions = route_scores(scores)
    assert len(decisions) == n
    for d in decisions:
        assert d.decision in ("REFUSE", "REWRITE")
        assert d.domain in DOMAINS

    if os.path.exists(EXPECTED):
        exp = json.load(open(EXPECTED))
        np.testing.assert_allclose(scores.refuse, exp["refuse"], atol=2e-3)
        np.testing.assert_allclose(scores.domain, exp["domain"], atol=2e-3)
