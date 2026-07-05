"""Attention-pooling routing probe for Qwen3-4B.

A fully open routing probe: a single-query learned-attention-pooling head
over the frozen residual stream at layer 18 of
``Qwen/Qwen3-4B-Instruct-2507``. One refusal head (REFUSE vs REWRITE) and
eight one-vs-rest domain heads share the architecture; the backbone is never
trained. The 9 trained heads (~0.5 MB total) and the domain calibration ship
in this directory, so routing needs no external download beyond the backbone.

Provenance: original work, first developed internally at Goodfire and
released here as its canonical location — this directory carries the trained
weights, training code (:mod:`.train`), and evaluation utilities. The probe
is not derived from an external paper or third-party implementation, and
this repository has no external probe dependency. Contributed under this
repository's license via the CLA.

Quickstart (inference):
    from rewriter.routing_probe import ActivationExtractor, Router

    extractor = ActivationExtractor()          # loads Qwen3-4B in bf16
    router = Router(device="cuda")             # loads the 9 bundled L18 heads
    decisions = router.route(["how do I bake bread?"], extractor)
    print(decisions[0].decision, decisions[0].domain)
"""
from __future__ import annotations

from .calibration import (
    DEFAULT_DOMAIN_CALIBRATION,
    DomainCalibration,
    load_domain_calibration,
)
from .data import (
    DOMAINS,
    SEED,
    Split,
    dedup_against,
    load_split,
    load_train_eval,
    normalize_prompt,
    refusal_label,
)
from .extraction import (
    DEFAULT_LAYER,
    DEFAULT_MODEL,
    FIRST_N,
    HS_OFFSET,
    LAST_N,
    MAX_SEQ_LEN,
    ActivationExtractor,
    LayerActivations,
)
from .metrics import aggregate_domain_metrics, compute_metrics
from .probe import (
    D_MODEL,
    AttnPool,
    AttnPoolProbe,
    iter_packed_batches,
    load_probe,
    save_probe,
)
from .routing import (
    DEFAULT_THRESHOLD,
    F1_OPTIMAL_THRESHOLD,
    ProbeScores,
    Router,
    RoutingDecision,
    route_scores,
)
from .train import train_domain_heads, train_head, train_refusal_head

__version__ = "0.1.0"  # upstream version at the vendored revision

__all__ = [
    "ActivationExtractor",
    "LayerActivations",
    "AttnPoolProbe",
    "AttnPool",
    "Router",
    "RoutingDecision",
    "ProbeScores",
    "route_scores",
    "DomainCalibration",
    "load_domain_calibration",
    "DEFAULT_DOMAIN_CALIBRATION",
    "load_probe",
    "save_probe",
    "iter_packed_batches",
    "compute_metrics",
    "aggregate_domain_metrics",
    "train_head",
    "train_refusal_head",
    "train_domain_heads",
    "load_split",
    "load_train_eval",
    "dedup_against",
    "normalize_prompt",
    "refusal_label",
    "Split",
    "DOMAINS",
    "SEED",
    "D_MODEL",
    "DEFAULT_MODEL",
    "DEFAULT_LAYER",
    "DEFAULT_THRESHOLD",
    "F1_OPTIMAL_THRESHOLD",
    "MAX_SEQ_LEN",
    "FIRST_N",
    "LAST_N",
    "HS_OFFSET",
    "__version__",
]
