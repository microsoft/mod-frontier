"""Calibrated routing: probe scores -> domain + REFUSE/REWRITE decision.

This is the routing layer validated end-to-end on real MBO traffic in SafeFlow
experiment #5. Given the nine attention-pooling heads (one refusal + eight
one-vs-rest domain), routing a prompt is:

    domain   = argmax over the (optionally calibrated) 8 domain head scores
    decision = REFUSE if refuse_prob >= threshold else REWRITE

The refusal threshold has two documented operating points:

    DEFAULT_THRESHOLD  = 0.161   recall-favoring, the experiment-#5 default
    F1_OPTIMAL_THRESHOLD = 0.370 F1-optimal alternate (in-distribution)

Domain calibration. By default the domain argmax is taken over a per-head
temperature + bias calibration of the eight one-vs-rest logits
(:data:`~rewriter.routing_probe.calibration.DEFAULT_DOMAIN_CALIBRATION`, fit in
SafeFlow experiment #11), which raises held-out domain macro-F1 from 0.8508 to
0.8721. Pass ``domain_calibration=None`` to recover the original raw-argmax
behavior. The refusal threshold is unaffected -- calibration only re-ranks the
domain heads.

Note: the routing decision is strictly binary (REFUSE / REWRITE). There is no
third "ALLOW" state in the SafeFlow routing logic; a downstream allow-band would
require an additional threshold and is not part of this validated method.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from .calibration import DEFAULT_DOMAIN_CALIBRATION, DomainCalibration
from .data import DOMAINS
from .probe import D_MODEL, AttnPoolProbe, load_probe

DEFAULT_THRESHOLD = 0.161
F1_OPTIMAL_THRESHOLD = 0.370
DEFAULT_LAYER = 18

REFUSE = "REFUSE"
REWRITE = "REWRITE"


@dataclass
class RoutingDecision:
    """One routed prompt."""

    domain: str                 # argmax domain (calibrated by default)
    decision: str               # "REFUSE" or "REWRITE"
    domain_confidence: float    # winning head's score: calibrated marginal, or
                                # raw max OvR probability when uncalibrated
    refuse_probability: float   # refusal-head positive probability


@dataclass
class ProbeScores:
    """Raw per-prompt probe outputs (model already run)."""

    refuse: np.ndarray   # [N] refusal positive-class probability
    domain: np.ndarray   # [N, 8] one-vs-rest domain probabilities (DOMAINS order)


def route_scores(
    scores: ProbeScores,
    threshold: float = DEFAULT_THRESHOLD,
    domains: list[str] = DOMAINS,
    domain_calibration: DomainCalibration | None = DEFAULT_DOMAIN_CALIBRATION,
) -> list[RoutingDecision]:
    """Map raw probe scores to routing decisions (re-thresholdable, no model).

    Args:
        scores: per-prompt refusal + domain head probabilities.
        threshold: refusal operating point; REFUSE iff ``refuse >= threshold``.
        domains: domain names, indexing ``scores.domain`` columns.
        domain_calibration: per-head temperature + bias applied to the domain
            logits before the argmax. Defaults to the shipped experiment-#11
            vector (:data:`~rewriter.routing_probe.calibration.DEFAULT_DOMAIN_CALIBRATION`);
            pass ``None`` for the original raw-argmax behavior. When calibrated,
            ``domain_confidence`` is the winning head's calibrated marginal;
            uncalibrated, it is the raw max OvR probability.
    """
    if domain_calibration is None:
        arg = scores.domain.argmax(1)
        dom_conf = scores.domain.max(1)
    else:
        if list(domain_calibration.domains) != list(domains):
            raise ValueError(
                "domain_calibration.domains does not match the routing domains "
                f"({list(domain_calibration.domains)} vs {list(domains)})"
            )
        zprime = domain_calibration.scaled_logits(scores.domain)  # [N, K]
        arg = zprime.argmax(1)
        dom_conf = 1.0 / (1.0 + np.exp(-zprime[np.arange(len(arg)), arg]))
    out = []
    for i in range(len(scores.refuse)):
        out.append(RoutingDecision(
            domain=domains[arg[i]],
            decision=REFUSE if scores.refuse[i] >= threshold else REWRITE,
            domain_confidence=float(dom_conf[i]),
            refuse_probability=float(scores.refuse[i]),
        ))
    return out


class Router:
    """Loads the nine routing heads and scores / routes prompts.

    Args:
        weights_dir: directory containing ``L{layer}_attn_pool_{refusal,domain_*}.pt``.
            Defaults to the weights bundled with this package.
        layer: transformer block the probes were trained on (18).
        d_model: residual width (2560 for Qwen3-4B-Instruct-2507).
        device: device for probe forwards.
    """

    def __init__(
        self,
        weights_dir: str | None = None,
        layer: int = DEFAULT_LAYER,
        d_model: int = D_MODEL,
        device: str = "cuda",
        domains: list[str] = DOMAINS,
    ):
        if weights_dir is None:
            weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.weights_dir = weights_dir
        self.layer = layer
        self.d_model = d_model
        self.device = device
        self.domains = domains
        self.refusal_probe: AttnPoolProbe = self._load("refusal")
        self.domain_probes: dict[str, AttnPoolProbe] = {
            d: self._load(f"domain_{d}") for d in domains
        }

    def _load(self, name: str) -> AttnPoolProbe:
        path = os.path.join(self.weights_dir, f"L{self.layer}_attn_pool_{name}.pt")
        return load_probe(path, d_model=self.d_model, map_location="cpu")

    def score_packed(
        self, tokens: torch.Tensor, offsets: np.ndarray, batch_size: int = 256,
    ) -> ProbeScores:
        """Score pre-extracted packed activations with all nine heads."""
        refuse = self.refusal_probe.positive_proba(
            tokens, offsets, device=self.device, batch_size=batch_size)
        dom = np.stack(
            [self.domain_probes[d].positive_proba(
                tokens, offsets, device=self.device, batch_size=batch_size)
             for d in self.domains],
            axis=1,
        )
        return ProbeScores(refuse=refuse, domain=dom)

    def score_prompts(self, prompts: list[str], extractor, batch_size: int = 16) -> ProbeScores:
        """Extract activations with ``extractor`` then score all nine heads.

        ``extractor`` is an ``ActivationExtractor`` (imported lazily by the caller
        so that scoring pre-extracted activations needs no transformers/CUDA).
        """
        acts = extractor.extract(prompts, layer=self.layer, batch_size=batch_size)
        return self.score_packed(*acts.packed)

    def route(
        self, prompts: list[str], extractor, threshold: float = DEFAULT_THRESHOLD,
        batch_size: int = 16,
        domain_calibration: DomainCalibration | None = DEFAULT_DOMAIN_CALIBRATION,
    ) -> list[RoutingDecision]:
        """End-to-end: prompts -> activations -> probe scores -> decisions.

        ``domain_calibration`` is forwarded to :func:`route_scores` (defaults to
        the shipped experiment-#11 vector; pass ``None`` for raw argmax).
        """
        scores = self.score_prompts(prompts, extractor, batch_size=batch_size)
        return route_scores(scores, threshold=threshold, domains=self.domains,
                            domain_calibration=domain_calibration)
