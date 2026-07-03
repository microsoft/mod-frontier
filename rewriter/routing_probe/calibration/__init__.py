"""Per-head domain calibration for the eight one-vs-rest routing heads.

The eight domain heads are independent one-vs-rest probes; the raw routing
decision is ``argmax`` over their eight positive-class probabilities. SafeFlow
experiment #11 found that a per-head temperature + bias transform of each head's
logit -- fit jointly to maximize the calibrated softmax likelihood against
gpt-5-mini domain labels -- re-ranks the heads enough to raise domain macro-F1
from 0.8508 (raw argmax) to 0.8721 (+0.021, 95% CI [0.0056, 0.0367]) on a
held-out split.

The transform, per prompt, for head ``c`` with raw probability ``p_c``:

    z_c    = logit(p_c)               # invert the head's sigmoid
    z'_c   = z_c / temp_c + bias_c    # per-head affine calibration
    domain = argmax_c z'_c            # re-ranked routing decision

A sigmoid is monotonic, so applying the calibration and then taking ``argmax``
over the softmax of ``z'`` is identical to taking ``argmax`` over ``z'``
directly; this module works in the ``z'`` (scaled-logit) space. The winning
head's calibrated marginal ``sigmoid(z'_winner)`` is what the router reports as
``domain_confidence`` on the calibrated path.

The fitted vector for the shipped L18 heads lives in ``domain_L18.json`` and is
exposed as :data:`DEFAULT_DOMAIN_CALIBRATION`.

Important: the gain is in cross-head *routing* (which head wins the argmax), not
in per-head marginal calibration -- per-head ECE is flat-to-slightly-worse after
the transform, because temp+bias is fit to the joint softmax, not to each head's
reliability. Use this to route, not to read a calibrated per-domain probability.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "domain_L18.json")
_LOGIT_EPS = 1e-6  # clip before logit so p in {0, 1} stays finite (matches issue-11)


def logit(p: np.ndarray, eps: float = _LOGIT_EPS) -> np.ndarray:
    """Inverse sigmoid with clipping so 0/1 probabilities stay finite."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


@dataclass(frozen=True)
class DomainCalibration:
    """A fitted per-head temperature + bias calibration for the domain heads.

    Attributes:
        temp: ``[K]`` per-head temperatures (positive); divides each head's logit.
        bias: ``[K]`` per-head additive biases on the scaled logit.
        domains: the ``K`` domain names, in the column order ``temp`` / ``bias``
            and the scored probability matrix are indexed by.
        layer: the transformer block the heads were trained on (provenance only).
    """

    temp: np.ndarray
    bias: np.ndarray
    domains: tuple[str, ...]
    layer: int = 18

    def __post_init__(self) -> None:
        temp = np.asarray(self.temp, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        object.__setattr__(self, "temp", temp)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "domains", tuple(self.domains))
        if not (temp.shape == bias.shape == (len(self.domains),)):
            raise ValueError(
                f"temp {temp.shape} / bias {bias.shape} must both be 1-D of length "
                f"{len(self.domains)} (number of domains)"
            )
        if np.any(temp <= 0):
            raise ValueError(f"temperatures must be positive, got {temp.tolist()}")

    @property
    def k(self) -> int:
        """Number of domain heads."""
        return len(self.domains)

    def _check_shape(self, domain_probs: np.ndarray) -> np.ndarray:
        P = np.asarray(domain_probs, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != self.k:
            raise ValueError(
                f"domain_probs must be [N, {self.k}] (one column per domain), got {P.shape}"
            )
        return P

    def scaled_logits(self, domain_probs: np.ndarray) -> np.ndarray:
        """Per-head calibrated logits ``z' = logit(p)/temp + bias``, shape ``[N, K]``.

        ``argmax`` over this is the calibrated routing decision.
        """
        z = logit(self._check_shape(domain_probs))
        return z / self.temp + self.bias

    def marginal_proba(self, domain_probs: np.ndarray) -> np.ndarray:
        """Per-head calibrated marginal ``sigmoid(z')``, shape ``[N, K]``.

        The winning head's value is the calibrated ``domain_confidence``. Note
        this is a per-head marginal, not a softmax over domains.
        """
        zp = self.scaled_logits(domain_probs)
        return 1.0 / (1.0 + np.exp(-zp))

    def class_proba(self, domain_probs: np.ndarray) -> np.ndarray:
        """Calibrated multiclass distribution ``softmax(z')``, shape ``[N, K]``.

        ``argmax`` of this matches :meth:`scaled_logits` (softmax is monotonic);
        provided for callers that want a normalized distribution over domains.
        """
        zp = self.scaled_logits(domain_probs)
        zp = zp - zp.max(axis=1, keepdims=True)
        e = np.exp(zp)
        return e / e.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict:
        """Serialize to the ``domain_L{layer}.json`` schema."""
        return {
            "layer": self.layer,
            "domains": list(self.domains),
            "temp": self.temp.tolist(),
            "bias": self.bias.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DomainCalibration":
        """Build from a parsed ``domain_L{layer}.json`` mapping."""
        return cls(
            temp=np.asarray(d["temp"], dtype=np.float64),
            bias=np.asarray(d["bias"], dtype=np.float64),
            domains=tuple(d["domains"]),
            layer=int(d.get("layer", 18)),
        )


def load_domain_calibration(path: str = _DEFAULT_PATH) -> DomainCalibration:
    """Load a :class:`DomainCalibration` from a ``domain_L{layer}.json`` file."""
    with open(path) as f:
        return DomainCalibration.from_dict(json.load(f))


#: The fitted calibration for the bundled L18 domain heads (SafeFlow issue-11).
DEFAULT_DOMAIN_CALIBRATION = load_domain_calibration()

__all__ = [
    "DomainCalibration",
    "load_domain_calibration",
    "DEFAULT_DOMAIN_CALIBRATION",
    "logit",
]
