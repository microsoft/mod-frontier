"""Train attention-pooling routing heads on labeled activations.

Trains the single-query ``AttnPoolProbe`` head on packed per-token activations
(the backbone is frozen; only the head is trained). One refusal head plus eight
one-vs-rest domain heads, all the same architecture. Class-balanced
cross-entropy, AdamW, seed 42. This is the training recipe that produced the
shipped heads in ``weights/``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .data import DOMAINS, SEED, Split
from .probe import AttnPoolProbe, iter_packed_batches


def _class_weights(y, device) -> torch.Tensor:
    y = np.asarray(y)
    n = len(y)
    w = np.array([n / (2 * max((y == c).sum(), 1)) for c in (0, 1)], dtype=np.float32)
    return torch.tensor(w, device=device)


def train_head(
    tokens_tr: torch.Tensor,
    off_tr: np.ndarray,
    y_tr,
    d_model: int,
    device: str = "cuda",
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 128,
) -> AttnPoolProbe:
    """Train one ``AttnPoolProbe`` head on packed activations + binary labels.

    Returns the trained probe (on CPU, eval mode).
    """
    torch.manual_seed(SEED)
    model = AttnPoolProbe(d_model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=_class_weights(y_tr, device))
    yt = torch.tensor(np.asarray(y_tr), dtype=torch.long)
    n = len(y_tr)
    for ep in range(epochs):
        model.train()
        rng = np.random.default_rng(SEED + ep)
        perm = rng.permutation(np.arange(n))
        for x, mask, bidx in iter_packed_batches(tokens_tr, off_tr, perm, batch_size, device):
            yb = yt[bidx].to(device)
            opt.zero_grad()
            loss = lossf(model(x, mask), yb)
            loss.backward()
            opt.step()
    model.eval()
    model.to("cpu")
    return model


def train_refusal_head(
    tokens_tr: torch.Tensor, off_tr: np.ndarray, train: Split,
    d_model: int, device: str = "cuda", **kw,
) -> AttnPoolProbe:
    """Train the REFUSE-vs-REWRITE head."""
    return train_head(tokens_tr, off_tr, train.refusal, d_model, device=device, **kw)


def train_domain_heads(
    tokens_tr: torch.Tensor, off_tr: np.ndarray, train: Split,
    d_model: int, device: str = "cuda", domains: list[str] = DOMAINS, **kw,
) -> dict[str, AttnPoolProbe]:
    """Train the eight one-vs-rest domain heads. Returns ``{domain: probe}``."""
    out = {}
    for dom in domains:
        y = train.domain_onehot(dom)
        out[dom] = train_head(tokens_tr, off_tr, y, d_model, device=device, **kw)
    return out
