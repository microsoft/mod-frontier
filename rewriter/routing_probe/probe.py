"""The attention-pooling routing probe head.

``AttnPoolProbe`` is a single-query learned-attention-pooling head over a frozen
transformer's residual stream. It is the architecture selected by SafeFlow
experiment #4 from a 12-architecture x 5-layer sweep, applied to the residual
stream at layer 18 of ``Qwen/Qwen3-4B-Instruct-2507``.

Architecture (one shared design serves every routing head):

    h      = LayerNorm(x)                       # [B, T, D]
    scores = (h @ query) * D ** -0.5            # [B, T]   single learned query
    scores = scores.masked_fill(~mask, -inf)
    attn   = softmax(scores, dim=tokens)        # [B, T]
    pooled = sum(attn * h, dim=tokens)          # [B, D]
    logits = Linear(D -> 2)(pooled)             # [B, 2]   binary head

The model under it is **never fine-tuned**; only this head is trained. A refusal
head (REFUSE vs REWRITE) and eight one-vs-rest domain heads all use this same
class with different trained weights.

Note on provenance: this is the *single-query* pooling head. An 8-head
multi-head variant was evaluated in experiment #4 and lost (macro-F1 0.907 vs
0.929), so it is intentionally not the shipped architecture. The saved
state_dict keys are exactly ``query``, ``norm.weight``, ``norm.bias``,
``head.weight``, ``head.bias`` and are preserved here for load compatibility
with the experiment-#4 weights shipped in ``weights/``.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
import torch.nn as nn

D_MODEL = 2560  # Qwen3-4B-Instruct-2507 hidden size


class AttnPoolProbe(nn.Module):
    """Single-query learned attention pooling -> binary linear head.

    Args:
        d_model: residual-stream width (2560 for Qwen3-4B-Instruct-2507).
    """

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.head = nn.Linear(d_model, 2)
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, T, D] residual-stream activations.
            mask: [B, T] bool, True for valid tokens.

        Returns:
            [B, 2] class logits.
        """
        h = self.norm(x)
        scores = (h @ self.query) * self.scale  # [B, T]
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        pooled = (attn * h).sum(1)  # [B, D]
        return self.head(pooled)

    @torch.no_grad()
    def positive_proba(
        self,
        tokens: torch.Tensor,
        offsets: np.ndarray,
        device: str = "cuda",
        batch_size: int = 256,
    ) -> np.ndarray:
        """Positive-class (class 1) probability for each packed sequence.

        Args:
            tokens: [T_total, D] fp16/fp32 per-token activations, sequences concatenated.
            offsets: [N+1] int, sequence ``i`` is ``tokens[offsets[i]:offsets[i+1]]``.
            device: device to run the forward on.
            batch_size: padded batch size.

        Returns:
            [N] float32 positive-class probabilities.
        """
        self.eval()
        self.to(device)
        n = len(offsets) - 1
        out = np.zeros(n, dtype=np.float32)
        for x, mask, bidx in iter_packed_batches(
            tokens, offsets, np.arange(n), batch_size, device
        ):
            out[bidx] = torch.softmax(self(x, mask), -1)[:, 1].cpu().numpy()
        return out


# Backwards-compatible alias for the experiment-#4 class name.
AttnPool = AttnPoolProbe


def iter_packed_batches(
    tokens: torch.Tensor,
    offsets: np.ndarray,
    idx: np.ndarray,
    batch_size: int,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
    """Yield padded ``(x [b,Tmax,D], mask [b,Tmax], batch_idx)`` for sample indices.

    Non-finite activation entries (e.g. fp16-overflow inf dims seen on some
    backbones) are zeroed so pooling stays finite.
    """
    for i in range(0, len(idx), batch_size):
        bidx = idx[i:i + batch_size]
        blocks = [tokens[offsets[j]:offsets[j + 1]] for j in bidx]
        lens = [b.shape[0] for b in blocks]
        t_max = max(lens)
        d = tokens.shape[1]
        x = torch.zeros(len(bidx), t_max, d, dtype=dtype)
        mask = torch.zeros(len(bidx), t_max, dtype=torch.bool)
        for r, (b, length) in enumerate(zip(blocks, lens)):
            x[r, :length] = b.to(dtype)
            mask[r, :length] = True
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        yield x.to(device), mask.to(device), bidx


def load_probe(path: str, d_model: int = D_MODEL, map_location: str = "cpu") -> AttnPoolProbe:
    """Load an ``AttnPoolProbe`` from an experiment-#4 ``.pt`` checkpoint.

    The checkpoint is ``torch.save({"arch", "head", "layer", "artifact": {"state_dict": ...}})``.
    Only ``arch == "attn_pool"`` checkpoints are supported.
    """
    # weights_only=True: the checkpoint schema (strings/ints/dicts/tensors)
    # is fully compatible, and it closes the arbitrary-code-execution pickle
    # path for user-suppliable weights_dir/paths.
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    arch = ckpt.get("arch")
    if arch not in ("attn_pool", None):
        raise ValueError(f"{path}: expected arch 'attn_pool', got {arch!r}")
    state = ckpt["artifact"]["state_dict"]
    model = AttnPoolProbe(d_model)
    model.load_state_dict(state)
    model.eval()
    return model


def save_probe(
    model: AttnPoolProbe,
    path: str,
    head: str,
    layer: int = 18,
    metrics: dict | None = None,
) -> None:
    """Save an ``AttnPoolProbe`` in the experiment-#4 checkpoint layout."""
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(
        {"arch": "attn_pool", "head": head, "layer": layer,
         "artifact": {"state_dict": state}, "metrics": metrics},
        path,
    )
