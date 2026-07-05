"""Frozen-backbone residual-stream activation extraction.

Captures the residual stream at a transformer block of a frozen causal LM, with
the SafeFlow hybrid tokenization recipe. The backbone is loaded in bf16 and
never trained.

Layer convention (per experiment #4): ``block`` is a 0-indexed transformer
block. Its residual-stream output is HF
``output_hidden_states[block + HS_OFFSET]`` with ``HS_OFFSET = 1`` (because
``hidden_states[0]`` is the embedding output). So block 18 -> ``hidden_states[19]``.
Verified in experiment #4 that block 18's residual output matches
``model.layers.18`` (mean cosine > 0.999).

Hybrid tokenization: each prompt is truncated to ``MAX_SEQ_LEN`` tokens; if the
sequence is longer than ``FIRST_N + LAST_N`` the kept window is the first
``FIRST_N`` plus the last ``LAST_N`` tokens (concatenated). This keeps the
instruction head and the recent tail while bounding sequence length.

Outputs per extracted block:
  * packed per-token activations ``(tokens [T_total, D] fp16, offsets [N+1])``
    for the attention-pooling probe;
  * ``mean`` [N, D] fp32 mean-pooled and ``last`` [N, D] fp32 last-token vectors
    (for linear/pooled probe baselines).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch

# block residual output = hidden_states[block + HS_OFFSET]
HS_OFFSET = 1

# Tokenization recipe (per experiment #4)
MAX_SEQ_LEN = 650
FIRST_N = 512
LAST_N = 128

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_LAYER = 18


@dataclass
class LayerActivations:
    """Captured activations for one block over a set of prompts."""

    tokens: torch.Tensor   # [T_total, D] fp16, sequences concatenated
    offsets: np.ndarray    # [N+1] int64
    mean: np.ndarray       # [N, D] fp32 mean-pooled
    last: np.ndarray       # [N, D] fp32 last-token

    @property
    def packed(self) -> tuple[torch.Tensor, np.ndarray]:
        return self.tokens, self.offsets


class ActivationExtractor:
    """Loads a frozen causal LM and extracts residual-stream activations.

    Example:
        ex = ActivationExtractor()                       # loads Qwen3-4B in bf16
        acts = ex.extract(prompts, layer=18)             # LayerActivations
        probs = probe.positive_proba(*acts.packed)       # route with a probe
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        hf_token: str | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = hf_token or os.environ.get("HF_TOKEN")
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Honor the device parameter: "cpu" loads without accelerate dispatch
        # (fp32 -- bf16 matmuls are unsupported or very slow on most CPUs),
        # and an explicit "cuda:N" places the backbone on that GPU rather
        # than always GPU 0.
        if str(device) == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32, token=token
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, device_map={"": str(device)}, token=token
            )
        self.model.eval()
        self.d_model = self.model.config.hidden_size

    @property
    def input_device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def extract(
        self,
        prompts: list[str],
        layer: int = DEFAULT_LAYER,
        batch_size: int = 16,
        progress: bool = True,
    ) -> LayerActivations:
        """Extract residual activations at ``layer`` for ``prompts``.

        Prompts are tokenized with the hybrid recipe, sorted by length to
        minimize padding, run through the frozen model, and the residual stream
        at ``hidden_states[layer + HS_OFFSET]`` is captured per token.
        """
        acts = self.extract_layers(prompts, [layer], batch_size, progress)
        return acts[layer]

    @torch.no_grad()
    def extract_layers(
        self,
        prompts: list[str],
        layers: list[int],
        batch_size: int = 16,
        progress: bool = True,
    ) -> dict[int, LayerActivations]:
        """Extract residual activations at several blocks in a single pass."""
        tok = self.tokenizer
        model = self.model
        device = self.input_device
        n = len(prompts)
        d = model.config.hidden_size

        enc = tok(prompts, truncation=True, max_length=MAX_SEQ_LEN, padding=False)
        ids_all = enc["input_ids"]
        # Guard zero-token prompts up front (Qwen's tokenizer returns [] for
        # "", no BOS): an empty sequence would IndexError on last-token
        # pooling part-way through a long GPU run. Error immediately, with
        # the offending positions named, before any forward pass.
        empty = [i for i in range(n) if len(ids_all[i]) == 0]
        if empty:
            raise ValueError(
                f"{len(empty)} prompt(s) tokenize to zero tokens "
                f"(e.g. positions {empty[:5]}) — the probe cannot route an "
                "empty prompt; filter these rows or fix the prompt field "
                "before extraction"
            )
        order = sorted(range(n), key=lambda i: len(ids_all[i]))

        per_layer_tokens = {b: [None] * n for b in layers}
        mean_pool = {b: np.zeros((n, d), dtype=np.float32) for b in layers}
        last_pool = {b: np.zeros((n, d), dtype=np.float32) for b in layers}

        t0 = time.time()
        for bi in range(0, n, batch_size):
            batch_idx = order[bi:bi + batch_size]
            batch_ids = [ids_all[i] for i in batch_idx]
            maxlen = max(len(x) for x in batch_ids)
            input_ids = torch.full((len(batch_ids), maxlen), tok.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(batch_ids), maxlen), dtype=torch.long)
            for r, ids in enumerate(batch_ids):
                input_ids[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                attn[r, :len(ids)] = 1
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attn.to(device),
                output_hidden_states=True,
                use_cache=False,
            )
            for b in layers:
                hs = out.hidden_states[b + HS_OFFSET]  # [B, L, D]
                for r, gi in enumerate(batch_idx):
                    length = len(batch_ids[r])
                    v = hs[r, :length].float()
                    if length > FIRST_N + LAST_N:
                        v = torch.cat([v[:FIRST_N], v[-LAST_N:]], dim=0)
                    mean_pool[b][gi] = v.mean(0).cpu().numpy()
                    last_pool[b][gi] = v[-1].cpu().numpy()
                    per_layer_tokens[b][gi] = v.half().cpu()
            del out
            if progress and (bi // batch_size) % 20 == 0:
                rate = (bi + len(batch_idx)) / max(time.time() - t0, 1e-6)
                print(f"[extract] {bi + len(batch_idx)}/{n} ({rate:.0f} seq/s)", flush=True)

        result = {}
        for b in layers:
            toks = per_layer_tokens[b]
            offsets = np.zeros(n + 1, dtype=np.int64)
            for i in range(n):
                offsets[i + 1] = offsets[i] + toks[i].shape[0]
            big = torch.empty((int(offsets[-1]), d), dtype=torch.float16)
            for i in range(n):
                big[offsets[i]:offsets[i + 1]] = toks[i]
                toks[i] = None
            result[b] = LayerActivations(big, offsets, mean_pool[b], last_pool[b])
        if progress:
            print(f"[extract] done {n} seqs in {time.time() - t0:.0f}s", flush=True)
        return result
