#!/usr/bin/env python3
"""DSPy GEPA harness for the SafeFlow rewrite-stage prompt optimization.

Derived from the SafeFlow GEPA optimization experiments. Provides everything
``run_gepa.py`` needs around the optimizer itself:

* :class:`LocalTaskLM` / :func:`load_task_model` — a DSPy-compatible wrapper
  around a local Hugging Face causal LM (the *task* model whose prompt GEPA
  optimizes; Qwen/Qwen3-4B-Instruct-2507, bfloat16, greedy decoding).
* :func:`load_reflection_model` — the reflection/mutation LM (gpt-5,
  temperature 1.0) as a plain ``dspy.LM``.
* :class:`SafeRewriter` / :class:`WeakRewriter` — the DSPy signature and
  module GEPA optimizes. The signature *instructions* are the rewrite prompt;
  GEPA mutates them.
* :func:`build_gepa` — the configured ``dspy.teleprompt.GEPA`` optimizer.

The task model runs generation under a lock: GEPA evaluates the metric from
multiple threads, and ``generate`` on a single CUDA model is not thread-safe.

Note on :class:`WeakRewriter`: each instance gets its OWN signature copy via
``SafeRewriter.with_instructions(...)``. Mutating ``signature.__doc__`` in
place would rewrite the instructions on the shared ``SafeRewriter`` class
object, so every rewriter built in the same process would silently read the
last-set prompt — fatal when evaluating several prompt packs (per-domain +
unified + refusal) in one process. GEPA optimizes the per-instance
instructions just the same.
"""
from __future__ import annotations

import contextlib
import json
import random
import threading
from pathlib import Path
from typing import Any

import dspy
import torch
from dspy.teleprompt import GEPA

DEFAULT_TASK_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_REFLECTION_MODEL = "gpt-5"


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Seed python, numpy (if present), and torch (incl. CUDA) RNGs."""
    random.seed(seed)
    with contextlib.suppress(Exception):
        import numpy as np

        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts (blank lines skipped).

    A corrupt or truncated line is a hard error, not a silent drop: GEPA
    would otherwise optimize and report val scores over a smaller train/val
    split than ``result.json`` claims.
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{ln}: corrupt JSONL line ({e}) — refusing to "
                    "silently drop dataset rows; fix or regenerate the file"
                ) from e
    return rows


# --------------------------------------------------------------------------
# Task model: local HF causal LM behind the DSPy LM interface
# --------------------------------------------------------------------------
class LocalTaskLM(dspy.LM):
    """DSPy LM over a local Hugging Face causal LM.

    Chat messages are rendered with the tokenizer's chat template; generation
    is greedy at ``temperature=0`` (the production rewrite setting) and
    serialized behind a lock so GEPA's multi-threaded metric evaluation cannot
    interleave forward passes on one CUDA model.
    """

    def __init__(self, model, tokenizer, model_id: str,
                 temperature: float = 0.0, max_new_tokens: int = 1024) -> None:
        self.model_obj = model
        self.tokenizer = tokenizer
        self.model = model_id
        self.kwargs = {"temperature": float(temperature), "max_tokens": int(max_new_tokens)}
        self._gen_lock = threading.Lock()

    # -- prompt rendering ---------------------------------------------------
    def _render(self, messages=None, prompt=None) -> str:
        if messages is None and prompt is None:
            raise ValueError("Either messages or prompt must be provided")
        if messages is None:
            return str(prompt)
        if isinstance(messages, str):
            return messages
        if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            if hasattr(self.tokenizer, "apply_chat_template"):
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            lines = [f"{(m.get('role') or 'user').upper()}:\n{m.get('content') or ''}\n"
                     for m in messages]
            return "\n".join(lines) + "ASSISTANT:\n"
        return str(messages)

    # -- generation -----------------------------------------------------------
    def __call__(self, messages=None, prompt=None, **kwargs) -> list[str]:
        text = self._render(messages, prompt)
        temperature = float(kwargs.get("temperature", self.kwargs["temperature"]) or 0.0)
        max_tokens = int(kwargs.get("max_tokens", self.kwargs["max_tokens"]))
        sample = temperature > 0.0

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model_obj.device)
        with self._gen_lock, torch.no_grad():
            output_ids = self.model_obj.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=sample,
                top_p=0.9 if sample else None,
                top_k=50 if sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        return [self.tokenizer.decode(new_tokens, skip_special_tokens=True)]


def load_task_model(
    model_name: str = DEFAULT_TASK_MODEL,
    *,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> LocalTaskLM:
    """Load the local task model (weights + tokenizer) as a :class:`LocalTaskLM`."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        dtype.lower(), torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True
    )
    return LocalTaskLM(model, tokenizer, model_name,
                       temperature=temperature, max_new_tokens=max_new_tokens)


def load_reflection_model(
    model_id: str = DEFAULT_REFLECTION_MODEL,
    *,
    api_key: str,
    temperature: float = 1.0,
    max_tokens: int = 16000,
) -> dspy.LM:
    """Build the GEPA reflection/mutation LM (gpt-5 via the OpenAI API)."""
    return dspy.LM(model=model_id, api_key=api_key,
                   temperature=temperature, max_tokens=max_tokens)


# --------------------------------------------------------------------------
# The program GEPA optimizes
# --------------------------------------------------------------------------
class SafeRewriter(dspy.Signature):
    """Initial instructions will be injected and optimized by GEPA."""

    user_prompt: str = dspy.InputField(desc="The user's original request")
    unsafe_draft: str = dspy.InputField(desc="The potentially unsafe content to sanitize")
    safe_rewrite: str = dspy.OutputField(desc="A safe, sanitized version of the content")


class WeakRewriter(dspy.Module):
    """The rewriter program GEPA optimizes: a single Predict step whose
    signature instructions ARE the rewrite prompt.

    Built with ``SafeRewriter.with_instructions(initial_prompt)`` so each
    instance owns its signature copy (see the module docstring for why).
    """

    def __init__(self, initial_prompt: str) -> None:
        super().__init__()
        self.rewriter = dspy.Predict(SafeRewriter.with_instructions(initial_prompt))

    def forward(self, user_prompt: str, unsafe_draft: str):
        return self.rewriter(user_prompt=user_prompt, unsafe_draft=unsafe_draft)


def get_optimized_prompt(program: dspy.Module) -> str:
    """Extract the (optimized) signature instructions from a compiled program."""
    sig = program.rewriter.signature
    return getattr(sig, "instructions", None) or (sig.__doc__ or "")


# --------------------------------------------------------------------------
# Optimizer construction
# --------------------------------------------------------------------------
def build_gepa(
    *,
    metric_fn,
    reflection_lm: dspy.LM,
    max_metric_calls: int,
    num_threads: int,
    use_wandb: bool = False,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    log_dir: str | None = None,
    seed: int = 42,
    reflection_minibatch_size: int = 3,
    use_merge: bool = True,
    track_stats: bool = True,
    track_best_outputs: bool = True,
    skip_perfect_score: bool = False,
) -> GEPA:
    """Build the configured GEPA optimizer.

    ``metric_fn`` must return ``ScoreWithFeedback`` (see ``reward.py``); the
    feedback text is what the reflection LM reads when mutating the prompt.
    ``use_cloudpickle`` is required: DSPy's dynamically-created signatures are
    not picklable with stdlib pickle, and without it GEPA's per-round state
    checkpoints (which enable resume-on-interruption) silently fail to write.
    """
    wandb_init_kwargs = None
    if use_wandb and (wandb_project or wandb_run_name):
        wandb_init_kwargs = {}
        if wandb_project:
            wandb_init_kwargs["project"] = wandb_project
        if wandb_run_name:
            wandb_init_kwargs["name"] = wandb_run_name

    return GEPA(
        metric=metric_fn,
        max_metric_calls=max_metric_calls,
        num_threads=num_threads,
        track_stats=track_stats,
        track_best_outputs=track_best_outputs,
        use_wandb=use_wandb,
        wandb_init_kwargs=wandb_init_kwargs,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=reflection_minibatch_size,
        use_merge=use_merge,
        skip_perfect_score=skip_perfect_score,
        log_dir=log_dir,
        seed=int(seed),
        gepa_kwargs={"use_cloudpickle": True},
    )
