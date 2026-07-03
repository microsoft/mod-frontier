"""Dataset loading, label extraction, and leakage-safe deduplication.

The SafeFlow routing datasets are JSONL with (at least) these fields per row:

    original_user_prompt : str   the prompt to route on
    decision             : str   "REFUSE" or "REWRITE" (refusal-head label)
    domain               : str   one of DOMAINS (domain-head label)

Leakage discipline (must match experiment #4 exactly, or reported metrics are
invalid): the held-out eval split is deduplicated against train by *normalized
prompt* (lowercase + whitespace-collapse), dropping any eval row whose
normalized prompt also appears in train. Seed is fixed at 42 throughout.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

SEED = 42
PROMPT_FIELD = "original_user_prompt"

DOMAINS = [
    "casual_interaction",
    "creative_writing",
    "information_seeking",
    "other",
    "practical_content",
    "roleplay",
    "task_assistance",
    "translation_transcription",
]


def normalize_prompt(s: str) -> str:
    """Lowercase + collapse all whitespace runs to single spaces, stripped.

    This is the canonical key used for leakage deduplication. Two prompts that
    differ only in case or whitespace are treated as the same item.
    """
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def refusal_label(decision) -> int:
    """1 if the routing decision is REFUSE, else 0 (REWRITE)."""
    return 1 if (decision or "").strip().upper() == "REFUSE" else 0


@dataclass
class Split:
    """A loaded data split.

    Attributes:
        prompts: list[str] of raw prompts.
        refusal: [N] int {0,1} refusal labels.
        domains: list[str] domain labels (raw strings; may be None).
    """

    prompts: list[str]
    refusal: np.ndarray
    domains: list

    def __len__(self) -> int:
        return len(self.prompts)

    def domain_onehot(self, domain: str) -> np.ndarray:
        """[N] int {0,1} one-vs-rest labels for a single domain."""
        return (np.asarray(self.domains) == domain).astype(np.int64)


def load_split(path: str, prompt_field: str = PROMPT_FIELD) -> Split:
    """Load a JSONL split. Rows missing a prompt are skipped."""
    prompts, dec, dom = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            p = d.get(prompt_field)
            if not p:
                continue
            prompts.append(p)
            dec.append(refusal_label(d.get("decision")))
            dom.append(d.get("domain"))
    return Split(prompts, np.array(dec, dtype=np.int64), dom)


def dedup_against(eval_split: Split, train_split: Split) -> tuple[Split, int]:
    """Drop eval rows whose normalized prompt appears in train (leakage dedup).

    Returns:
        (deduped_eval_split, n_dropped).
    """
    train_norm = {normalize_prompt(p) for p in train_split.prompts}
    keep = [i for i, p in enumerate(eval_split.prompts)
            if normalize_prompt(p) not in train_norm]
    dropped = len(eval_split) - len(keep)
    deduped = Split(
        prompts=[eval_split.prompts[i] for i in keep],
        refusal=eval_split.refusal[keep],
        domains=[eval_split.domains[i] for i in keep],
    )
    return deduped, dropped


def load_train_eval(
    train_path: str,
    eval_path: str,
    prompt_field: str = PROMPT_FIELD,
) -> tuple[Split, Split, int]:
    """Load train + eval and leakage-dedup eval against train.

    Returns:
        (train_split, deduped_eval_split, n_dropped).
    """
    train = load_split(train_path, prompt_field)
    ev = load_split(eval_path, prompt_field)
    ev_dedup, dropped = dedup_against(ev, train)
    return train, ev_dedup, dropped
