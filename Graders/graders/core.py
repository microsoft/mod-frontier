"""Shared LLM-judge engine used by every grader in this package.

A *grader spec* (see ``specs/*.json``) is a JSON object::

    {
      "name": "toxicity_v10",
      "task": "toxicity",
      "paired": false,            # true => each sample is {prompt, response}
      "output_field": "label",    # JSON key the model returns
      "label_space": [0, 1],      # allowed label values
      "label_name": "toxicity_label",  # column written to output rows
      "aggregation": "majority",  # majority | or | and | single
      "members": [
        {"model": "gpt-4.1", "prompt": "toxicity/v07",
         "temperature": 0.0, "max_tokens": 300}
      ]
    }

Each member renders a prompt template with the sample, calls a judge model, and
parses a single label from ``output_field``. ``aggregation`` combines members.

The same engine powers both shipped graders:
  * ``toxicity_v10``  — binary (0/1) ensemble, unpaired ``user_input``.
  * ``relevance_v01`` — 1-3 score, paired ``response`` vs ``prompt``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from graders.azure_client import AzureChatClient, is_reasoning_model

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"
SPECS_DIR = ROOT / "specs"
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# A sample is either a plain string (unpaired) or a {"prompt", "response"} dict.
Sample = str | dict


# --------------------------- loading ---------------------------
def load_spec(name: str) -> dict[str, Any]:
    """Load a grader spec by name (with or without the .json suffix)."""
    path = SPECS_DIR / (name if name.endswith(".json") else f"{name}.json")
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in SPECS_DIR.glob("*.json")))
        raise FileNotFoundError(f"Spec '{name}' not found. Available: {available}")
    return json.loads(path.read_text())


def load_prompts(spec: dict[str, Any]) -> dict[str, str]:
    """Load every prompt template referenced by a spec's members."""
    prompts: dict[str, str] = {}
    for member in spec["members"]:
        name = member["prompt"]
        if name not in prompts:
            prompts[name] = (PROMPTS_DIR / f"{name}.txt").read_text()
    return prompts


# --------------------------- parsing ---------------------------
_BINARY_POS = {"1", "toxic", "yes", "true", "harmful", "unsafe"}
_BINARY_NEG = {"0", "non-toxic", "nontoxic", "no", "false", "safe", "benign"}


def parse_value(raw: str, output_field: str, label_space: list[int]) -> int | None:
    """Extract a label in ``label_space`` from a model response, else ``None``."""
    if not raw:
        return None
    text = raw.strip()

    # 1) Preferred: a JSON object carrying ``output_field``.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if output_field in obj:
                v = obj[output_field]
                if isinstance(v, bool):
                    v = int(v)
                if isinstance(v, (int, float)) and int(v) in label_space:
                    return int(v)
                if isinstance(v, str):
                    coerced = _coerce_token(v, label_space)
                    if coerced is not None:
                        return coerced
        except json.JSONDecodeError:
            pass

    # 2) Binary fallback: textual verdicts ("toxic"/"safe"/...).
    if set(label_space) == {0, 1}:
        binary = _parse_binary_text(text)
        if binary is not None:
            return binary

    # 3) Generic fallback: first standalone integer that is a valid label.
    for tok in re.findall(r"-?\d+", text):
        if int(tok) in label_space:
            return int(tok)
    return None


def _coerce_token(value: str, label_space: list[int]) -> int | None:
    v = value.strip().lower()
    if set(label_space) == {0, 1}:
        if v in _BINARY_POS:
            return 1
        if v in _BINARY_NEG:
            return 0
    if v.lstrip("-").isdigit() and int(v) in label_space:
        return int(v)
    return None


def _parse_binary_text(text: str) -> int | None:
    low = text.lower()
    if re.search(r"\b(label|verdict|answer)\b[^\n]*\b1\b", low):
        return 1
    if re.search(r"\b(label|verdict|answer)\b[^\n]*\b0\b", low):
        return 0
    if re.search(r"\btoxic\b", low) and not re.search(r"non[- ]?toxic", low):
        return 1
    if re.search(r"non[- ]?toxic|\bsafe\b|\bbenign\b", low):
        return 0
    if low.strip() in _BINARY_POS:
        return 1
    if low.strip() in _BINARY_NEG:
        return 0
    return None


# --------------------------- rendering ---------------------------
def render_messages(template: str, sample: Sample) -> list[dict[str, str]]:
    """Fill a template's placeholders with a sample (one user message).

    Placeholders are substituted literally (not via ``str.format``) so prompt
    text may freely contain JSON braces. Supported placeholders:
      * ``{text}``                  — unpaired string sample
      * ``{prompt}`` / ``{response}`` — paired {prompt, response} sample
    """
    content = template
    if isinstance(sample, dict):
        content = content.replace("{prompt}", str(sample.get("prompt", "")))
        content = content.replace("{response}", str(sample.get("response", "")))
    else:
        content = content.replace("{text}", str(sample))
    return [{"role": "user", "content": content}]


# --------------------------- cache ---------------------------
def _sample_repr(sample: Sample) -> str:
    if isinstance(sample, dict):
        return json.dumps(sample, ensure_ascii=False, sort_keys=True)
    return str(sample)


def _cache_key(model: str, prompt: str, sample: Sample) -> str:
    h = hashlib.sha256()
    for part in (model, prompt, _sample_repr(sample)):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()


def _cache_get(key: str) -> dict | None:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _cache_put(key: str, value: dict) -> None:
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(value))


# --------------------------- grading ---------------------------
async def grade_member(
    client: AzureChatClient,
    member: dict[str, Any],
    template: str,
    samples: list[Sample],
    *,
    output_field: str,
    label_space: list[int],
    regrade: bool = False,
) -> list[int | None]:
    model = member["model"]
    temperature = member.get("temperature", 0.0)
    max_tokens = member.get("max_tokens", 300)

    async def one(sample: Sample) -> int | None:
        key = _cache_key(model, template, sample)
        if not regrade:
            cached = _cache_get(key)
            if cached is not None:
                return cached.get("label")
        messages = render_messages(template, sample)
        kwargs: dict[str, Any] = {}
        if is_reasoning_model(model):
            kwargs["max_completion_tokens"] = max(max_tokens, 2000)
        else:
            kwargs["max_tokens"] = max_tokens
        try:
            raw = await client.chat(model, messages, temperature=temperature, **kwargs)
        except Exception:  # noqa: BLE001 - a failed call yields a None label
            return None
        label = parse_value(raw, output_field, label_space)
        _cache_put(key, {"raw": raw, "label": label})
        return label

    return await asyncio.gather(*[one(s) for s in samples])


def aggregate(per_member: list[list[int | None]], mode: str) -> list[int | None]:
    n = len(per_member[0])
    out: list[int | None] = []
    for i in range(n):
        votes = [m[i] for m in per_member if m[i] is not None]
        if not votes:
            out.append(None)
        elif mode == "single" or len(votes) == 1:
            out.append(votes[0])
        elif mode == "or":  # binary: positive if any member is positive
            out.append(1 if any(v == 1 for v in votes) else 0)
        elif mode == "and":  # binary: positive only if all members agree
            out.append(1 if all(v == 1 for v in votes) else 0)
        else:  # majority: most common value (ties -> highest label)
            counts: dict[int, int] = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            best = max(counts, key=lambda k: (counts[k], k))
            out.append(best)
    return out


async def run_grader(
    spec: dict[str, Any],
    samples: list[Sample],
    prompts: dict[str, str] | None = None,
    *,
    workers: int = 16,
    regrade: bool = False,
) -> list[int | None]:
    """Run a full grader spec over samples, returning one label per sample."""
    prompts = prompts or load_prompts(spec)
    output_field = spec["output_field"]
    label_space = spec["label_space"]
    client = AzureChatClient(workers=workers)
    try:
        per_member = []
        for member in spec["members"]:
            template = prompts[member["prompt"]]
            labels = await grade_member(
                client,
                member,
                template,
                samples,
                output_field=output_field,
                label_space=label_space,
                regrade=regrade,
            )
            per_member.append(labels)
        return aggregate(per_member, spec.get("aggregation", "majority"))
    finally:
        await client.aclose()


# --------------------------- metrics (binary) ---------------------------
def compute_metrics(y_true: list[int], y_pred: list[int | None]) -> dict[str, Any]:
    """Binary metrics with positive class = 1. ``None`` predictions score as 0."""
    tp = fp = tn = fn = n_none = 0
    for t, p in zip(y_true, y_pred):
        pp = 0 if p is None else p
        if p is None:
            n_none += 1
        if t == 1 and pp == 1:
            tp += 1
        elif t == 0 and pp == 1:
            fp += 1
        elif t == 0 and pp == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    prec0 = tn / (tn + fn) if (tn + fn) else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) else 0.0
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) else 0.0
    macro_f1 = (f1 + f1_0) / 2
    return {
        "n": len(y_true),
        "f1": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(acc, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_unparsed": n_none,
        "pos_rate_true": round(sum(y_true) / len(y_true), 4) if y_true else 0.0,
        "pos_rate_pred": round(sum(1 for p in y_pred if p == 1) / len(y_pred), 4)
        if y_pred else 0.0,
    }
