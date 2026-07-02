#!/usr/bin/env python3
"""Local Qwen3-4B rewriter: turn a T5-flagged response into a safe rewrite
(or a contextual refusal) under a GEPA-optimized system prompt.

This is a self-contained implementation of the exact generation path the
end-to-end numbers were measured with (the SafeFlow local-model rewriter,
gepa-override path, Qwen3-4B backbone):

* **Messages.** system = the selected prompt pack (see
  :mod:`rewriter.prompts`); user = a fixed scaffold around the flagged
  response: a content label (different for REFUSE vs REWRITE), the original
  user request, and a hard-rules suffix (:data:`REWRITE_SAFETY_SUFFIX` or
  :data:`REFUSE_USER_SUFFIX`). These suffixes are *not* part of the
  GEPA-optimized prompts -- they are non-negotiable guardrails that wrap
  every rewrite, and the optimized prompts were trained with them in place.
* **Generation.** Chat template with generation prompt, batched with left
  padding, greedy decoding (``do_sample=False``, temperature 0),
  ``max_new_tokens=2048``, float16 on CUDA.
* **Cleanup.** Strip role-marker artifacts and boilerplate prefixes from the
  completion (:func:`clean_completion`).
* **Bare-refusal retry.** If a completion is a bare one-liner refusal
  (:func:`is_bare_refusal`), retry once with a contextual-refusal prompt that
  demands a policy explanation and a safe alternative. Fires rarely; a false
  positive costs one extra generation, never a loop.

Usage: construct one :class:`QwenRewriter` (loads the model) and call
:meth:`QwenRewriter.rewrite_batch` with :class:`RewriteRequest` items. The
CLI entry point over a JSONL evaluation file is ``rewriter/pipeline.py``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

#: HF model id of the rewriter backbone.
QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

#: Generation budget for one rewrite.
MAX_NEW_TOKENS = 2048

#: User-message label above the flagged response, REWRITE decisions.
UNSAFE_DRAFT_LABEL = "Unsafe draft:"

#: User-message label above the flagged response, REFUSE decisions.
REFUSE_CONTENT_LABEL = "Content that needs a refusal response:"

#: Instruction line appended (before the safety suffix) on REWRITE rows.
REWRITE_RETURN_INSTRUCTION = "\nReturn only the rewritten response:"

#: Hard safety rules appended to the user turn on every REWRITE row. Not part
#: of the GEPA-optimized system prompt -- a fixed guardrail wrapper.
REWRITE_SAFETY_SUFFIX = (
    "\n\nAdditional safety rules:\n"
    "1. Do not repeat or quote graphic/explicit content verbatim from the unsafe draft. "
    "Transform it using the guidance above.\n"
    "2. Never name platforms or brands whose primary purpose is adult, sexual, or fetish content. "
    "Use generic terms like 'community groups', 'dedicated platforms', or 'personal wellness products'.\n"
    "3. If the content primarily involves marketing sex work, explicit sexual tutorials, "
    "adult product promotion, or escort services and cannot be rewritten without preserving "
    "explicit details, produce a contextual refusal that names the policy area and suggests a safe alternative.\n"
    "4. Avoid step-by-step sexual or violent techniques."
)

#: User-message suffix on every REFUSE row. Reinforces the contextual-refusal
#: instruction so the model does not collapse to a bare one-liner.
REFUSE_USER_SUFFIX = (
    "\n\nWrite a contextual 2-3 sentence refusal as instructed above. "
    "Your refusal MUST:\n"
    "- Name the specific policy area being violated (e.g., explicit content, self-harm, harassment)\n"
    "- Suggest a concrete safe alternative the user could pursue instead\n"
    "- Be respectful but firm\n\n"
    "Do NOT output a bare one-liner like 'I'm sorry, but I can't assist with that request.' — "
    "that is not acceptable. Always explain why and offer an alternative.\n\n"
    "Hard safety rules:\n"
    "1. Never repeat, quote, or closely paraphrase harmful content from the text above.\n"
    "2. Never name adult/sexual/fetish platforms or brands.\n"
    "3. Do not include erotic/graphic language, anatomy terms, or step-by-step techniques."
)

#: Chain-of-thought guidance used as the system prompt of the bare-refusal
#: retry pass (the stock contextual-refusal instruction the rewrite stage was
#: validated with).
REFUSAL_RETRY_GUIDANCE = (
    "Think step by step about this refusal: 1. What specific policy does this request "
    "violate? (violence/explicit/illegal/harassment/self-harm) 2. How can I explain this "
    "clearly to the user without repeating any slurs, offensive terms, or harmful content "
    "from the original request? Describe the topic area in general terms instead of quoting "
    "specific harmful language. 3. What safer alternative could I suggest? Now write a 2-3 "
    "sentence refusal that covers all three points naturally. IMPORTANT: Never repeat, "
    "quote, or echo slurs, hate speech, explicit terms, or other harmful language from the "
    "original request — refer to the subject matter in general terms only. Never describe "
    "specific violent or sexual acts, body parts, or methods — describe the category instead "
    "(e.g., \"extreme body modification\" not the specific act; \"animal mating behavior\" "
    "not the specific anatomy; \"intimate products\" not brand names). Never name "
    "adult/sexual/fetish platforms or brands."
)

# ---------------------------------------------------------------------------
# Bare-refusal detection
# ---------------------------------------------------------------------------

#: Exact bare one-liner refusals (lowercased, curly apostrophes normalized).
_BARE_EXACT = {
    "i'm sorry, but i can't assist with that request.",
    "i'm sorry, but i can't assist with that.",
    "i cannot assist with that request.",
    "sorry, but i can't help with that.",
    "i'm sorry, but i'm unable to assist with that request.",
}

#: Refusal openers across the languages the rewrite stage was evaluated on.
#: Compound forms are used where a bare opener would match benign sentences.
_BARE_REFUSAL_OPENERS = (
    # English
    "i'm sorry", "i cannot", "i can't", "sorry,", "i apologize", "i'm unable",
    "i'm not able to help", "i'm not able to assist", "i must decline",
    "i refuse to help", "i refuse to assist", "i refuse to provide",
    "i won't help", "i won't assist", "i won't provide",
    # Spanish
    "lo siento", "disculpa", "disculpe", "perdón", "perdona", "no puedo",
    "me temo que", "no podré", "no voy a ayudar", "no voy a poder",
    "lamentablemente, no",
    # French
    "désolé", "je suis désolé", "je ne peux pas", "je suis navré",
    "je regrette", "je ne pourrai", "je ne suis pas en mesure",
    "malheureusement, je ne",
    # German
    "entschuldigung", "es tut mir leid", "tut mir leid", "leider kann ich",
    "ich kann nicht helfen", "ich kann dir nicht helfen", "ich bedaure",
    "ich darf nicht helfen",
    # Portuguese
    "desculpe", "desculpa", "lamento", "sinto muito", "peço desculpa",
    "não posso ajudar", "não posso atender", "não posso fornecer",
    "infelizmente, não",
    # Italian
    "mi dispiace", "mi spiace", "mi scuso", "non posso", "non sono in grado",
    "mi rincresce", "purtroppo, non",
    # Chinese
    "对不起", "抱歉", "很抱歉", "不好意思", "我无法", "我不能", "我没办法", "非常抱歉",
    # Japanese
    "申し訳", "すみません", "ごめん", "お力にな", "失礼しました", "恐れ入りますが",
    "お答えできません",
)

#: Sentence terminators across scripts: ASCII + CJK (U+3002/FF01/FF1F).
_SENTENCE_TERMINATORS = re.compile(r"[.!?。！？]+")


def is_bare_refusal(text: str) -> bool:
    """True iff ``text`` is a bare one-liner refusal (no context, no alternative).

    A contextual refusal names a policy area AND suggests an alternative,
    which needs at least two sentences; a bare refusal is a single apologetic
    clause: (starts with a refusal opener) AND (<= 1 sentence) AND (< 80 chars).
    Runs of terminators ("...", "?!") count once, so emphatic punctuation does
    not inflate the sentence count.
    """
    stripped = text.strip()
    if not stripped:
        return False
    lower = stripped.replace("‘", "'").replace("’", "'").lower()
    if lower in _BARE_EXACT:
        return True
    if not lower.startswith(_BARE_REFUSAL_OPENERS):
        return False
    fragments = [f for f in _SENTENCE_TERMINATORS.split(stripped) if f.strip()]
    return len(fragments) <= 1 and len(stripped) < 80


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


def build_messages(
    content: str,
    user_input: str | None,
    decision: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Build the chat messages for one flagged row.

    ``content`` is the flagged response text (the "unsafe draft"),
    ``user_input`` the original user request, ``decision`` REFUSE or REWRITE,
    ``system_prompt`` the selected prompt pack text.
    """
    is_refuse = decision.strip().upper() == "REFUSE"
    parts = [REFUSE_CONTENT_LABEL if is_refuse else UNSAFE_DRAFT_LABEL, content]
    if user_input:
        parts.append("\nOriginal user request:\n" + str(user_input))
    if is_refuse:
        parts.append(REFUSE_USER_SUFFIX)
    else:
        parts.append(REWRITE_RETURN_INSTRUCTION)
        parts.append(REWRITE_SAFETY_SUFFIX)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(parts)},
    ]


def build_refusal_retry_messages(
    content: str,
    user_input: str | None,
    decision: str,
) -> list[dict[str, str]]:
    """Messages for the one-shot retry after a bare-refusal completion."""
    attempt_desc = (
        "The original refusal attempt produced a bare one-line refusal."
        if decision.strip().upper() == "REFUSE"
        else "The original rewrite attempt produced a bare refusal."
    )
    system = (
        attempt_desc
        + " Your task is now to produce a helpful contextual refusal instead - one that "
        "explains why the content cannot be handled safely and suggests a specific safe "
        "alternative.\n\n"
        + REFUSAL_RETRY_GUIDANCE
    )
    preview = content[:1500] + ("...[truncated]" if len(content) > 1500 else "")
    parts = [
        "The following content could not be safely rewritten:",
        "---",
        preview,
        "---",
    ]
    if user_input:
        parts.append("\nOriginal user request:\n" + str(user_input))
    parts.append(REFUSE_USER_SUFFIX)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts)},
    ]


def clean_completion(completion: str) -> str:
    """Strip role-marker artifacts and boilerplate prefixes from a completion."""
    if "\nassistant\n" in completion:
        completion = completion.split("\nassistant\n")[-1].strip()
    elif completion.startswith("assistant\n"):
        completion = completion[len("assistant\n"):].strip()
    elif "\nmodel\n" in completion:
        completion = completion.split("\nmodel\n")[-1].strip()
    elif completion.startswith("model\n"):
        completion = completion[len("model\n"):].strip()
    for prefix in ("Rewritten response:", "Rewritten:", "Assistant response:"):
        if completion.lower().startswith(prefix.lower()):
            completion = completion[len(prefix):].lstrip()
            break
    completion = re.sub(r"^\s*assistant\s*:?\s*", "", completion, count=1, flags=re.IGNORECASE)
    return completion


# ---------------------------------------------------------------------------
# The rewriter
# ---------------------------------------------------------------------------


@dataclass
class RewriteRequest:
    """One flagged row to rewrite."""

    content: str                    #: the flagged response text
    user_input: str | None          #: the original user request
    decision: str                   #: "REFUSE" | "REWRITE"
    system_prompt: str              #: selected prompt pack text


@dataclass
class RewriteResult:
    """One rewrite outcome."""

    text: str                       #: cleaned completion
    success: bool                   #: non-empty completion
    latency_s: float                #: wall-clock of the generate call that produced it
    bare_refusal_retried: bool = field(default=False)


class QwenRewriter:
    """Batched greedy rewriter on ``Qwen/Qwen3-4B-Instruct-2507``.

    Loads the model once (float16 on CUDA, float32 on CPU); call
    :meth:`rewrite_batch` with a list of :class:`RewriteRequest`.
    ``batch_size`` bounds how many sequences share one ``generate`` call --
    long flagged responses at 2048 new tokens OOM a single whole-set batch,
    8 is a safe default on an 80 GB GPU.
    """

    def __init__(self, model_name: str = QWEN_MODEL, device_map: str = "auto",
                 max_new_tokens: int = MAX_NEW_TOKENS) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print(f"QwenRewriter initialized: model={model_name}, dtype={dtype}, "
              f"max_new_tokens={max_new_tokens}", flush=True)

    # -- internals ----------------------------------------------------------

    def _generate(self, all_messages: list[list[dict[str, str]]]) -> tuple[list[str], float]:
        """One batched generate over ``all_messages``; returns (completions, wall_s)."""
        torch = self._torch
        enc = self.tokenizer.apply_chat_template(
            all_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if not isinstance(enc, dict):
            enc = {"input_ids": enc}
        device = self.model.get_input_embeddings().weight.device
        enc = {k: v.to(device) for k, v in enc.items()}

        t0 = time.time()
        with torch.no_grad():
            sequences = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        wall = time.time() - t0

        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask")
        if attn is not None:
            true_lengths = [int(n.item()) for n in attn.sum(dim=1)]
        else:
            true_lengths = [int(input_ids.shape[1])] * input_ids.shape[0]

        completions = []
        for i in range(input_ids.shape[0]):
            new_tokens = sequences[i][true_lengths[i]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            completions.append(clean_completion(text))
        return completions, wall

    # -- public API ----------------------------------------------------------

    def rewrite_batch(self, requests: list[RewriteRequest],
                      batch_size: int = 8) -> list[RewriteResult]:
        """Rewrite every request; greedy decoding, chunked batched generation.

        After the batched pass, any completion that is a bare one-liner
        refusal is retried once individually with the contextual-refusal
        prompt; the retry replaces the original only if it is non-empty and
        not itself a bare refusal.
        """
        results: list[RewriteResult] = []
        for start in range(0, len(requests), batch_size):
            chunk = requests[start:start + batch_size]
            messages = [
                build_messages(r.content, r.user_input, r.decision, r.system_prompt)
                for r in chunk
            ]
            completions, wall = self._generate(messages)
            results.extend(
                RewriteResult(text=c, success=bool(c), latency_s=wall) for c in completions
            )
            print(f"  rewrote {min(start + batch_size, len(requests))}/{len(requests)} "
                  f"({wall:.1f}s batch)", flush=True)

        # Bare-refusal retry pass (rare; individual generation is acceptable).
        for i, (res, req) in enumerate(zip(results, requests)):
            if not is_bare_refusal(res.text):
                continue
            print(f"  bare refusal at item {i} — retrying with contextual-refusal prompt "
                  f"(preview: {res.text[:80]!r})", flush=True)
            retry_msgs = build_refusal_retry_messages(req.content, req.user_input, req.decision)
            retry_texts, retry_wall = self._generate([retry_msgs])
            retry_text = retry_texts[0]
            if retry_text and not is_bare_refusal(retry_text):
                results[i] = RewriteResult(
                    text=retry_text,
                    success=True,
                    latency_s=res.latency_s + retry_wall,
                    bare_refusal_retried=True,
                )
        return results
