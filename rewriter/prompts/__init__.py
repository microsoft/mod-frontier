"""GEPA-optimized prompt packs and the selection rule.

Five prompt files ship alongside this module, produced by GEPA prompt
optimization (960 metric calls per scope) against a composite reward that
scores the rewrite with the ToxicChat T5 filter, a relevance judge, and a
harm judge (see ``rewriter/repro/gepa/``):

* ``creative_writing.md``, ``information_seeking.md``, ``other.md`` --
  per-domain rewrite prompts for the three dense routing domains.
* ``unified.md`` -- the fallback rewrite prompt for all other domains
  (sparse in the flagged distribution: casual_interaction, practical_content,
  roleplay, task_assistance, translation_transcription).
* ``refusal.md`` -- the contextual-refusal prompt used when the router
  decides REFUSE.

Selection rule (validated end-to-end on the GPT-5 evaluation set):

* decision REFUSE  -> the refusal prompt.
* decision REWRITE -> the per-domain prompt if the routed domain is one of
  the three dense scopes, else the unified fallback.

Prompt ids are ``gepa960/<scope>`` and are logged per row by the pipeline so
downstream checks can verify which prompt fired.
"""

from __future__ import annotations

from pathlib import Path

#: The three dense per-domain scopes; every other routed domain falls back to
#: the unified prompt.
DOMAIN_PACKS = ("creative_writing", "information_seeking", "other")

#: All prompt scopes shipped in this directory.
SCOPES = DOMAIN_PACKS + ("unified", "refusal")

#: Prefix for prompt ids logged by the pipeline.
PROMPT_ID_PREFIX = "gepa960"

_PROMPTS_DIR = Path(__file__).parent
_cache: dict[str, str] = {}


def load_prompt(scope: str) -> str:
    """Return the prompt text for ``scope`` (cached; raises on unknown/empty)."""
    if scope not in SCOPES:
        raise KeyError(f"unknown prompt scope {scope!r}; expected one of {SCOPES}")
    if scope not in _cache:
        path = _PROMPTS_DIR / f"{scope}.md"
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"empty prompt file: {path}")
        _cache[scope] = text
    return _cache[scope]


def select_rewrite_prompt(domain: str | None) -> tuple[str, str]:
    """Return ``(prompt_id, prompt_text)`` for a REWRITE row routed to ``domain``.

    Domains outside the three dense scopes (including ``None``) use the
    unified fallback.
    """
    scope = domain if domain in DOMAIN_PACKS else "unified"
    return f"{PROMPT_ID_PREFIX}/{scope}", load_prompt(scope)


def refusal_prompt() -> tuple[str, str]:
    """Return ``(prompt_id, prompt_text)`` for a REFUSE row."""
    return f"{PROMPT_ID_PREFIX}/refusal", load_prompt("refusal")


def verify_all() -> dict[str, int]:
    """Load every scope and return ``{scope: char_len}``.

    Raises if any prompt file is missing or empty -- call this before a long
    run to fail fast on a broken installation.
    """
    return {s: len(load_prompt(s)) for s in SCOPES}
