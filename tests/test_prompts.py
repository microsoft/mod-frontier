"""Prompt-pack loading and the selection rule (no GPU, no network)."""

import pytest

from rewriter import prompts


def test_all_scopes_load_nonempty():
    lens = prompts.verify_all()
    assert set(lens) == set(prompts.SCOPES)
    assert all(n > 100 for n in lens.values()), lens


def test_unknown_scope_raises():
    with pytest.raises(KeyError):
        prompts.load_prompt("nonexistent_scope")


@pytest.mark.parametrize("domain", prompts.DOMAIN_PACKS)
def test_dense_domain_selects_its_pack(domain):
    pid, text = prompts.select_rewrite_prompt(domain)
    assert pid == f"gepa960/{domain}"
    assert text == prompts.load_prompt(domain)


@pytest.mark.parametrize("domain", [
    "casual_interaction", "practical_content", "roleplay",
    "task_assistance", "translation_transcription", None, "unknown_domain",
])
def test_sparse_domain_falls_back_to_unified(domain):
    pid, text = prompts.select_rewrite_prompt(domain)
    assert pid == "gepa960/unified"
    assert text == prompts.load_prompt("unified")


def test_refusal_prompt():
    pid, text = prompts.refusal_prompt()
    assert pid == "gepa960/refusal"
    assert text == prompts.load_prompt("refusal")


def test_refusal_differs_from_rewrite_prompts():
    refusal = prompts.load_prompt("refusal")
    for scope in prompts.DOMAIN_PACKS + ("unified",):
        assert prompts.load_prompt(scope) != refusal
