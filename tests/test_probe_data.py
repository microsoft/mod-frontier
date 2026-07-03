"""Data loading: label extraction, normalization, leakage dedup."""
import numpy as np

from rewriter.routing_probe import (
    DOMAINS,
    Split,
    dedup_against,
    normalize_prompt,
    refusal_label,
)


def test_normalize_prompt():
    assert normalize_prompt("  Hello   World  ") == "hello world"
    assert normalize_prompt("Hello\n\tWorld") == "hello world"
    assert normalize_prompt("HELLO world") == normalize_prompt("hello   WORLD")
    assert normalize_prompt(None) == ""


def test_refusal_label():
    assert refusal_label("REFUSE") == 1
    assert refusal_label("refuse") == 1
    assert refusal_label(" Refuse ") == 1
    assert refusal_label("REWRITE") == 0
    assert refusal_label("") == 0
    assert refusal_label(None) == 0


def test_domain_onehot():
    s = Split(prompts=["a", "b", "c"], refusal=np.array([0, 1, 0]),
              domains=["roleplay", "other", "roleplay"])
    assert s.domain_onehot("roleplay").tolist() == [1, 0, 1]
    assert s.domain_onehot("other").tolist() == [0, 1, 0]


def test_dedup_against_drops_normalized_overlap():
    train = Split(prompts=["Hello World", "keep me"], refusal=np.array([0, 1]),
                  domains=["other", "other"])
    ev = Split(
        prompts=["hello   world", "unique eval prompt", "HELLO WORLD"],
        refusal=np.array([1, 0, 1]),
        domains=["roleplay", "other", "roleplay"],
    )
    deduped, dropped = dedup_against(ev, train)
    # rows 0 and 2 normalize to "hello world" which is in train -> dropped
    assert dropped == 2
    assert len(deduped) == 1
    assert deduped.prompts == ["unique eval prompt"]
    assert deduped.refusal.tolist() == [0]


def test_dedup_no_overlap():
    train = Split(prompts=["a"], refusal=np.array([0]), domains=["other"])
    ev = Split(prompts=["b", "c"], refusal=np.array([1, 0]), domains=["other", "roleplay"])
    deduped, dropped = dedup_against(ev, train)
    assert dropped == 0
    assert len(deduped) == 2


def test_domains_canonical():
    assert len(DOMAINS) == 8
    assert DOMAINS == sorted(DOMAINS)  # alphabetical, matches OvR head ordering
    assert "other" in DOMAINS
