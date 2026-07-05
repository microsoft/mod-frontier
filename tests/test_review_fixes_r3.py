"""Regression tests for the PR #5 round-3 code-review fixes.

Each test encodes a reviewed failure scenario; numbers refer to the round-3
review's finding ids. The identity-binding tests are the load-bearing ones:
stale or mismatched artifacts (routing caches, grade records, prompt-level
label files) must hard-error instead of silently attaching another run's
decisions to the published outputs.
"""

import json
from pathlib import Path

import pytest

from rewriter.rewrite import clean_completion

REPO = Path(__file__).resolve().parent.parent


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 9: clean_completion must not strip a legitimate leading "Assistant..."
# ---------------------------------------------------------------------------


class TestCleanCompletionWordBoundary:
    def test_assistant_as_first_word_preserved(self):
        text = "Assistant managers should complete the certification course."
        assert clean_completion(text) == text

    def test_assistantship_preserved(self):
        text = "Assistantship applications are due in March."
        assert clean_completion(text) == text

    def test_role_marker_with_colon_stripped(self):
        assert clean_completion("assistant: Here is a safer answer.") == (
            "Here is a safer answer."
        )
        assert clean_completion("Assistant:\nHere is a safer answer.") == (
            "Here is a safer answer."
        )

    def test_role_marker_alone_on_first_line_stripped(self):
        assert clean_completion("assistant\nHere is a safer answer.") == (
            "Here is a safer answer."
        )
