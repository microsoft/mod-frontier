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

from rewriter import eval_e2e
from rewriter.pipeline import load_routing_cache
from rewriter.rewrite import clean_completion
from rewriter.routing import file_sha256

# The round-2 test module carries the shared tiny-universe assemble helper.
from test_review_fixes import GOOD_GRADE, GOOD_RW, assemble

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


# ---------------------------------------------------------------------------
# 1: routing cache must be bound to the exact input file, not just threshold
# ---------------------------------------------------------------------------


class TestRoutingCacheInputBinding:
    def _cache(self, tmp_path, meta):
        p = tmp_path / "cache.json"
        p.write_text(json.dumps({"_meta": meta,
                                 "0": {"decision": "REWRITE", "domain": "other"}}))
        return str(p)

    def test_missing_input_hash_is_hard_error(self, tmp_path):
        inp = tmp_path / "data.jsonl"
        inp.write_text('{"user_input": "p0"}\n')
        p = self._cache(tmp_path, {"threshold": 0.161})  # old-format meta
        with pytest.raises(SystemExit):
            load_routing_cache(p, 0.161, str(inp))

    def test_regenerated_input_file_is_hard_error(self, tmp_path):
        inp = tmp_path / "data.jsonl"
        inp.write_text('{"user_input": "p0"}\n')
        p = self._cache(tmp_path, {"threshold": 0.161, "input_sha256": file_sha256(str(inp))})
        # the user regenerates/filters the data file (row order changes)
        inp.write_text('{"user_input": "pX"}\n{"user_input": "p0"}\n')
        with pytest.raises(SystemExit):
            load_routing_cache(p, 0.161, str(inp))

    def test_same_input_file_loads(self, tmp_path):
        inp = tmp_path / "data.jsonl"
        inp.write_text('{"user_input": "p0"}\n')
        p = self._cache(tmp_path, {"threshold": 0.161, "input_sha256": file_sha256(str(inp))})
        cache = load_routing_cache(p, 0.161, str(inp))
        assert "0" in cache


# ---------------------------------------------------------------------------
# 11: an explicitly requested --routing-cache that doesn't exist must error,
#     never silently fall through to a live GPU re-route
# ---------------------------------------------------------------------------


class TestMissingRoutingCachePath:
    def test_nonexistent_cache_path_is_hard_error(self, tmp_path, monkeypatch):
        from rewriter import pipeline

        data = tmp_path / "data.jsonl"
        data.write_text('{"conv_id": "aaa", "user_input": "p0", '
                        '"model_output_gpt5": "r0", "T5_model_output_gpt5": 1}\n')
        monkeypatch.setattr("sys.argv", [
            "pipeline.py", "-i", str(data), "-o", str(tmp_path / "out.jsonl"),
            "--routing-cache", str(tmp_path / "DOES_NOT_EXIST.json"),
        ])
        with pytest.raises(SystemExit, match="does not exist"):
            pipeline.main()


# ---------------------------------------------------------------------------
# 2: a grade record is bound to the exact rewrite text it graded
# ---------------------------------------------------------------------------


class TestGradeTextBinding:
    def test_stale_grade_on_new_text_is_hard_error(self, tmp_path):
        # same index, same conv_id (both legacy checks pass), different text
        new_rw = dict(GOOD_RW, model_output_rw_probe_probe="a DIFFERENT rewrite")
        with pytest.raises(SystemExit, match="DIFFERENT rewrite text"):
            assemble(tmp_path, [new_rw], [GOOD_GRADE], {"1": 0})

    def test_hashless_legacy_grade_is_hard_error(self, tmp_path):
        legacy = {k: v for k, v in GOOD_GRADE.items() if k != "text_sha256"}
        with pytest.raises(SystemExit, match="text_sha256"):
            assemble(tmp_path, [GOOD_RW], [legacy], {"1": 0})

    def test_matching_hash_assembles(self, tmp_path):
        out = assemble(tmp_path, [GOOD_RW], [GOOD_GRADE], {"1": 0})
        assert out[1]["model_output_rw_probe_probe"] == "a safe rewrite"


# ---------------------------------------------------------------------------
# 8: duplicate index values (two runs' outputs in one file) are a hard error
# ---------------------------------------------------------------------------


class TestDuplicateIndices:
    def test_load_jsonl_by_index_rejects_duplicates(self, tmp_path):
        p = tmp_path / "rw.jsonl"
        write_jsonl(p, [GOOD_RW, dict(GOOD_RW, model_output_rw_probe_probe="other run")])
        with pytest.raises(SystemExit, match="duplicate"):
            eval_e2e.load_jsonl_by_index(str(p))

    def test_require_unique_indices_rejects_duplicates(self):
        with pytest.raises(SystemExit, match="duplicate"):
            eval_e2e.require_unique_indices([GOOD_RW, dict(GOOD_RW)], "rw.jsonl")

    def test_unique_indices_pass(self):
        eval_e2e.require_unique_indices([GOOD_RW, dict(GOOD_RW, index=2)], "rw.jsonl")


# ---------------------------------------------------------------------------
# 10: golden-label cache is keyed by (row id, model); labeler recorded on row
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, text):
        self.content = [type("B", (), {"type": "text", "text": text})()]


class _FakeAnthropic:
    def __init__(self, toxic):
        self._toxic = toxic
        self.calls = 0

    @property
    def messages(self):
        outer = self

        class _M:
            def create(self, **kwargs):
                outer.calls += 1
                return _FakeMessage(json.dumps({
                    "domain": "other", "toxic": outer._toxic,
                    "severity": outer._toxic, "rationale": "stub",
                }))

        return _M()


class TestGoldenCacheModelKey:
    def test_cache_does_not_cross_models(self, tmp_path):
        from rewriter.repro.build_dataset import GoldenCache, label_one

        cache = GoldenCache(str(tmp_path / "golden_cache.json"))
        row = {"id": "row-1", "prompt": "p", "response": "r"}

        a = _FakeAnthropic(toxic=0)
        res_a = label_one(a, dict(row), cache, "model-a")
        assert (res_a["golden_toxic"], res_a["golden_model"]) == (0, "model-a")

        # A different labeler model must NOT be served model-a's cached label.
        b = _FakeAnthropic(toxic=1)
        res_b = label_one(b, dict(row), cache, "model-b")
        assert b.calls == 1  # actually called, not a cache hit
        assert (res_b["golden_toxic"], res_b["golden_model"]) == (1, "model-b")

    def test_same_model_resumes_from_cache(self, tmp_path):
        from rewriter.repro.build_dataset import GoldenCache, label_one

        cache = GoldenCache(str(tmp_path / "golden_cache.json"))
        row = {"id": "row-1", "prompt": "p", "response": "r"}
        a = _FakeAnthropic(toxic=0)
        label_one(a, dict(row), cache, "model-a")
        label_one(a, dict(row), cache, "model-a")
        assert a.calls == 1  # second call served from the cache
