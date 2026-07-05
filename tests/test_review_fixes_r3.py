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
from test_review_fixes import GOOD_GRADE, GOOD_RW, GOOD_T5, assemble, base_rows

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
            assemble(tmp_path, [new_rw], [GOOD_GRADE], GOOD_T5)

    def test_hashless_legacy_grade_is_hard_error(self, tmp_path):
        legacy = {k: v for k, v in GOOD_GRADE.items() if k != "text_sha256"}
        with pytest.raises(SystemExit, match="text_sha256"):
            assemble(tmp_path, [GOOD_RW], [legacy], GOOD_T5)

    def test_matching_hash_assembles(self, tmp_path):
        out = assemble(tmp_path, [GOOD_RW], [GOOD_GRADE], GOOD_T5)
        assert out[1]["model_output_rw_probe_probe"] == "a safe rewrite"


# ---------------------------------------------------------------------------
# 3: prompt-level label files must cover every row (no fail-open 0-fills)
# ---------------------------------------------------------------------------


def assemble_args(d, *, t5_prompts, grade_prompts, flag_field="T5_model_output_gpt5",
                  rows=None, rewrites=None, grades=None):
    import argparse

    write_jsonl(d / "data.jsonl", rows if rows is not None else base_rows())
    write_jsonl(d / "rw.jsonl", rewrites if rewrites is not None else [GOOD_RW])
    write_jsonl(d / "grades.jsonl", grades if grades is not None else [GOOD_GRADE])
    (d / "t5_rw.json").write_text(json.dumps(GOOD_T5))
    (d / "t5_prompts.json").write_text(json.dumps(t5_prompts))
    (d / "grade_prompts.json").write_text(json.dumps(grade_prompts))
    return argparse.Namespace(
        input=str(d / "data.jsonl"), rewrites=str(d / "rw.jsonl"),
        grades=str(d / "grades.jsonl"), t5_rewrites=str(d / "t5_rw.json"),
        t5_prompts=str(d / "t5_prompts.json"), grade_prompts=str(d / "grade_prompts.json"),
        output=str(d / "out.jsonl"), arm="rw_probe_probe",
        prompt_field="user_input", flag_field=flag_field,
    )


class TestPromptFileCoverage:
    def test_truncated_t5_prompts_is_hard_error(self, tmp_path):
        args = assemble_args(tmp_path, t5_prompts={"0": 0, "1": 1},  # row 2 missing
                             grade_prompts={"0": 0, "1": 1, "2": 0})
        with pytest.raises(SystemExit, match="t5-prompts"):
            eval_e2e.cmd_assemble(args)

    def test_truncated_grade_prompts_is_hard_error(self, tmp_path):
        args = assemble_args(tmp_path, t5_prompts={"0": 0, "1": 1, "2": 0},
                             grade_prompts={"1": 1})  # rows 0 and 2 missing
        with pytest.raises(SystemExit, match="grade-prompts"):
            eval_e2e.cmd_assemble(args)

    def test_full_coverage_assembles(self, tmp_path):
        args = assemble_args(tmp_path, t5_prompts={"0": 0, "1": 1, "2": 0},
                             grade_prompts={"0": 0, "1": 1, "2": None})
        eval_e2e.cmd_assemble(args)  # must not raise; None = unparsed, explicit
        out = [json.loads(l) for l in open(tmp_path / "out.jsonl")]
        assert out[2]["grader_user_input"] == 0  # explicit unparsed record


# ---------------------------------------------------------------------------
# 4: cmd_metrics must hard-error on missing arm columns, not report zeros
# ---------------------------------------------------------------------------


class TestMetricsArmColumnGuard:
    def _metrics_args(self, tmp_path, rows, arm="rw_probe_probe"):
        import argparse

        write_jsonl(tmp_path / "in.jsonl", rows)
        return argparse.Namespace(input=str(tmp_path / "in.jsonl"),
                                  output=str(tmp_path / "m.json"), arm=arm,
                                  prompt_field="user_input",
                                  flag_field="T5_model_output_gpt5", n_boot=0)

    def _assembled_rows(self):
        rows = base_rows()
        for r in rows:
            r.update({"grader_model_output_rw_probe_probe": 0,
                      "relevance_score_rw_probe_probe": 3,
                      "T5_model_output_rw_probe_probe": 0,
                      "T5_user_input": 0, "grader_user_input": 0})
        return rows

    def test_pre_assemble_input_is_hard_error(self, tmp_path):
        # base columns exist, but no arm columns and no prompt-level columns
        args = self._metrics_args(tmp_path, base_rows())
        with pytest.raises(SystemExit, match="no row has the column"):
            eval_e2e.cmd_metrics(args)

    def test_mistyped_arm_is_hard_error(self, tmp_path):
        args = self._metrics_args(tmp_path, self._assembled_rows(), arm="rw_probe")
        with pytest.raises(SystemExit, match="rw_probe"):
            eval_e2e.cmd_metrics(args)

    def test_assembled_input_passes(self, tmp_path):
        args = self._metrics_args(tmp_path, self._assembled_rows())
        eval_e2e.cmd_metrics(args)  # must not raise
        assert json.loads((tmp_path / "m.json").read_text())["meta"]["n_total"] == 3


# ---------------------------------------------------------------------------
# 6: pass-through columns come from --flag-field's base arm, never a
#    hardcoded *_gpt5
# ---------------------------------------------------------------------------


class TestPassThroughBaseArm:
    def test_non_gpt5_base_arm_pass_through(self, tmp_path):
        # rows carry two baselines with DIFFERENT values; flag against otherbase
        rows = []
        for k, conv in enumerate(("aaa", "bbb", "ccc")):
            rows.append({
                "conv_id": conv, "user_input": f"p{k}",
                "model_output_gpt5": "resp", "grader_model_output_gpt5": 1,
                "relevance_score_gpt5": 1, "T5_model_output_gpt5": 1,
                "model_output_otherbase": "resp2", "grader_model_output_otherbase": 0,
                "relevance_score_otherbase": 3, "T5_model_output_otherbase": 0,
            })
        args = assemble_args(tmp_path, t5_prompts={"0": 0, "1": 0, "2": 0},
                             grade_prompts={"0": 0, "1": 0, "2": 0},
                             flag_field="T5_model_output_otherbase",
                             rows=rows, rewrites=[], grades=[])
        eval_e2e.cmd_assemble(args)
        out = [json.loads(l) for l in open(tmp_path / "out.jsonl")]
        # every row is unflagged pass-through: values must be otherbase's, not gpt5's
        for r in out:
            assert r["grader_model_output_rw_probe_probe"] == 0
            assert r["relevance_score_rw_probe_probe"] == 3
            assert r["T5_model_output_rw_probe_probe"] == 0

    def test_underivable_flag_field_is_hard_error(self):
        with pytest.raises(SystemExit, match="flag-field"):
            eval_e2e.base_arm_from_flag_field("custom_flag_column")


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


# ---------------------------------------------------------------------------
# 7: zero-token prompts must be a named error up front, not an IndexError
#    part-way through a GPU extraction run
# ---------------------------------------------------------------------------


class TestEmptyPromptGuard:
    def _extractor_stub(self):
        import torch.nn as nn
        from types import SimpleNamespace

        from rewriter.routing_probe.extraction import ActivationExtractor

        ex = ActivationExtractor.__new__(ActivationExtractor)  # no 4B model load
        model = nn.Linear(2, 2)  # supplies .parameters() for input_device
        model.config = SimpleNamespace(hidden_size=8)
        ex.model = model

        class FakeTok:
            pad_token_id = 0

            def __call__(self, prompts, **kw):
                # Qwen behavior: "" tokenizes to [] (no BOS)
                return {"input_ids": [[1, 2] if p else [] for p in prompts]}

        ex.tokenizer = FakeTok()
        return ex

    def test_empty_prompt_is_named_error_before_forward(self):
        ex = self._extractor_stub()
        with pytest.raises(ValueError, match="zero tokens"):
            ex.extract_layers(["a fine prompt", "", "another"], layers=[0],
                              progress=False)


# ---------------------------------------------------------------------------
# 12: is_reasoning_model matches explicit families, not any name starting "o"
# ---------------------------------------------------------------------------


class TestReasoningModelPrefixes:
    def test_reasoning_families_match(self):
        from rewriter.grader_transport import is_reasoning_model

        for m in ("o1", "o1-preview", "o3", "o3-mini", "o4-mini",
                  "gpt-5", "gpt-5-mini", "O3-MINI"):
            assert is_reasoning_model(m), m

    def test_non_reasoning_o_names_do_not_match(self):
        from rewriter.grader_transport import is_reasoning_model

        for m in ("oai-gpt-4.1", "omega-chat", "o365-grader", "gpt-4o",
                  "gpt-4o-mini", "olmo-2"):
            assert not is_reasoning_model(m), m


# ---------------------------------------------------------------------------
# 13: corrupt dataset lines are a hard error, not a silent row drop
# ---------------------------------------------------------------------------


class TestGepaLoadJsonl:
    def test_corrupt_line_is_hard_error(self, tmp_path):
        pytest.importorskip("dspy")  # harness imports dspy at module level
        from rewriter.repro.gepa.harness import load_jsonl

        p = tmp_path / "dataset.jsonl"
        p.write_text('{"ok": 1}\n{"truncated": \n{"ok": 2}\n')
        with pytest.raises(ValueError, match="corrupt JSONL"):
            load_jsonl(str(p))

    def test_blank_lines_still_skipped(self, tmp_path):
        pytest.importorskip("dspy")
        from rewriter.repro.gepa.harness import load_jsonl

        p = tmp_path / "dataset.jsonl"
        p.write_text('{"ok": 1}\n\n{"ok": 2}\n')
        assert len(load_jsonl(str(p))) == 2


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


# ---------------------------------------------------------------------------
# 5: the grader disk cache must not be shared across transports
# ---------------------------------------------------------------------------


class TestTransportScopedGraderCache:
    def test_openai_transport_scopes_cache_key(self, monkeypatch):
        import sys as _sys

        _sys.path.insert(0, str(REPO / "Graders"))
        graders_core = pytest.importorskip("graders.core")
        import rewriter.grader_transport as gt

        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.delenv("GRADERS_DEPLOYMENT", raising=False)
        stock = getattr(graders_core, "_unscoped_cache_key", None) or graders_core._cache_key
        azure_key = stock("gpt-4.1", "tmpl", "sample")

        gt.install()  # no deployment override: transport alone must scope
        openai_key = graders_core._cache_key("gpt-4.1", "tmpl", "sample")
        assert openai_key != azure_key

        monkeypatch.setenv("GRADERS_DEPLOYMENT", "gpt-4o-mini")
        gt.install()  # deployment override scopes further, within the transport
        overridden_key = graders_core._cache_key("gpt-4.1", "tmpl", "sample")
        assert overridden_key not in (azure_key, openai_key)

        gt.install()  # repeated install: stable, no wrapper stacking
        assert graders_core._cache_key("gpt-4.1", "tmpl", "sample") == overridden_key
