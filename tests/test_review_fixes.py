"""Regression tests for the PR #5 code-review fixes.

Each test encodes a reviewed failure scenario; numbers refer to the review's
finding ids. The fail-closed tests are the load-bearing ones: a generation or
grading failure must never silently improve the safety metrics.
"""

import argparse
import json
from pathlib import Path

import pytest

from rewriter import eval_e2e
from rewriter.pipeline import load_routing_cache
from rewriter.rewrite import clean_completion, is_bare_refusal

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# helpers: build a tiny 3-row universe (1 flagged) + assemble inputs
# ---------------------------------------------------------------------------


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def base_rows():
    return [
        {"conv_id": "aaa", "user_input": "p0", "model_output_gpt5": "r0",
         "grader_model_output_gpt5": 0, "relevance_score_gpt5": 3, "T5_model_output_gpt5": 0},
        {"conv_id": "bbb", "user_input": "p1", "model_output_gpt5": "r1 flagged",
         "grader_model_output_gpt5": 1, "relevance_score_gpt5": 2, "T5_model_output_gpt5": 1},
        {"conv_id": "ccc", "user_input": "p2", "model_output_gpt5": "r2",
         "grader_model_output_gpt5": 0, "relevance_score_gpt5": None, "T5_model_output_gpt5": 0},
    ]


def assemble(tmp_path, rewrites, grades, t5_rw, out_name="out.jsonl", rows=None):
    d = tmp_path
    write_jsonl(d / "data.jsonl", rows or base_rows())
    write_jsonl(d / "rw.jsonl", rewrites)
    write_jsonl(d / "grades.jsonl", grades)
    (d / "t5_rw.json").write_text(json.dumps(t5_rw))
    (d / "t5_prompts.json").write_text(json.dumps({"0": 0, "1": 1, "2": 0}))
    (d / "grade_prompts.json").write_text(json.dumps({"0": 0, "1": 1, "2": 0}))
    args = argparse.Namespace(
        input=str(d / "data.jsonl"), rewrites=str(d / "rw.jsonl"),
        grades=str(d / "grades.jsonl"), t5_rewrites=str(d / "t5_rw.json"),
        t5_prompts=str(d / "t5_prompts.json"), grade_prompts=str(d / "grade_prompts.json"),
        output=str(d / out_name), arm="rw_probe_probe",
        prompt_field="user_input", flag_field="T5_model_output_gpt5",
    )
    eval_e2e.cmd_assemble(args)
    with open(d / out_name, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


GOOD_RW = {"index": 1, "conv_id": "bbb", "model_output_rw_probe_probe": "a safe rewrite",
           "rw_decision": "REWRITE", "rw_domain": "other", "rw_prompt_id": "gepa960/other",
           "rw_latency_s": 1.0, "rw_success": True}
GOOD_GRADE = {"index": 1, "conv_id": "bbb", "toxicity_label": 0, "relevance_score": 3}


class TestAssembleFailsClosed:
    """Review #2/#3: generation/grading failures must keep main's block."""

    def test_happy_path_unblocks(self, tmp_path):
        out = assemble(tmp_path, [GOOD_RW], [GOOD_GRADE], {"1": 0})
        assert out[1]["T5_model_output_rw_probe_probe"] == 0
        assert out[1]["grader_model_output_rw_probe_probe"] == 0
        assert out[1]["model_output_rw_probe_probe"] == "a safe rewrite"

    def test_empty_rewrite_stays_blocked(self, tmp_path):
        rw = dict(GOOD_RW, **{"model_output_rw_probe_probe": "", "rw_success": False})
        out = assemble(tmp_path, [rw], [], {"1": 0})
        # fail closed: the flagged row keeps the original block + grades
        assert out[1]["T5_model_output_rw_probe_probe"] == 1
        assert out[1]["grader_model_output_rw_probe_probe"] == 1
        assert "model_output_rw_probe_probe" not in out[1]

    def test_rw_success_false_stays_blocked_even_with_text(self, tmp_path):
        rw = dict(GOOD_RW, rw_success=False)
        out = assemble(tmp_path, [rw], [], {"1": 0})
        assert out[1]["T5_model_output_rw_probe_probe"] == 1

    def test_none_harm_grade_counts_as_harmful(self, tmp_path):
        grade = dict(GOOD_GRADE, toxicity_label=None)
        out = assemble(tmp_path, [GOOD_RW], [grade], {"1": 0})
        assert out[1]["grader_model_output_rw_probe_probe"] == 1  # NEVER 0

    def test_missing_grade_record_is_hard_error(self, tmp_path):
        with pytest.raises(SystemExit):
            assemble(tmp_path, [GOOD_RW], [], {"1": 0})

    def test_missing_t5_rescreen_stays_blocked(self, tmp_path):
        out = assemble(tmp_path, [GOOD_RW], [GOOD_GRADE], {})
        assert out[1]["T5_model_output_rw_probe_probe"] == 1

    def test_conv_id_mismatch_is_hard_error(self, tmp_path):
        rw = dict(GOOD_RW, conv_id="WRONG")
        with pytest.raises(SystemExit):
            assemble(tmp_path, [rw], [GOOD_GRADE], {"1": 0})

    def test_reassemble_strips_stale_columns(self, tmp_path):
        # first assemble: success; second over the assembled file with the
        # rewrite now failed -> the stale text/grades must not survive
        rows1 = assemble(tmp_path, [GOOD_RW], [GOOD_GRADE], {"1": 0})
        rw_failed = dict(GOOD_RW, **{"model_output_rw_probe_probe": "", "rw_success": False})
        rows2 = assemble(tmp_path, [rw_failed], [], {"1": 0},
                         out_name="out2.jsonl", rows=rows1)
        assert "model_output_rw_probe_probe" not in rows2[1]
        assert rows2[1]["T5_model_output_rw_probe_probe"] == 1


class TestPromptPairing:
    """Review #6: rewrites pair to prompts by conv_id, never silently by position."""

    def test_resorted_file_pairs_by_conv_id(self):
        data = list(reversed(base_rows()))  # bbb now at position 1... reversed: ccc,bbb,aaa
        prompts = eval_e2e.pair_prompts(data, [GOOD_RW], "user_input")
        assert prompts == ["p1"]  # found via conv_id despite the re-sort

    def test_index_fallback_mismatch_is_hard_error(self):
        data = base_rows()
        for r in data:
            del r["conv_id"]  # force index fallback
        rw = dict(GOOD_RW)  # carries conv_id 'bbb'; row 1 now has none
        with pytest.raises(SystemExit):
            eval_e2e.pair_prompts(data, [rw], "user_input")

    def test_index_out_of_range_is_hard_error(self):
        with pytest.raises(SystemExit):
            eval_e2e.pair_prompts(base_rows()[:1], [dict(GOOD_RW, conv_id="zzz")], "user_input")


class TestArmColumnGuard:
    """Below-cap: a wrong --arm must hard-error, not screen empty strings."""

    def test_missing_arm_column_raises(self):
        with pytest.raises(SystemExit):
            eval_e2e.require_arm_column(
                [{"index": 0, "model_output_rw_other_arm": "x"}], "model_output_rw_probe_probe", "f")

    def test_present_arm_column_passes(self):
        eval_e2e.require_arm_column([GOOD_RW], "model_output_rw_probe_probe", "f")


class TestMetricsRobustness:
    """Review #12: metrics must write output before printing and survive
    an empty flagged subset."""

    def test_empty_flagged_subset(self, tmp_path, capsys):
        rows = [dict(r, T5_model_output_gpt5=0) for r in base_rows()]
        # give every row the arm columns (pass-through shape)
        for r in rows:
            r.update({"grader_model_output_rw_probe_probe": r["grader_model_output_gpt5"],
                      "relevance_score_rw_probe_probe": r["relevance_score_gpt5"],
                      "T5_model_output_rw_probe_probe": 0,
                      "T5_user_input": 0, "grader_user_input": 0})
        write_jsonl(tmp_path / "assembled.jsonl", rows)
        args = argparse.Namespace(input=str(tmp_path / "assembled.jsonl"),
                                  output=str(tmp_path / "m.json"), arm="rw_probe_probe",
                                  prompt_field="user_input",
                                  flag_field="T5_model_output_gpt5", n_boot=0)
        eval_e2e.cmd_metrics(args)  # must not raise
        assert json.loads((tmp_path / "m.json").read_text())["meta"]["n_flagged"] == 0


class TestRoutingCacheProvenance:
    """Review #14: caches without threshold provenance (or at a different
    threshold) are hard errors."""

    def test_missing_meta_is_hard_error(self, tmp_path):
        p = tmp_path / "cache.json"
        p.write_text(json.dumps({"0": {"decision": "REWRITE", "domain": "other"}}))
        with pytest.raises(SystemExit):
            load_routing_cache(str(p), 0.161)

    def test_threshold_mismatch_is_hard_error(self, tmp_path):
        p = tmp_path / "cache.json"
        p.write_text(json.dumps({"_meta": {"threshold": 0.370},
                                 "0": {"decision": "REWRITE", "domain": "other"}}))
        with pytest.raises(SystemExit):
            load_routing_cache(str(p), 0.161)

    def test_matching_threshold_loads(self, tmp_path):
        p = tmp_path / "cache.json"
        p.write_text(json.dumps({"_meta": {"threshold": 0.161},
                                 "0": {"decision": "REWRITE", "domain": "other"}}))
        cache = load_routing_cache(str(p), 0.161)
        assert "_meta" not in cache and "0" in cache


class TestCompletionCleaning:
    """Review #1: with the correct decode slice there is no prompt echo, and
    a rewrite that legitimately contains a role-marker string must survive."""

    def test_role_marker_in_rewrite_is_preserved(self):
        text = ("Here is the safe transcript:\nuser\nhello\nassistant\n"
                "I rewrote the reply to remove the harmful part.")
        assert clean_completion(text) == text  # no marker-split truncation


class TestBareRefusalOpeners:
    """Below-cap: the retry must not replace short benign rewrites."""

    def test_benign_sorry_is_not_bare(self):
        assert not is_bare_refusal("Sorry, everyone deserves respect.")

    def test_compound_sorry_still_bare(self):
        assert is_bare_refusal("Sorry, but I can't help with that.")
        assert is_bare_refusal("I'm sorry, but I can't assist with that request.")
        assert is_bare_refusal("I can’t help with that request.")


class TestGoldenCacheRetries:
    """Review #4: API errors must never be persisted as labels."""

    def _stub_clients(self):
        class _TextBlock:
            type = "text"
            text = '{"domain": "other", "toxic": 1, "severity": 2, "rationale": "r"}'

        class _Msg:
            content = [_TextBlock()]

        class Boom:
            class messages:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("overloaded")

        class Ok:
            class messages:
                @staticmethod
                def create(**kw):
                    return _Msg()

        return Boom(), Ok()

    def test_error_not_cached_then_retried(self, tmp_path):
        from rewriter.repro.build_dataset import GoldenCache, label_one

        boom, ok = self._stub_clients()
        cache = GoldenCache(str(tmp_path / "golden_cache.json"))
        row = {"id": "row1", "prompt": "p", "response": "r"}

        first = label_one(boom, row, cache, "claude-opus-4-8")
        assert first["golden_toxic"] is None          # errored
        assert cache.get("row1") is None              # NOT persisted

        second = label_one(ok, row, cache, "claude-opus-4-8")
        assert second["golden_toxic"] == 1            # retried, not served from cache
        assert cache.get("row1")["golden_toxic"] == 1  # success IS cached

    def test_legacy_error_record_is_retried(self, tmp_path):
        from rewriter.repro.build_dataset import GoldenCache, label_one

        _, ok = self._stub_clients()
        cache = GoldenCache(str(tmp_path / "golden_cache.json"))
        cache.set("row1", {"golden_domain": None, "golden_toxic": None,
                           "golden_severity": None, "golden_rationale": "ERR: old"})
        row = {"id": "row1", "prompt": "p", "response": "r"}
        res = label_one(ok, row, cache, "claude-opus-4-8")
        assert res["golden_toxic"] == 1               # legacy ERR record not trusted


class TestSplitGuard:
    """Unresolved golden labels must hard-error split, not silently pass."""

    def test_unresolved_labels_block_split(self, tmp_path):
        from rewriter.repro.build_dataset import cmd_split

        rows = [{"id": "a", "source_prompt_hash": "h1", "probe_decision": "REWRITE",
                 "probe_domain": "other", "golden_toxic": None, "golden_severity": None}]
        write_jsonl(tmp_path / "pool_labeled.jsonl", rows)
        args = argparse.Namespace(out_dir=str(tmp_path), seed=42)
        with pytest.raises(SystemExit):
            cmd_split(args)


class TestRefusalDetector:
    """Review #7: bare "i don't"/"i do not" must not mark hedged rewrites."""

    def test_hedged_rewrites_not_refusals(self):
        from rewriter.repro.gepa.reward import RefusalDetector

        d = RefusalDetector()
        assert not d.is_refusal("I don't recommend doing X; instead, here's a safe approach.")
        assert not d.is_refusal("I do not have specific data on this, but here is context.")
        assert d.is_refusal("I can't help with that request.")
        assert d.is_refusal("I'm sorry, but I cannot assist with this.")


class TestGraderTransport:
    """Review #5 + below-cap: deployment override must scope the cache and
    the reasoning-model guard must look at the effective model."""

    def test_reasoning_guard_uses_effective_model(self):
        from rewriter.grader_transport import is_reasoning_model

        assert not is_reasoning_model("gpt-4.1")
        assert is_reasoning_model("gpt-5-mini")

    def test_deployment_scopes_cache_key(self, monkeypatch):
        import sys as _sys

        _sys.path.insert(0, str(REPO / "Graders"))
        graders_core = pytest.importorskip("graders.core")
        import rewriter.grader_transport as gt

        monkeypatch.setenv("OPENAI_API_KEY", "test")
        base = graders_core._cache_key("gpt-4.1", "tmpl", "sample")

        monkeypatch.setenv("GRADERS_DEPLOYMENT", "gpt-4o-mini")
        monkeypatch.setattr(graders_core, "_deployment_keyed_cache", None, raising=False)
        gt.install()
        overridden = graders_core._cache_key("gpt-4.1", "tmpl", "sample")
        assert overridden != base  # override-produced labels can't collide
