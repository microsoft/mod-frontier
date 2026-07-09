"""Regression tests for the PR #5 round-4 code-review fixes.

The agreed final punch-list, by round-4 finding id:

* #1  — T5 re-screen records are hash-bound to the exact rewrite text they
        screened (mirroring the grade records), so a stale ``t5_rewrites.json``
        can no longer attach an old text's flag to new rewrite text.
* #10 — the split-stage integrity gates (eval-set disjointness, group
        disjointness) raise ``SystemExit`` and therefore survive ``python -O``.
* #12 — the package docstring characterizes the stage's true end-to-end
        effect and labels the 78.3% -> 84.3% figure as the prompt ablation.
* #13 — the documented repro readers open the (now UTF-8) data file with an
        explicit encoding, so the README commands work on non-UTF-8 locales.
"""

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import rewriter
from rewriter import eval_e2e
from rewriter.repro import build_dataset

# Shared tiny-universe assemble helper + hash-bound fixtures.
from test_review_fixes import GOOD_GRADE, GOOD_RW, GOOD_T5, assemble, write_jsonl

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1: a T5 re-screen record is bound to the exact rewrite text it screened
# ---------------------------------------------------------------------------


class TestT5RescreenTextBinding:
    def test_stale_t5_flag_on_new_text_is_hard_error(self, tmp_path):
        # Same index (all the old format keyed on), different text: re-running
        # pipeline.py produces new rewrite text at the same flagged indices, so
        # a previous run's re-screen must refuse to attach its flag 0.
        stale = {"1": {"t5": 0,
                       "text_sha256": eval_e2e.text_sha256("an OLD rewrite")}}
        with pytest.raises(SystemExit, match="DIFFERENT rewrite text"):
            assemble(tmp_path, [GOOD_RW], [GOOD_GRADE], stale)

    def test_hashless_legacy_t5_record_is_hard_error(self, tmp_path):
        # A pre-round-4 flat {index: 0/1} file cannot be tied to any text.
        with pytest.raises(SystemExit, match="text_sha256"):
            assemble(tmp_path, [GOOD_RW], [GOOD_GRADE], {"1": 0})

    def test_matching_hash_unblocks(self, tmp_path):
        out = assemble(tmp_path, [GOOD_RW], [GOOD_GRADE], GOOD_T5)
        assert out[1]["T5_model_output_rw_probe_probe"] == 0

    def test_still_flagged_rewrite_stays_blocked(self, tmp_path):
        t5 = {"1": {"t5": 1,
                    "text_sha256": eval_e2e.text_sha256("a safe rewrite")}}
        out = assemble(tmp_path, [GOOD_RW], [GOOD_GRADE], t5)
        assert out[1]["T5_model_output_rw_probe_probe"] == 1

    def test_missing_record_still_fails_closed(self, tmp_path):
        # Absence keeps the block (fail closed) — only presence needs binding.
        out = assemble(tmp_path, [GOOD_RW], [GOOD_GRADE], {})
        assert out[1]["T5_model_output_rw_probe_probe"] == 1

    def test_cmd_t5_rewrites_emits_bound_records(self, tmp_path, monkeypatch):
        # The producer writes {index: {t5, text_sha256}} records that
        # round-trip through assemble's gate.
        write_jsonl(tmp_path / "rw.jsonl", [GOOD_RW])

        def fake_t5_model():
            run = lambda texts, model, tok, device, batch_size: [0] * len(texts)
            return run, None, None, "cpu"

        monkeypatch.setattr(eval_e2e, "_t5_model", fake_t5_model)
        args = argparse.Namespace(rewrites=str(tmp_path / "rw.jsonl"),
                                  arm="rw_probe_probe", batch_size=8,
                                  output=str(tmp_path / "t5.json"))
        eval_e2e.cmd_t5_rewrites(args)
        rec = json.loads((tmp_path / "t5.json").read_text())["1"]
        assert rec == {"t5": 0,
                       "text_sha256": eval_e2e.text_sha256("a safe rewrite")}
        out = assemble(tmp_path, [GOOD_RW], [GOOD_GRADE], {"1": rec})
        assert out[1]["T5_model_output_rw_probe_probe"] == 0


# ---------------------------------------------------------------------------
# 10: split-stage integrity gates survive python -O (SystemExit, not assert)
# ---------------------------------------------------------------------------


POOL_ROW = {"id": "bt_a", "source_prompt_hash": "h1", "golden_toxic": 0,
            "golden_severity": 0, "probe_decision": "REWRITE",
            "probe_domain": "other", "t5": 1, "tox_v10": 1}


def split_args(tmp_path, rows):
    write_jsonl(tmp_path / build_dataset.F_POOL_LABELED, rows)
    return argparse.Namespace(out_dir=str(tmp_path), seed=42)


class TestSplitIntegrityGates:
    def test_eval_overlap_is_systemexit(self, tmp_path):
        row = dict(POOL_ROW, in_excluded_eval=True)
        with pytest.raises(SystemExit, match="overlap the excluded eval"):
            build_dataset.cmd_split(split_args(tmp_path, [row]))

    def test_duplicate_group_is_systemexit(self, tmp_path):
        rows = [dict(POOL_ROW, id="bt_a"), dict(POOL_ROW, id="bt_b")]
        with pytest.raises(SystemExit, match="appear >1x"):
            build_dataset.cmd_split(split_args(tmp_path, rows))

    def test_gates_fire_under_pythonoptimize(self, tmp_path):
        # The whole point of the fix: `python -O` strips asserts, so the gate
        # must be a raise. Run the violating split in an optimized subprocess.
        write_jsonl(tmp_path / build_dataset.F_POOL_LABELED,
                    [dict(POOL_ROW, in_excluded_eval=True)])
        code = (
            f"import argparse, sys; sys.path.insert(0, {str(REPO)!r}); "
            "from rewriter.repro.build_dataset import cmd_split; "
            f"cmd_split(argparse.Namespace(out_dir={str(tmp_path)!r}, seed=42))"
        )
        r = subprocess.run([sys.executable, "-O", "-c", code],
                           capture_output=True, text=True)
        assert r.returncode != 0
        assert "overlap the excluded eval" in r.stderr


# ---------------------------------------------------------------------------
# 12: the package docstring must not sell the ablation delta as end-to-end
# ---------------------------------------------------------------------------


class TestPackageDocstring:
    def _paragraphs(self):
        return rewriter.__doc__.split("\n\n")

    def test_end_to_end_paragraph_characterizes_shipped_filter(self):
        e2e = next(p for p in self._paragraphs() if p.startswith("End-to-end"))
        # the true end-to-end effect: 100% block / usefulness 0% -> 84.3% / 12.2%
        assert "84.3" in e2e and "12.2" in e2e
        assert "0%" in e2e
        # the ablation arm's number must not appear as the end-to-end baseline
        assert "78.3" not in e2e

    def test_ablation_delta_is_labeled_as_the_prompt_ablation(self):
        para = next(p for p in self._paragraphs() if "78.3" in p)
        assert "ablation" in para or "optimization" in para


# ---------------------------------------------------------------------------
# 13: the documented repro readers open the UTF-8 data file explicitly
# ---------------------------------------------------------------------------


NON_ASCII_ROW = {"user_input": "¿Cómo estás? — céad míle fáilte, 你好",
                 "model_output_gpt5": "réponse sûre"}


def run_in_c_locale(code: str) -> subprocess.CompletedProcess:
    """Run a snippet with locale-default encoding forced to ASCII (the
    Windows/cp1252 failure mode, portably): LC_ALL=C plus UTF-8 mode off."""
    env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONUTF8="0")
    return subprocess.run([sys.executable, "-X", "utf8=0", "-c", code],
                          capture_output=True, text=True, env=env)


class TestUtf8DataReaders:
    def _data_file(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text(json.dumps(NON_ASCII_ROW, ensure_ascii=False) + "\n",
                     encoding="utf-8")
        return p

    def test_calculate_metrics_reader(self, tmp_path):
        p = self._data_file(tmp_path)
        code = (
            f"import sys; sys.path.insert(0, {str(REPO / 'metrics')!r}); "
            "from calculate_metrics import load_data; "
            f"rows = load_data({str(p)!r}); assert len(rows) == 1, rows"
        )
        r = run_in_c_locale(code)
        assert r.returncode == 0, r.stderr

    def test_build_dataset_reader(self, tmp_path):
        p = self._data_file(tmp_path)
        code = (
            f"import sys; sys.path.insert(0, {str(REPO)!r}); "
            "from rewriter.repro.build_dataset import read_jsonl; "
            f"rows = read_jsonl({str(p)!r}); assert len(rows) == 1, rows"
        )
        r = run_in_c_locale(code)
        assert r.returncode == 0, r.stderr

    def test_gepa_harness_reader_is_utf8(self):
        # harness.load_jsonl is the reader behind run_gepa.py --dataset.
        # Importing the module needs dspy/torch (not test deps), so pin the
        # fix at the AST level: every .open() inside load_jsonl must pass
        # encoding="utf-8".
        src = (REPO / "rewriter/repro/gepa/harness.py").read_text(encoding="utf-8")
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "load_jsonl")
        opens = [n for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "open"]
        assert opens, "load_jsonl no longer calls .open()?"
        for call in opens:
            kwargs = {k.arg: getattr(k.value, "value", None)
                      for k in call.keywords}
            assert kwargs.get("encoding") == "utf-8"
