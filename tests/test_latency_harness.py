"""Latency-harness unit tests (no GPU, no server, no network)."""

import json
from pathlib import Path

from rewriter.repro.latency.measure_rewrite import REPO_ROOT, draw_sample, pct, stats

DATA = Path(REPO_ROOT) / "data" / "toxicchat_with_GPT5Response.jsonl"


class TestStats:
    def test_stats_fields(self):
        s = stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s["mean"] == 3.0
        assert s["median"] == 3.0
        assert s["min"] == 1.0 and s["max"] == 5.0

    def test_pct_nearest_rank(self):
        values = [float(i) for i in range(1, 11)]
        assert pct(values, 0.90) == 9.0
        assert pct(values, 0.0) == 1.0
        assert pct(values, 1.0) == 10.0

    def test_bimodal_median_below_mean(self):
        # the shape the README warns about: short mode + long tail
        vals = [0.3] * 20 + [5.0] * 10
        s = stats(vals)
        assert s["median"] < s["mean"]


class TestSampling:
    def test_deterministic_and_disjoint(self):
        idxs = list(range(230))
        w1, m1 = draw_sample(idxs, 3, 30)
        w2, m2 = draw_sample(idxs, 3, 30)
        assert (w1, m1) == (w2, m2)          # seed 42 fixed
        assert len(w1) == 3 and len(m1) == 30
        assert not set(w1) & set(m1)          # warm-up rows never measured

    def test_matches_flagged_universe(self):
        rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
        idxs = [i for i, r in enumerate(rows) if str(r.get("T5_model_output_gpt5")) == "1"]
        assert len(idxs) == 230
        warm, meas = draw_sample(idxs, 3, 30)
        assert all(i in idxs for i in warm + meas)


class TestModules:
    def test_measure_t5_importable(self):
        from rewriter.repro.latency import measure_t5

        assert measure_t5.BATCH_SIZE == 32
        assert measure_t5.MODEL_CHECKPOINT == "lmsys/toxicchat-t5-large-v1.0"
        assert Path(measure_t5.REPO_ROOT, "metrics", "calculate_metrics.py").exists()

    def test_default_data_path_exists(self):
        assert DATA.exists()

    def test_token_budget_matches_package(self):
        from rewriter.repro.latency.measure_rewrite import MAX_TOKENS
        from rewriter.rewrite import MAX_NEW_TOKENS

        assert MAX_TOKENS == MAX_NEW_TOKENS


class TestDocsScope:
    """The latency docs must describe rewrite-stage scope, not user wait.

    Regression for the re-screen-before-display correction: the harness
    measures conditional routing + rewrite generation after a response is
    flagged; under the paper's serving policy the first streamed token is
    not user-visible, so TTFT is a serving diagnostic.
    """

    README = Path(REPO_ROOT) / "rewriter" / "repro" / "latency" / "README.md"

    def _texts(self):
        import rewriter.repro.latency.measure_rewrite as mr

        return {
            "README": self.README.read_text(encoding="utf-8"),
            "docstring": mr.__doc__ or "",
        }

    def test_no_user_wait_claim(self):
        for name, text in self._texts().items():
            low = text.lower()
            assert "what a user waits" not in low, name
            assert "what a user would wait" not in low, name

    def test_no_ttft_user_visible_claim(self):
        for name, text in self._texts().items():
            low = text.lower()
            assert "users see output from ttft onward" not in low, name
            # TTFT must be framed as not user-visible under the paper's policy
            assert "not yet user-visible" in low, name
            assert "serving diagnostic" in low, name

    def test_scope_exclusions_stated(self):
        joined = {n: " ".join(t.lower().split()) for n, t in self._texts().items()}
        for name, low in joined.items():
            assert "re-screen" in low, name
            assert "excludes original-response generation" in low, name

    def test_e2e_s_defined_as_rewrite_stage(self):
        joined = {n: " ".join(t.lower().split()) for n, t in self._texts().items()}
        for name, low in joined.items():
            assert "route + prompt select" in low or "route + select" in low, name
            assert "not full conversation end-to-end" in low, name
