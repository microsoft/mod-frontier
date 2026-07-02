#!/usr/bin/env python3
"""End-to-end evaluation of the rewrite stage on a graded JSONL file.

Stages (subcommands), in run order after ``rewriter/pipeline.py`` produced a
rewrites file. Each stage reuses this repository's own components -- the T5
helper from ``moderation/inference_field.py``, the grader specs in
``Graders/``, and ``compute_metrics`` from ``metrics/calculate_metrics.py``
-- so the rewrite stage is scored by exactly the published pipeline:

* ``t5-rewrites``  -- re-screen the rewrites with the ToxicChat T5 filter
  (GPU). Block-after-rewrite means the rewritten response is still flagged.
* ``t5-prompts``   -- T5 over the user prompts of every row (GPU); produces
  the ``T5_user_input`` column for the prompt-filter scenarios.
* ``grade``        -- toxicity_v10 (response harm, 0/1) + relevance_v01
  (1/2/3, refusal-aware) over the rewrites (API).
* ``grade-prompts``-- toxicity_v10 over the user prompts of every row (API);
  produces ``grader_user_input`` (the FP-rate reference).
* ``assemble``     -- merge everything into the evaluation JSONL as new
  columns (see below).
* ``metrics``      -- the end-to-end table (full universe + T5-flagged
  subset, x prompt/response/both filter scenarios) with bootstrap 95% CIs.

Columns written by ``assemble`` for arm ``<arm>`` (default ``rw_probe_probe``),
following the file's ``<base_field>_<system>`` naming convention:

* ``model_output_<arm>``          -- the rewrite/refusal text (flagged rows).
* ``grader_model_output_<arm>``   -- response-harm label of what the system
  shows: the rewrite's grade on flagged rows, the original response's grade
  on pass-through rows.
* ``relevance_score_<arm>``       -- same pass-through semantics.
* ``T5_model_output_<arm>``       -- T5 flag of what the system shows;
  a flagged row whose rewrite is still T5-positive stays blocked.
* ``T5_user_input``, ``grader_user_input`` -- prompt-level fields required by
  ``metrics/calculate_metrics.py`` (prompt scenarios + FP rate), computed
  once per file (arm-independent).

Grader auth: with Azure AD available the stock ``Graders`` client is used;
otherwise set ``GRADERS_AUTH=openai`` (and ``OPENAI_API_KEY``) to route the
same specs through api.openai.com (see ``rewriter/grader_transport.py``).

Example (the 30-row reproduction check):

    python rewriter/pipeline.py -i data/toxicchat_with_GPT5Response.jsonl \
        -o /tmp/rw.jsonl --limit 30
    python rewriter/eval_e2e.py t5-rewrites --rewrites /tmp/rw.jsonl -o /tmp/rw_t5.json
    GRADERS_AUTH=openai python rewriter/eval_e2e.py grade \
        -i data/toxicchat_with_GPT5Response.jsonl --rewrites /tmp/rw.jsonl -o /tmp/rw_grades.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Default arm suffix (probe intent + probe domain routing).
DEFAULT_ARM = "rw_probe_probe"

#: Bootstrap parameters (percentile CIs by row resampling).
BOOTSTRAP_SEED = 42
N_BOOT = 2000

RATE_KEYS = ["block_rate", "fp_rate", "harmful_rate", "not_useful_rate", "usefulness_rate"]


# ---------------------------------------------------------------------------
# Imports from this repository's other stages (by path; moderation/ and
# metrics/ are script dirs, not packages)
# ---------------------------------------------------------------------------


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def t5_helpers():
    """(run_t5_on_texts, MODEL_CHECKPOINT, TOKENIZER_NAME) from moderation/."""
    mod = _load_module("moderation/inference_field.py", "modfrontier_inference_field")
    return mod.run_t5_on_texts, mod.MODEL_CHECKPOINT, mod.TOKENIZER_NAME


def compute_metrics_fn():
    """``compute_metrics`` from metrics/calculate_metrics.py."""
    return _load_module("metrics/calculate_metrics.py", "modfrontier_metrics").compute_metrics


def graders_core(auth: str | None = None):
    """Import ``graders.core`` from Graders/, optionally with OpenAI transport."""
    graders_dir = str(REPO_ROOT / "Graders")
    if graders_dir not in sys.path:
        sys.path.insert(0, graders_dir)
    from graders import core

    if (auth or os.environ.get("GRADERS_AUTH", "")).lower() == "openai":
        from rewriter.grader_transport import install

        install()
        print("Grader transport: plain OpenAI (OPENAI_API_KEY)", flush=True)
    return core


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_jsonl_by_index(path: str) -> dict[int, dict]:
    return {int(r["index"]): r for r in load_rows(path)}


def write_json_atomic(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def flagged_indices(rows: list[dict], flag_field: str) -> list[int]:
    return [i for i, r in enumerate(rows) if str(r.get(flag_field)) == "1"]


def as_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _t5_model():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    run_t5, checkpoint, tok_name = t5_helpers()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"T5 device: {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(tok_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    model.eval()
    return run_t5, model, tok, device


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_t5_rewrites(args) -> None:
    """T5 re-screen of the rewrites -> ``{index: 0/1}``."""
    rewrites = load_rows(args.rewrites)
    col = f"model_output_{args.arm}"
    texts = [str(r.get(col, "") or "") for r in rewrites]
    run_t5, model, tok, device = _t5_model()
    preds = run_t5(texts, model, tok, device, batch_size=args.batch_size)
    out = {str(r["index"]): int(p) for r, p in zip(rewrites, preds)}
    write_json_atomic(out, args.output)
    print(f"T5 positives on rewrites: {sum(preds)}/{len(preds)} -> {args.output}", flush=True)


def cmd_t5_prompts(args) -> None:
    """T5 over every row's user prompt -> ``{index: 0/1}``."""
    rows = load_rows(args.input)
    texts = [str(r[args.prompt_field]) for r in rows]
    run_t5, model, tok, device = _t5_model()
    preds = run_t5(texts, model, tok, device, batch_size=args.batch_size)
    write_json_atomic({str(i): int(p) for i, p in enumerate(preds)}, args.output)
    print(f"T5 positives on prompts: {sum(preds)}/{len(preds)} -> {args.output}", flush=True)


def cmd_grade(args) -> None:
    """toxicity_v10 + relevance_v01 over the rewrites -> grades JSONL."""
    import asyncio

    core = graders_core(args.grader_auth)
    data = load_rows(args.input)
    rewrites = load_rows(args.rewrites)
    col = f"model_output_{args.arm}"
    responses = [str(r.get(col, "") or "") for r in rewrites]
    prompts = [str(data[r["index"]][args.prompt_field]) for r in rewrites]

    async def run():
        tox_spec = core.load_spec("toxicity_v10")
        rel_spec = core.load_spec("relevance_v01")
        print(f"Grading {len(responses)} rewrites (toxicity_v10 + relevance_v01)", flush=True)
        tox = await core.run_grader(tox_spec, responses, workers=args.workers)
        rel = await core.run_grader(
            rel_spec,
            [{"prompt": p, "response": r} for p, r in zip(prompts, responses)],
            workers=args.workers,
        )
        return tox, rel

    tox, rel = asyncio.run(run())
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r, t, rl in zip(rewrites, tox, rel):
            f.write(json.dumps({
                "index": r["index"],
                "conv_id": r.get("conv_id"),
                "toxicity_label": t,
                "relevance_score": rl,
            }) + "\n")
    os.replace(tmp, args.output)
    pos = sum(1 for t in tox if t == 1)
    print(f"Toxicity positives {pos}/{len(tox)} "
          f"(unparsed tox={sum(t is None for t in tox)}, rel={sum(r is None for r in rel)}) "
          f"-> {args.output}", flush=True)


def cmd_grade_prompts(args) -> None:
    """toxicity_v10 over every row's user prompt -> ``{index: 0/1}``."""
    import asyncio

    core = graders_core(args.grader_auth)
    rows = load_rows(args.input)
    texts = [str(r[args.prompt_field]) for r in rows]

    async def run():
        spec = core.load_spec("toxicity_v10")
        print(f"Grading {len(texts)} user prompts with toxicity_v10", flush=True)
        return await core.run_grader(spec, texts, workers=args.workers)

    labels = asyncio.run(run())
    out = {str(i): (int(l) if l is not None else None) for i, l in enumerate(labels)}
    write_json_atomic(out, args.output)
    print(f"Prompt-harm positives: {sum(1 for v in out.values() if v == 1)}/{len(out)} "
          f"(unparsed {sum(1 for v in out.values() if v is None)}) -> {args.output}", flush=True)


def cmd_assemble(args) -> None:
    """Merge rewrites + grades + T5 results into the evaluation JSONL columns."""
    rows = load_rows(args.input)
    flagged = set(flagged_indices(rows, args.flag_field))
    rewrites = load_jsonl_by_index(args.rewrites)
    grades = load_jsonl_by_index(args.grades)
    t5_rw = json.load(open(args.t5_rewrites, encoding="utf-8"))
    t5_prompts = json.load(open(args.t5_prompts, encoding="utf-8"))
    grader_prompts = json.load(open(args.grade_prompts, encoding="utf-8"))

    arm = args.arm
    text_col = f"model_output_{arm}"
    harm_col = f"grader_model_output_{arm}"
    rel_col = f"relevance_score_{arm}"
    t5_col = f"T5_model_output_{arm}"

    n_rw = 0
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            out = dict(row)
            out["T5_user_input"] = int(t5_prompts.get(str(i), 0))
            gui = grader_prompts.get(str(i))
            out["grader_user_input"] = int(gui) if gui is not None else 0
            if i in flagged and i in rewrites:
                n_rw += 1
                g = grades.get(i, {})
                harm = g.get("toxicity_label")
                rel = g.get("relevance_score")
                out[text_col] = rewrites[i].get(text_col)
                # Conservative fills for grader failures: harm None -> 0,
                # relevance None -> 1, missing T5 re-screen -> still blocked.
                out[harm_col] = 0 if harm is None else int(harm)
                out[rel_col] = 1 if rel is None else int(rel)
                out[t5_col] = int(t5_rw.get(str(i), 1))
            else:
                # Pass-through: the system shows the original response.
                out[harm_col] = as_int(row.get("grader_model_output_gpt5"), 0)
                raw_rel = row.get("relevance_score_gpt5")
                out[rel_col] = None if raw_rel is None else as_int(raw_rel, None)
                out[t5_col] = as_int(row.get("T5_model_output_gpt5"), 0)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    os.replace(tmp, args.output)
    print(f"Assembled {len(rows)} rows ({n_rw} rewritten) -> {args.output}", flush=True)


def _bootstrap_cis(rows, blocked_fn, compute_metrics, n_boot=N_BOOT, seed=BOOTSTRAP_SEED):
    """Percentile 95% CIs for each rate via row resampling."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(rows)
    if n == 0:
        return {k: [0.0, 0.0] for k in RATE_KEYS}
    samples = {k: [] for k in RATE_KEYS}
    idx_all = np.arange(n)
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        m = compute_metrics([rows[i] for i in idx], blocked_fn=blocked_fn)
        for k in RATE_KEYS:
            samples[k].append(m[k])
    return {
        k: [float(np.percentile(samples[k], 2.5)), float(np.percentile(samples[k], 97.5))]
        for k in RATE_KEYS
    }


def cmd_metrics(args) -> None:
    """End-to-end tables for the arm and the un-rewritten baselines."""
    compute_metrics = compute_metrics_fn()
    rows = load_rows(args.input)
    flagged_idx = flagged_indices(rows, args.flag_field)

    def canonical(arm: str | None) -> list[dict]:
        """Rows in calculate_metrics' field vocabulary; ``arm=None`` = original."""
        suffix = arm if arm else "gpt5"
        out = []
        for i, r in enumerate(rows):
            rel = r.get(f"relevance_score_{suffix}")
            out.append({
                "grader_user_input": as_int(r.get("grader_user_input"), 0),
                "grader_model_output": as_int(r.get(f"grader_model_output_{suffix}"), 0),
                "relevance_score": None if rel is None else as_int(rel, None),
                "T5_user_input": as_int(r.get("T5_user_input"), 0),
                "T5_model_output": as_int(r.get(f"T5_model_output_{suffix}"), 0),
            })
        return out

    scenarios = {
        "prompt": lambda r: r.get("T5_user_input", 0) == 1,
        "response": lambda r: r.get("T5_model_output", 0) == 1,
        "both": lambda r: r.get("T5_user_input", 0) == 1 or r.get("T5_model_output", 0) == 1,
    }

    result = {"meta": {"n_total": len(rows), "n_flagged": len(flagged_idx),
                       "seed": BOOTSTRAP_SEED, "n_boot": args.n_boot,
                       "flag_field": args.flag_field},
              "arms": {}}
    for name, arm in (("no_rewrite", None), (args.arm, args.arm)):
        crows = canonical(arm)
        frows = [crows[i] for i in flagged_idx]
        entry = {"full": {}, "flagged": {}}
        for sc, fn in scenarios.items():
            for split_name, split_rows in (("full", crows), ("flagged", frows)):
                m = compute_metrics(split_rows, blocked_fn=fn)
                if args.n_boot:
                    ci = _bootstrap_cis(split_rows, fn, compute_metrics, n_boot=args.n_boot)
                    for k in RATE_KEYS:
                        m[f"{k}_ci"] = ci[k]
                entry[split_name][sc] = m
        result["arms"][name] = entry
        f_resp = entry["flagged"]["response"]
        print(f"[{name}] flagged/response: block={f_resp['block_rate']:.4f} "
              f"({f_resp['blocked']}/{f_resp['total']}) "
              f"harmful_shown={f_resp['harmful_in_shown']}/{f_resp['shown']} "
              f"useful={f_resp['usefulness_rate']:.4f} ({f_resp['useful']}/{f_resp['total']})",
              flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote metrics -> {args.output}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p, need_input=True):
        if need_input:
            p.add_argument("-i", "--input", required=True, help="Evaluation JSONL")
        p.add_argument("--arm", default=DEFAULT_ARM)
        p.add_argument("--prompt-field", default="user_input")
        p.add_argument("--flag-field", default="T5_model_output_gpt5")

    p = sub.add_parser("t5-rewrites", help="T5 re-screen of the rewrites")
    common(p, need_input=False)
    p.add_argument("--rewrites", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.set_defaults(fn=cmd_t5_rewrites)

    p = sub.add_parser("t5-prompts", help="T5 over every row's user prompt")
    common(p)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.set_defaults(fn=cmd_t5_prompts)

    p = sub.add_parser("grade", help="Grade the rewrites (toxicity_v10 + relevance_v01)")
    common(p)
    p.add_argument("--rewrites", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--grader-auth", choices=["azure", "openai"], default=None)
    p.set_defaults(fn=cmd_grade)

    p = sub.add_parser("grade-prompts", help="toxicity_v10 over every user prompt")
    common(p)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--grader-auth", choices=["azure", "openai"], default=None)
    p.set_defaults(fn=cmd_grade_prompts)

    p = sub.add_parser("assemble", help="Merge outputs into the evaluation JSONL")
    common(p)
    p.add_argument("--rewrites", required=True)
    p.add_argument("--grades", required=True)
    p.add_argument("--t5-rewrites", required=True)
    p.add_argument("--t5-prompts", required=True)
    p.add_argument("--grade-prompts", required=True)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(fn=cmd_assemble)

    p = sub.add_parser("metrics", help="End-to-end metric tables with bootstrap CIs")
    common(p)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.set_defaults(fn=cmd_metrics)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
