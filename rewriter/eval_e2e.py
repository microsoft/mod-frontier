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
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if __package__ in (None, ""):  # invoked by path: python rewriter/eval_e2e.py
    sys.path.insert(0, str(REPO_ROOT))

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


def _duplicate_index_error(path: str, index: int) -> SystemExit:
    return SystemExit(
        f"{path}: duplicate record for index {index} — the file mixes two runs' "
        "outputs (an appended re-run or concatenated shards?); one run's grade "
        "would silently attach to another run's text. Regenerate the file"
    )


def load_jsonl_by_index(path: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for r in load_rows(path):
        i = int(r["index"])
        if i in out:
            raise _duplicate_index_error(path, i)
        out[i] = r
    return out


def require_unique_indices(records: list[dict], path: str) -> None:
    """Hard-error on duplicate ``index`` values in an order-preserving list."""
    seen: set[int] = set()
    for r in records:
        i = int(r["index"])
        if i in seen:
            raise _duplicate_index_error(path, i)
        seen.add(i)


def text_sha256(text) -> str:
    """Identity hash of a rewrite text (binds a grade record to what it graded)."""
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


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


def require_arm_column(rewrites: list[dict], col: str, path: str) -> None:
    """Hard-error when the arm's text column is absent from every record.

    A wrong ``--arm`` would otherwise silently screen/grade empty strings,
    which read as safe -- a fail-open path.
    """
    if rewrites and not any(col in r for r in rewrites):
        raise SystemExit(
            f"{path}: no record has the column {col!r} — wrong --arm? "
            f"Columns present: {sorted(k for k in rewrites[0] if k.startswith('model_output'))}"
        )


def rewrite_ok(record: dict, col: str) -> bool:
    """True iff the rewrite record is usable: successful and non-empty text."""
    return bool(record.get("rw_success")) and bool(str(record.get(col) or "").strip())


def base_arm_from_flag_field(flag_field: str) -> str:
    """Baseline arm suffix implied by ``--flag-field`` (``T5_model_output_gpt5`` -> ``gpt5``).

    The pass-through/baseline columns must come from the same base arm the
    flag did: hardcoding ``gpt5`` would silently populate pass-through rows
    with one baseline's values while flagging against another — an incoherent
    hybrid of two baselines with no error.
    """
    prefix = "T5_model_output_"
    base = flag_field[len(prefix):] if flag_field.startswith(prefix) else ""
    if not base:
        raise SystemExit(
            f"cannot derive the baseline arm from --flag-field {flag_field!r} "
            "(expected the shape T5_model_output_<base>); the pass-through/"
            "baseline columns would be ambiguous"
        )
    return base


def require_columns(rows: list[dict], cols: tuple[str, ...], path: str, hint: str) -> None:
    """Hard-error when any of ``cols`` is absent from every row.

    ``as_int(missing, 0)`` would otherwise silently read the whole column as
    0 — which for the safety columns is the fail-open, better-than-real fill.
    """
    for col in cols:
        if rows and not any(col in r for r in rows):
            raise SystemExit(
                f"{path}: no row has the column {col!r} — {hint}. Refusing to "
                "default it to 0 on every row (fail-open)"
            )


def cmd_t5_rewrites(args) -> None:
    """T5 re-screen of the rewrites -> ``{index: 0/1}``."""
    rewrites = load_rows(args.rewrites)
    require_unique_indices(rewrites, args.rewrites)
    col = f"model_output_{args.arm}"
    require_arm_column(rewrites, col, args.rewrites)
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


def pair_prompts(data: list[dict], rewrites: list[dict], prompt_field: str) -> list[str]:
    """Pair each rewrite record to its user prompt, by conv_id.

    conv_id pairing survives a filtered or re-sorted data file; the recorded
    positional index is only a fallback, and any conv_id disagreement is a
    hard error — the graders must never silently score against the wrong
    prompt (relevance would be garbage with no warning).
    """
    by_conv: dict = {}
    dup_conv: set = set()
    for j, row in enumerate(data):
        c = row.get("conv_id")
        if c in by_conv:
            dup_conv.add(c)
        elif c is not None:
            by_conv[c] = j
    prompts: list[str] = []
    n_moved = 0
    for r in rewrites:
        c, idx = r.get("conv_id"), int(r["index"])
        if c is not None and c in by_conv and c not in dup_conv:
            j = by_conv[c]
            if j != idx:
                n_moved += 1
            prompts.append(str(data[j][prompt_field]))
        else:
            if idx >= len(data):
                raise SystemExit(
                    f"rewrite index {idx} (conv_id {c!r}) not found in the data file: "
                    "it does not match the file pipeline.py was run on"
                )
            if c is not None and data[idx].get("conv_id") != c:
                raise SystemExit(
                    f"conv_id mismatch at index {idx}: rewrites say {c!r}, "
                    f"data file says {data[idx].get('conv_id')!r} — it does not "
                    "match the file pipeline.py was run on"
                )
            prompts.append(str(data[idx][prompt_field]))
    if n_moved:
        print(f"conv_id pairing: {n_moved} rewrites matched at a different row "
              f"position than recorded (re-sorted/filtered data file)", flush=True)
    return prompts


def cmd_grade(args) -> None:
    """toxicity_v10 + relevance_v01 over the rewrites -> grades JSONL."""
    import asyncio

    core = graders_core(args.grader_auth)
    data = load_rows(args.input)
    rewrites = load_rows(args.rewrites)
    require_unique_indices(rewrites, args.rewrites)
    col = f"model_output_{args.arm}"
    require_arm_column(rewrites, col, args.rewrites)
    responses = [str(r.get(col, "") or "") for r in rewrites]
    prompts = pair_prompts(data, rewrites, args.prompt_field)

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
        for r, resp, t, rl in zip(rewrites, responses, tox, rel):
            f.write(json.dumps({
                "index": r["index"],
                "conv_id": r.get("conv_id"),
                # Bind the grade to the exact text it graded: assemble refuses
                # to attach this record to any other rewrite text.
                "text_sha256": text_sha256(resp),
                "toxicity_label": t,
                "relevance_score": rl,
            }, ensure_ascii=False) + "\n")
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
    base = base_arm_from_flag_field(args.flag_field)
    text_col = f"model_output_{arm}"
    harm_col = f"grader_model_output_{arm}"
    rel_col = f"relevance_score_{arm}"
    t5_col = f"T5_model_output_{arm}"

    # Pass-through rows show the base arm's original response, so its grade
    # and T5 columns must exist on the input (fail-closed: a wrong input file
    # or flag field must not read as all-zeros = all-safe).
    require_columns(
        rows, (f"grader_model_output_{base}", f"T5_model_output_{base}"),
        args.input, "wrong --flag-field base, or not a graded evaluation file",
    )

    # Prompt-level files must cover every row: absent indices would silently
    # default to unflagged/harmless, under-blocking the prompt and both-filter
    # scenarios and corrupting the FP-rate reference (fail-open) — unlike the
    # fail-closed handling of rewrite grades and T5 re-screens below.
    for label, path, d in (("--t5-prompts", args.t5_prompts, t5_prompts),
                           ("--grade-prompts", args.grade_prompts, grader_prompts)):
        missing = [i for i in range(len(rows)) if str(i) not in d]
        if missing:
            raise SystemExit(
                f"{path} covers {len(rows) - len(missing)}/{len(rows)} rows "
                f"(e.g. missing indices {missing[:5]}) — {label} was produced "
                "from a different or truncated dataset; re-run that stage over "
                "this exact input file"
            )

    # A flagged row's rewrite must exist, be marked successful, be non-empty,
    # and carry a parseable harm grade -- anything less FAILS CLOSED: the row
    # keeps main's block (T5 flag = 1) instead of silently becoming a shown,
    # "harmless" row. Missing grade records are a hard error, not a fill.
    missing_grades = [i for i in sorted(flagged)
                      if i in rewrites and rewrite_ok(rewrites[i], text_col)
                      and i not in grades]
    if missing_grades:
        raise SystemExit(
            f"{len(missing_grades)} successful rewrites have no grade record "
            f"(e.g. indices {missing_grades[:5]}); run the grade stage over the "
            "same rewrites file before assembling"
        )

    n_rw = n_failed = n_harm_unparsed = 0
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            out = dict(row)
            # Strip this arm's columns before re-adding: a re-assemble over an
            # already-assembled file must not leave stale text/grades behind
            # (e.g. a rewrite that succeeded before but fails closed now).
            for col in (text_col, harm_col, rel_col, t5_col):
                out.pop(col, None)
            out["T5_user_input"] = int(t5_prompts[str(i)])
            gui = grader_prompts[str(i)]  # coverage checked above; None = unparsed
            out["grader_user_input"] = int(gui) if gui is not None else 0
            rw = rewrites.get(i)
            if i in flagged and rw is not None and rewrite_ok(rw, text_col):
                if rw.get("conv_id") is not None and rw.get("conv_id") != row.get("conv_id"):
                    raise SystemExit(
                        f"conv_id mismatch at index {i}: rewrites say "
                        f"{rw.get('conv_id')!r}, input says {row.get('conv_id')!r}"
                    )
                n_rw += 1
                g = grades[i]
                # The grade must be bound to the exact text it graded: index and
                # conv_id both survive a re-run of pipeline.py (same flagged rows,
                # different rewrite text), so only the text hash catches stale
                # grades — which would otherwise attach the OLD text's harm label
                # to the NEW text, a fail-open outcome.
                g_hash = g.get("text_sha256")
                if g_hash is None:
                    raise SystemExit(
                        f"grade record for index {i} has no text_sha256 binding "
                        "(produced by an older grade stage?) — it cannot be tied "
                        "to this rewrites file; re-run the grade stage"
                    )
                if g_hash != text_sha256(rw.get(text_col)):
                    raise SystemExit(
                        f"grade record for index {i} was produced from DIFFERENT "
                        "rewrite text than this rewrites file contains — stale "
                        "grades from a previous run? Re-run the grade stage over "
                        "this exact rewrites file"
                    )
                harm = g.get("toxicity_label")
                rel = g.get("relevance_score")
                out[text_col] = rw.get(text_col)
                # Fail-closed fills: an unparsed harm grade counts as HARMFUL
                # (never 0 -- 0 is the favorable fill, not the conservative
                # one); unparsed relevance -> 1 (not useful); missing T5
                # re-screen -> still blocked.
                if harm is None:
                    n_harm_unparsed += 1
                out[harm_col] = 1 if harm is None else int(harm)
                out[rel_col] = 1 if rel is None else int(rel)
                out[t5_col] = int(t5_rw.get(str(i), 1))
            else:
                if i in flagged and rw is not None:
                    n_failed += 1  # failed/empty rewrite: keep main's block
                # Pass-through: the system shows (or blocks) the original
                # response — the BASE arm's, resolved from --flag-field.
                out[harm_col] = as_int(row.get(f"grader_model_output_{base}"), 0)
                raw_rel = row.get(f"relevance_score_{base}")
                out[rel_col] = None if raw_rel is None else as_int(raw_rel, None)
                out[t5_col] = as_int(row.get(f"T5_model_output_{base}"), 0)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    os.replace(tmp, args.output)
    if n_failed:
        print(f"WARNING: {n_failed} flagged rows had failed/empty rewrites — "
              "they keep the original block (fail closed)", flush=True)
    if n_harm_unparsed:
        print(f"WARNING: {n_harm_unparsed} rewrites had unparsed harm grades — "
              "counted as harmful (fail closed)", flush=True)
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
    base = base_arm_from_flag_field(args.flag_field)

    # Fail closed on a non-assembled input or a mistyped --arm: every per-arm
    # column read below defaults to 0 via as_int, so a missing column would
    # otherwise yield a perfect-looking (block_rate 0, harmful 0) table with
    # no warning — the hard error the other subcommands already enforce.
    for suffix in (base, args.arm):
        require_columns(
            rows, (f"grader_model_output_{suffix}", f"T5_model_output_{suffix}"),
            args.input, f"not an assembled file, or wrong --arm/--flag-field "
                        f"(expected the {suffix!r} arm columns)",
        )
    require_columns(rows, ("grader_user_input", "T5_user_input"), args.input,
                    "not an assembled file (prompt-level columns absent)")

    def canonical(arm: str | None) -> list[dict]:
        """Rows in calculate_metrics' field vocabulary; ``arm=None`` = original."""
        suffix = arm if arm else base
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
                if args.n_boot and m:  # skip CI attachment on an empty subset
                    ci = _bootstrap_cis(split_rows, fn, compute_metrics, n_boot=args.n_boot)
                    for k in RATE_KEYS:
                        m[f"{k}_ci"] = ci[k]
                entry[split_name][sc] = m
        result["arms"][name] = entry

    # Write results BEFORE any summary printing: a print-time surprise must
    # not lose a (potentially 2000-resample) bootstrap run.
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote metrics -> {args.output}", flush=True)

    if not flagged_idx:
        print(f"NOTE: no rows matched --flag-field {args.flag_field!r}; "
              "flagged tables are empty", flush=True)
    for name, entry in result["arms"].items():
        f_resp = entry["flagged"]["response"]
        if "block_rate" not in f_resp:  # compute_metrics returns {} for zero rows
            print(f"[{name}] flagged/response: (empty subset)", flush=True)
            continue
        print(f"[{name}] flagged/response: block={f_resp['block_rate']:.4f} "
              f"({f_resp['blocked']}/{f_resp['total']}) "
              f"harmful_shown={f_resp['harmful_in_shown']}/{f_resp['shown']} "
              f"useful={f_resp['usefulness_rate']:.4f} ({f_resp['useful']}/{f_resp['total']})",
              flush=True)


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
