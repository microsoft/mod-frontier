#!/usr/bin/env python3
"""GEPA-optimize one rewrite/refusal prompt against the T5 composite reward.

Derived from the SafeFlow GEPA optimization experiments; this is the driver
that produced the shipped prompt packs in ``rewriter/prompts/`` (one run per
scope at the full 960-metric-call budget).

Scopes (from the dataset built by ``rewriter/repro/build_dataset.py``):
  * ``--scope <domain>``  — per-domain rewrite prompt (train/val = the rows
    whose ``probe_domain`` is that domain).
  * ``--scope unified``   — unified fallback rewrite prompt (train/val = all
    domains' train/val rows).
  * ``--scope refusal``   — contextual refusal prompt (train/val = an 80/20
    split of the ``refusal_pool`` rows; the covert-refusal penalty is
    disabled because refusals are the correct output there).

Seed prompts come from ``seeds/`` next to this file (``gepa_seed_rewrite.md``
for rewrite scopes, ``baseline_refusal.md`` for the refusal scope) unless
``--seed-prompt-file`` overrides them.

Outputs (under ``--out-dir/<scope>/``): ``seed_prompt.md``,
``optimized_prompt.md``, ``result.json`` (baseline vs optimized val scores,
per-example records, reward-component rates), ``reward_trajectory.jsonl``
(one line per metric call), and the GEPA log dir (``gepa_log/``, which holds
the cloudpickle round checkpoints that make an interrupted run resumable).

Usage (one GPU; Qwen3-4B task LM + T5 both fit)::

    python rewriter/repro/gepa/run_gepa.py --scope unified \
        --dataset data/rewrite_stage/dataset.jsonl --out-dir runs/gepa \
        --max-metric-calls 960 --num-threads 4 --wandb

    # cheap smoke (tiny splits, no harm veto):
    python rewriter/repro/gepa/run_gepa.py --scope unified \
        --dataset data/rewrite_stage/dataset.jsonl --out-dir runs/smoke \
        --max-metric-calls 12 --num-threads 3 --smoke --no-harm-veto

Run every scope in ``scopes.txt`` (from the split stage) as separate jobs,
one GPU each.

Credentials / environment:
  * ``OPENAI_API_KEY``    — gpt-5 reflection LM (required).
  * ``ANTHROPIC_API_KEY`` — claude-opus-4-8 harm veto (unless --no-harm-veto).
  * Graders auth per ``Graders/README.md`` (relevance_v01 + toxicity_v10).
  * With ``--wandb``: ``WANDB_PROJECT`` overrides the project;
    ``WANDB_RUN_ID`` + ``WANDB_RESUME=allow`` pin the run identity so a
    re-run resumes the same wandb run; ``WANDB_ENTITY`` sets the entity.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

try:
    from rewriter.repro.gepa import harness as H
    from rewriter.repro.gepa import reward as R
except ImportError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import harness as H  # type: ignore
    import reward as R  # type: ignore

SEED = 42
SEEDS_DIR = Path(__file__).resolve().parent / "seeds"
SEED_REWRITE_FILE = SEEDS_DIR / "gepa_seed_rewrite.md"
SEED_REFUSAL_FILE = SEEDS_DIR / "baseline_refusal.md"
DEFAULT_WANDB_PROJECT = "safeflow-rewriter-gepa"


def load_split(dataset_path: str, scope: str, seed: int = SEED):
    """Return (train_rows, val_rows) for a scope from the split-tagged dataset."""
    rows = H.load_jsonl(dataset_path)
    if scope == "refusal":
        pool = [r for r in rows if r.get("split") == "refusal_pool"]
        random.Random(seed).shuffle(pool)  # deterministic 80/20 split
        n_tr = int(0.8 * len(pool))
        return pool[:n_tr], pool[n_tr:]
    if scope == "unified":
        train = [r for r in rows if r.get("split") == "train"]
        val = [r for r in rows if r.get("split") == "val"]
        return train, val
    # per-domain
    train = [r for r in rows if r.get("split") == "train" and r.get("probe_domain") == scope]
    val = [r for r in rows if r.get("split") == "val" and r.get("probe_domain") == scope]
    return train, val


def eval_program(program, examples, metric_fn) -> dict:
    """Score a program over examples with the metric; return mean + per-example."""
    scores, recs = [], []
    for ex in examples:
        pred = program(user_prompt=ex.user_prompt, unsafe_draft=ex.unsafe_draft)
        o = metric_fn(ex, pred, None, None, None)
        s = float(o["score"]) if hasattr(o, "__getitem__") else float(o)
        scores.append(s)
        recs.append({
            "user_prompt": ex.user_prompt[:300],
            "unsafe_draft": ex.unsafe_draft[:300],
            "rewrite": getattr(pred, "safe_rewrite", "")[:600],
            "score": s,
            "feedback": (o["feedback"] if hasattr(o, "__getitem__") else "")[:400],
        })
    mean = sum(scores) / max(1, len(scores))
    return {"mean_score": mean, "n": len(scores), "per_example": recs}


def main() -> None:
    ap = argparse.ArgumentParser(description="GEPA-optimize one rewrite/refusal prompt")
    ap.add_argument("--scope", required=True, help="<domain>|unified|refusal")
    ap.add_argument("--dataset", required=True, help="dataset.jsonl from build_dataset.py split")
    ap.add_argument("--out-dir", required=True, help="output root (per-scope subdir created)")
    ap.add_argument("--max-metric-calls", type=int, default=960,
                    help="GEPA metric-call budget (960 = the shipped runs)")
    ap.add_argument("--max-train", type=int, default=48, help="max GEPA train examples")
    ap.add_argument("--max-val", type=int, default=24, help="max GEPA val examples")
    ap.add_argument("--num-threads", type=int, default=4, help="GEPA metric threads")
    ap.add_argument("--grader-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--task-model", default=H.DEFAULT_TASK_MODEL, help="HF task model id")
    ap.add_argument("--no-harm-veto", action="store_true",
                    help="disable the claude-opus reliable harm veto (cheap smokes only)")
    ap.add_argument("--seed-prompt-file", default=None,
                    help="override the default seed prompt for this scope")
    ap.add_argument("--smoke", action="store_true", help="tiny train/val for a quick check")
    ap.add_argument("--wandb", action="store_true", help="log the GEPA run to Weights & Biases")
    args = ap.parse_args()

    import dspy

    H.seed_everything(args.seed)
    is_refusal = args.scope == "refusal"

    out_dir = os.path.join(args.out_dir, args.scope)
    os.makedirs(out_dir, exist_ok=True)

    # data
    train_rows, val_rows = load_split(args.dataset, args.scope, seed=args.seed)
    train_rows = train_rows[: args.max_train]
    val_rows = val_rows[: args.max_val]
    if args.smoke:
        train_rows, val_rows = train_rows[:4], val_rows[:3]
    if not train_rows or not val_rows:
        raise SystemExit(f"[{args.scope}] insufficient data: "
                         f"train={len(train_rows)} val={len(val_rows)}")
    trainset = [R.make_example(r) for r in train_rows]
    valset = [R.make_example(r) for r in val_rows]
    print(f"[{args.scope}] train={len(trainset)} val={len(valset)}", flush=True)

    # seed prompt
    seed_file = Path(args.seed_prompt_file) if args.seed_prompt_file else (
        SEED_REFUSAL_FILE if is_refusal else SEED_REWRITE_FILE
    )
    seed_prompt = seed_file.read_text()
    with open(os.path.join(out_dir, "seed_prompt.md"), "w") as f:
        f.write(seed_prompt)

    # instruments + reward
    print(f"loading task LM ({args.task_model}) + T5 ...", flush=True)
    task_lm = H.load_task_model(args.task_model, max_new_tokens=1024,
                                temperature=0.0, dtype="bfloat16")
    dspy.configure(lm=task_lm)
    t5 = R.T5Scorer()
    graders = R.GraderScorer(workers=args.grader_workers)
    harm_judge = None if args.no_harm_veto else R.OpusHarmJudge()
    trajectory_path = os.path.join(out_dir, "reward_trajectory.jsonl")
    open(trajectory_path, "w").close()  # truncate any prior run's trajectory
    metric_fn = R.build_t5_composite_metric(
        t5=t5,
        graders=graders,
        harm_judge=harm_judge,                # reliable veto (claude-opus), not toxicity_v10 alone
        require_non_refusal=not is_refusal,   # refusal scope: refusals are correct
        use_relevance=True,
        wandb_log=args.wandb,
        trajectory_path=trajectory_path,
        summary_every=10,
    )

    reflection_lm = H.load_reflection_model(api_key=os.environ["OPENAI_API_KEY"])

    # baseline (seed) val score
    seed_program = H.WeakRewriter(seed_prompt)
    print(f"[{args.scope}] scoring seed prompt on val ...", flush=True)
    baseline = eval_program(seed_program, valset, metric_fn)
    print(f"[{args.scope}] baseline val mean = {baseline['mean_score']:.3f}", flush=True)

    # GEPA compile. Run identity can be pinned via env (WANDB_PROJECT /
    # WANDB_RUN_ID / WANDB_RESUME) so the run URL is reconstructible and a
    # re-run resumes the same wandb run.
    wandb_project = (os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT) \
        if args.wandb else None
    wandb_entity = None
    wandb_run_id = os.environ.get("WANDB_RUN_ID")
    wandb_url = None
    if args.wandb:
        try:
            import wandb

            wandb_entity = os.environ.get("WANDB_ENTITY") or wandb.Api().default_entity
            if wandb_entity and wandb_project and wandb_run_id:
                wandb_url = f"https://wandb.ai/{wandb_entity}/{wandb_project}/runs/{wandb_run_id}"
                print(f"[{args.scope}] wandb run URL: {wandb_url}", flush=True)
        except Exception as e:
            print(f"[{args.scope}] wandb entity lookup failed: {e!r}", flush=True)
    gepa = H.build_gepa(
        metric_fn=metric_fn,
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
        num_threads=args.num_threads,
        use_wandb=args.wandb,
        wandb_project=wandb_project,
        wandb_run_name=f"gepa-{args.scope}",
        log_dir=os.path.join(out_dir, "gepa_log"),
        seed=args.seed,
    )
    print(f"[{args.scope}] compiling GEPA (max_metric_calls={args.max_metric_calls}) ...",
          flush=True)
    optimized = gepa.compile(seed_program, trainset=trainset, valset=valset)

    optimized_prompt = H.get_optimized_prompt(optimized)
    with open(os.path.join(out_dir, "optimized_prompt.md"), "w") as f:
        f.write(optimized_prompt)

    # optimized val score
    print(f"[{args.scope}] scoring optimized prompt on val ...", flush=True)
    opt_eval = eval_program(optimized, valset, metric_fn)
    print(f"[{args.scope}] optimized val mean = {opt_eval['mean_score']:.3f} "
          f"(baseline {baseline['mean_score']:.3f})", flush=True)

    # final reward-component rates (exact; wandb summary lags by <= summary_every calls)
    reward_components = None
    snap = getattr(metric_fn, "reward_snapshot", None)
    if callable(snap):
        try:
            reward_components = snap()
        except Exception:
            reward_components = None

    result = {
        "scope": args.scope,
        "n_train": len(trainset),
        "n_val": len(valset),
        "max_metric_calls": args.max_metric_calls,
        "seed": args.seed,
        "task_model": args.task_model,
        "baseline_val": baseline,
        "optimized_val": opt_eval,
        "seed_prompt": seed_prompt,
        "optimized_prompt": optimized_prompt,
        "reward_components": reward_components,
        "reward_trajectory_file": trajectory_path,
        "wandb": {
            "project": wandb_project,
            "entity": wandb_entity,
            "run_id": wandb_run_id,
            "run_name": f"gepa-{args.scope}",
            "url": wandb_url,
        } if args.wandb else None,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.scope}] wrote {out_dir}/optimized_prompt.md + result.json", flush=True)


if __name__ == "__main__":
    main()
