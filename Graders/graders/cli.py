"""Unified CLI for the graders in this package.

Examples
--------
Grade response relevance (paired: response vs prompt), adding ``relevance_score``::

    python -m graders grade -g relevance_v01 \\
        -f model_output -pf user_input \\
        -i data/in.jsonl -o data/out.jsonl

Grade prompt toxicity (unpaired), adding ``toxicity_label``::

    python -m graders grade -g toxicity_v10 -f user_input \\
        -i data/in.jsonl -o data/out.jsonl

Evaluate a grader against gold labels in a JSONL field::

    python -m graders evaluate -g toxicity_v10 -f user_input \\
        --label-field toxicity -i data/splits/test.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from graders.core import compute_metrics, load_prompts, load_spec, run_grader


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _build_samples(rows: list[dict], spec: dict, field: str | None, prompt_field: str | None):
    if spec.get("paired"):
        if not field or not prompt_field:
            raise SystemExit(
                f"Grader '{spec['name']}' needs paired input: pass both "
                "-f/--field (response) and -pf/--prompt-field (prompt)."
            )
        return [{"prompt": str(r[prompt_field]), "response": str(r[field])} for r in rows]
    if not field:
        raise SystemExit(f"Grader '{spec['name']}' needs -f/--field naming the text to grade.")
    return [str(r.get(field, "")) for r in rows]


async def _grade(args) -> None:
    spec = load_spec(args.grader)
    prompts = load_prompts(spec)
    rows = _read_jsonl(Path(args.input))
    if args.limit:
        rows = rows[: args.limit]
    samples = _build_samples(rows, spec, args.field, args.prompt_field)

    label_name = args.label_name or spec.get("label_name", "label")
    print(
        f"Grading {len(rows)} rows with '{spec['name']}' "
        f"({len(spec['members'])} member(s)) -> column '{label_name}'"
    )
    preds = await run_grader(spec, samples, prompts, workers=args.workers, regrade=args.regrade)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row, pred in zip(rows, preds):
            f.write(json.dumps({**row, label_name: pred}, ensure_ascii=False) + "\n")
    n_unparsed = sum(1 for p in preds if p is None)
    print(f"Wrote {len(rows)} rows to {out_path} (unparsed: {n_unparsed})")


async def _evaluate(args) -> None:
    spec = load_spec(args.grader)
    prompts = load_prompts(spec)
    rows = _read_jsonl(Path(args.input))
    if args.limit:
        rows = rows[: args.limit]
    samples = _build_samples(rows, spec, args.field, args.prompt_field)
    preds = await run_grader(spec, samples, prompts, workers=args.workers, regrade=args.regrade)

    y_true = [int(r[args.label_field]) for r in rows]
    metrics = compute_metrics(y_true, preds)
    print(json.dumps(metrics, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps({"grader": spec["name"], **metrics}, indent=2))
        print(f"Saved metrics to {args.output}")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-g", "--grader", required=True, help="Grader spec name (see specs/).")
    p.add_argument("-i", "--input", required=True, help="Input JSONL file.")
    p.add_argument("-f", "--field", help="Field holding the text to grade (e.g. response).")
    p.add_argument(
        "-pf", "--prompt-field", dest="prompt_field",
        help="Reference field for paired graders (e.g. prompt).",
    )
    p.add_argument("-w", "--workers", type=int, default=16, help="Concurrent workers.")
    p.add_argument("--limit", type=int, default=0, help="Only process the first N rows.")
    p.add_argument("--regrade", action="store_true", help="Ignore cache and regrade.")


def main() -> None:
    ap = argparse.ArgumentParser(prog="graders", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grade", help="Label a JSONL file and add a label column.")
    _add_common(g)
    g.add_argument("-o", "--output", required=True, help="Output JSONL file.")
    g.add_argument("--label-name", help="Output column name (defaults to spec.label_name).")
    g.set_defaults(func=_grade)

    e = sub.add_parser("evaluate", help="Score a grader against gold labels (binary).")
    _add_common(e)
    e.add_argument("--label-field", required=True, help="JSONL field with gold 0/1 labels.")
    e.add_argument("-o", "--output", help="Optional path to save metrics JSON.")
    e.set_defaults(func=_evaluate)

    args = ap.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
