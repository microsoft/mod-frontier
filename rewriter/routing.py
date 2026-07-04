#!/usr/bin/env python3
"""Probe-based routing: REFUSE/REWRITE decision + content domain per prompt.

Wraps the vendored routing probe (``rewriter/routing_probe/``, see its
docstring for provenance): a set of small attention-pooling probe heads over
frozen ``Qwen/Qwen3-4B-Instruct-2507`` layer-18 activations. One GPU forward
pass per batch of prompts, no LLM call, deterministic.

The refuse threshold (0.161) is the probe's shipped operating point and the
one every end-to-end number in this directory was measured at. Domain routing
uses the bundled per-domain calibration in ``rewriter/routing_probe/calibration/``.

Usage (standalone; also importable):

    python rewriter/routing.py \
        -i data/toxicchat_with_GPT5Response.jsonl \
        --flag-field T5_model_output_gpt5 --prompt-field user_input \
        -o routing.json

Writes ``{row_index: {domain, decision, refuse_probability,
domain_confidence}}`` for every row whose ``--flag-field`` equals 1. The
pipeline (``rewriter/pipeline.py``) calls :func:`route_prompts` directly; the
cache file exists so routing (GPU) and rewriting (GPU) can run as separate
jobs and so decisions are inspectable.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in (None, ""):  # invoked by path: python rewriter/routing.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Refuse-probability threshold: route REFUSE iff P(refuse) >= this value.
#: The validated operating point of the shipped probe heads.
REFUSE_THRESHOLD = 0.161

#: The eight content domains the probe routes into (canonical order).
DOMAINS = (
    "casual_interaction",
    "creative_writing",
    "information_seeking",
    "other",
    "practical_content",
    "roleplay",
    "task_assistance",
    "translation_transcription",
)


@dataclass
class RouteDecision:
    """One routing decision for one prompt."""

    domain: str
    decision: str  # "REFUSE" | "REWRITE"
    refuse_probability: float
    domain_confidence: float


def route_prompts(
    prompts: list[str],
    device: str | None = None,
    threshold: float = REFUSE_THRESHOLD,
    batch_size: int = 16,
) -> list[RouteDecision]:
    """Route ``prompts`` through the probe; returns one decision per prompt.

    Loads the frozen Qwen3-4B backbone and the bundled probe heads on first
    call (GPU strongly recommended). ``threshold`` is compared against the
    refusal head's probability; domain is the argmax of the calibrated
    one-vs-rest domain heads.
    """
    import torch
    from rewriter.routing_probe import ActivationExtractor, DEFAULT_THRESHOLD, Router

    if abs(DEFAULT_THRESHOLD - REFUSE_THRESHOLD) > 1e-9:
        raise RuntimeError(
            f"routing_probe DEFAULT_THRESHOLD ({DEFAULT_THRESHOLD}) does not "
            f"match the validated operating point ({REFUSE_THRESHOLD}); the vendored "
            "probe in rewriter/routing_probe/ has been modified"
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ActivationExtractor(device=device)
    router = Router(device=device)

    if sorted(DOMAINS) != sorted(router.domains):
        raise RuntimeError(
            f"probe package domains {sorted(router.domains)} do not match the "
            f"expected set {sorted(DOMAINS)}; the vendored probe has been modified"
        )

    decisions = router.route(prompts, extractor, threshold=threshold, batch_size=batch_size)
    return [
        RouteDecision(
            domain=d.domain,
            decision=d.decision,
            refuse_probability=float(d.refuse_probability),
            domain_confidence=float(d.domain_confidence),
        )
        for d in decisions
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Route flagged rows with the open probe")
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file")
    parser.add_argument("-o", "--output", required=True, help="Output routing JSON")
    parser.add_argument("--prompt-field", default="user_input")
    parser.add_argument("--flag-field", default="T5_model_output_gpt5",
                        help="Rows with this field == 1 are routed (default: T5 response flag)")
    parser.add_argument("--threshold", type=float, default=REFUSE_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Route only the first N flagged rows")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
    idxs = [i for i, r in enumerate(rows) if str(r.get(args.flag_field)) == "1"]
    if args.limit:
        idxs = idxs[: args.limit]
    prompts = [str(rows[i][args.prompt_field]) for i in idxs]
    print(f"Routing {len(prompts)} flagged prompts (threshold={args.threshold})", flush=True)

    decisions = route_prompts(prompts, threshold=args.threshold, batch_size=args.batch_size)

    cache = {str(i): asdict(d) for i, d in zip(idxs, decisions)}
    # Provenance: decisions are threshold-dependent, and a consumer mixing a
    # cache made at one threshold with live routing at another would produce
    # inconsistent REFUSE/REWRITE decisions. pipeline.py hard-errors when this
    # record is missing or disagrees with its own threshold.
    cache["_meta"] = {"threshold": args.threshold}
    dec_counts: dict[str, int] = {}
    dom_counts: dict[str, int] = {}
    for d in decisions:
        dec_counts[d.decision] = dec_counts.get(d.decision, 0) + 1
        dom_counts[d.domain] = dom_counts.get(d.domain, 0) + 1

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"Decision split: {dec_counts}", flush=True)
    print(f"Domain split: {json.dumps(dom_counts, indent=2)}", flush=True)
    print(f"Wrote {len(decisions)} routing decisions (threshold={args.threshold}) "
          f"-> {args.output}", flush=True)


if __name__ == "__main__":
    main()
