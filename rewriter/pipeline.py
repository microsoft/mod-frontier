#!/usr/bin/env python3
"""Rewrite the T5-flagged responses in a JSONL evaluation file.

The full rewrite stage over one file: find the rows the response filter
flagged, route each with the open probe (REFUSE/REWRITE + domain), select the
GEPA-optimized prompt for the route, rewrite with Qwen3-4B, and write one
record per flagged row.

Usage (GPU; ~2 h for 230 rows on one H100 at the default chunk size):

    python rewriter/pipeline.py \
        -i data/toxicchat_with_GPT5Response.jsonl \
        -o rewrites.jsonl \
        --arm rw_probe_probe

    # Routing can be precomputed/cached (rewriter/routing.py) so the two GPU
    # stages can run as separate jobs; pass --routing-cache to reuse it:
    python rewriter/pipeline.py -i ... -o ... --routing-cache routing.json

Output: JSONL with one record per flagged row:

    {"index": <row index in the input file>, "conv_id": ...,
     "model_output_<arm>": <rewrite or refusal text>,
     "rw_decision": "REFUSE"|"REWRITE", "rw_domain": <probe domain or null>,
     "rw_prompt_id": "gepa960/<scope>", "rw_latency_s": <generate wall-clock>,
     "rw_success": true|false}

Downstream: re-screen the rewrites with the T5 filter and grade them
(``rewriter/eval_e2e.py``), then compute end-to-end metrics with
``metrics/calculate_metrics.py``.
"""

from __future__ import annotations

import argparse
import json
import os

from rewriter import prompts as prompt_packs
from rewriter.rewrite import QwenRewriter, RewriteRequest
from rewriter.routing import route_prompts


def load_rows(path: str) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def flagged_indices(rows: list[dict], flag_field: str) -> list[int]:
    """Indices of rows whose ``flag_field`` equals 1 (string or int)."""
    return [i for i, r in enumerate(rows) if str(r.get(flag_field)) == "1"]


def select_prompt(decision: str, domain: str | None) -> tuple[str, str]:
    """Map a routing decision to ``(prompt_id, prompt_text)``."""
    if decision.strip().upper() == "REFUSE":
        return prompt_packs.refusal_prompt()
    return prompt_packs.select_rewrite_prompt(domain)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite T5-flagged responses")
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file")
    parser.add_argument("-o", "--output", required=True, help="Output rewrites JSONL")
    parser.add_argument("--arm", default="rw_probe_probe",
                        help="Arm suffix used in the output column name "
                             "model_output_<arm> (default: rw_probe_probe)")
    parser.add_argument("--prompt-field", default="user_input")
    parser.add_argument("--response-field", default="model_output_gpt5")
    parser.add_argument("--flag-field", default="T5_model_output_gpt5",
                        help="Rows with this field == 1 are rewritten")
    parser.add_argument("--routing-cache", default=None,
                        help="Optional routing JSON from rewriter/routing.py; "
                             "missing rows are routed live")
    parser.add_argument("--chunk", type=int, default=8,
                        help="Rows per generate call (default 8)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only the first N flagged rows (smoke runs)")
    args = parser.parse_args()

    rows = load_rows(args.input)
    idxs = flagged_indices(rows, args.flag_field)
    if args.limit:
        idxs = idxs[: args.limit]
    print(f"{len(rows)} rows, rewriting {len(idxs)} flagged rows", flush=True)

    # Fail fast if the prompt pack is broken.
    lens = prompt_packs.verify_all()
    print(f"Prompt pack loaded (chars): {json.dumps(lens)}", flush=True)

    # Routing: cache first, live for anything missing.
    routing: dict[str, dict] = {}
    if args.routing_cache and os.path.exists(args.routing_cache):
        with open(args.routing_cache, encoding="utf-8") as f:
            routing = json.load(f)
        print(f"Loaded {len(routing)} cached routing decisions", flush=True)
    missing = [i for i in idxs if str(i) not in routing]
    if missing:
        print(f"Routing {len(missing)} rows live", flush=True)
        decisions = route_prompts([str(rows[i][args.prompt_field]) for i in missing])
        for i, d in zip(missing, decisions):
            routing[str(i)] = {
                "domain": d.domain,
                "decision": d.decision,
                "refuse_probability": d.refuse_probability,
                "domain_confidence": d.domain_confidence,
            }

    # Build requests in flagged order.
    requests, prompt_ids = [], []
    for i in idxs:
        r = routing[str(i)]
        pid, prompt_text = select_prompt(r["decision"], r.get("domain"))
        prompt_ids.append(pid)
        requests.append(RewriteRequest(
            content=str(rows[i][args.response_field]),
            user_input=str(rows[i][args.prompt_field]),
            decision=r["decision"],
            system_prompt=prompt_text,
        ))

    rewriter = QwenRewriter()
    results = rewriter.rewrite_batch(requests, batch_size=args.chunk)

    col = f"model_output_{args.arm}"
    pid_counts: dict[str, int] = {}
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for i, pid, res in zip(idxs, prompt_ids, results):
            pid_counts[pid] = pid_counts.get(pid, 0) + 1
            rec = {
                "index": i,
                "conv_id": rows[i].get("conv_id"),
                col: res.text,
                "rw_decision": routing[str(i)]["decision"],
                "rw_domain": routing[str(i)].get("domain"),
                "rw_prompt_id": pid,
                "rw_latency_s": res.latency_s,
                "rw_success": res.success,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, args.output)
    print(f"Prompt selection split: {json.dumps(pid_counts)}", flush=True)
    print(f"Wrote {len(idxs)} rewrites -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
