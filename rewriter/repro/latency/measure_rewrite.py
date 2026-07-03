#!/usr/bin/env python3
"""Serving-latency measurement of the rewrite pipeline (route -> select -> rewrite).

Measures what a user would wait for a single flagged response, under a
serving-style deployment rather than the batched evaluator:

* a **dedicated vLLM server** for ``Qwen/Qwen3-4B-Instruct-2507`` (any
  OpenAI-compatible server works), so generation latency is not confounded
  with model-load or co-tenant traffic;
* **serial dispatch** -- one request at a time, so each sample sees an idle
  server (latency, not throughput);
* **streaming**, so time-to-first-token (TTFT) is observable;
* **warm-up samples** (excluded from the stats) before measurement, so CUDA
  graph/cache warm-up doesn't inflate the first rows;
* a **fixed seed** (42) drawing warm-up + measurement rows from the T5-flagged
  set.

Per sample it times: ``route_s`` (probe forward pass, batch of 1),
``select_s`` (prompt-pack lookup), ``ttft_s`` (request send -> first streamed
content token), ``gen_s`` (request send -> final token, including the
bare-refusal retry when it fires -- that retry is real user-visible latency),
and ``e2e_s`` (route + select + gen).

Routing, prompt selection, and message construction are imported from the
package itself (``rewriter.routing``, ``rewriter.pipeline.select_prompt``,
``rewriter.rewrite.build_messages``), so the measured path is the shipped
path; only the generation transport differs from the batched evaluator
(vLLM streaming vs batched HF ``generate``, same greedy decoding and token
budget).

**Report median and P90 beside the mean.** The latency distribution is
bimodal -- refusals and short rewrites finish in well under a second while
long rewrites run to thousands of tokens -- so the mean tracks the long tail,
not the typical request (reference run: e2e mean 1.60 s vs median 0.47 s).

Usage::

    # 1. start a dedicated server (one GPU):
    vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8100 \
        --gpu-memory-utilization 0.55 --max-model-len 16384

    # 2. measure (the probe shares the same GPU):
    python rewriter/repro/latency/measure_rewrite.py \
        --base-url http://localhost:8100/v1 -o rewrite_latency.json

    # --smoke: 1 warm-up + 3 samples at max_tokens=64, pipeline validation only
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

if __package__ in (None, ""):  # invoked by path
    sys.path.insert(0, str(REPO_ROOT))

SEED = 42
N_WARMUP = 3
N_MEASURE = 30
MAX_TOKENS = 2048  # matches rewriter.rewrite.MAX_NEW_TOKENS


def pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile on the sorted values."""
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def stats(values: list[float]) -> dict:
    """Summary stats; median/P90 matter here (bimodal distribution)."""
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": pct(values, 0.90),
        "p95": pct(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def draw_sample(idxs: list[int], n_warmup: int, n_measure: int,
                seed: int = SEED) -> tuple[list[int], list[int]]:
    """Deterministic warm-up + measurement rows from the flagged indices."""
    sample = random.Random(seed).sample(idxs, n_warmup + n_measure)
    return sample[:n_warmup], sample[n_warmup:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite-pipeline serving latency")
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "toxicchat_with_GPT5Response.jsonl"))
    parser.add_argument("--base-url", default="http://localhost:8100/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--flag-field", default="T5_model_output_gpt5")
    parser.add_argument("--smoke", action="store_true",
                        help="1 warm-up + 3 samples at max_tokens=64 (validation only)")
    args = parser.parse_args()

    from rewriter import prompts as prompt_packs
    from rewriter.pipeline import flagged_indices, load_rows, select_prompt
    from rewriter.rewrite import (
        build_messages,
        build_refusal_retry_messages,
        clean_completion,
        is_bare_refusal,
    )
    from rewriter.routing import REFUSE_THRESHOLD

    print(f"Prompt pack loaded (chars): {json.dumps(prompt_packs.verify_all())}", flush=True)

    rows = load_rows(args.data)
    idxs = flagged_indices(rows, args.flag_field)
    print(f"{len(rows)} rows, {len(idxs)} T5-flagged", flush=True)

    n_measure = 3 if args.smoke else N_MEASURE
    n_warmup = 1 if args.smoke else N_WARMUP
    max_tokens = 64 if args.smoke else MAX_TOKENS
    warmup_idx, measure_idx = draw_sample(idxs, n_warmup, n_measure)
    print(f"warm-up rows: {warmup_idx}; measurement rows: {measure_idx}", flush=True)

    # Probe routing stack, loaded once (as in production serving).
    import torch
    from safeflow_routing_probe import ActivationExtractor, Router

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    extractor = ActivationExtractor(device=device)
    router = Router(device=device)
    print(f"Probe stack loaded in {time.perf_counter() - t0:.1f}s on {device}", flush=True)

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)

    def run_one(i: int) -> dict:
        row = rows[i]
        prompt_text = str(row["user_input"])
        response_text = str(row["model_output_gpt5"])

        t0 = time.perf_counter()
        decision = router.route([prompt_text], extractor, threshold=REFUSE_THRESHOLD,
                                batch_size=1)[0]
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        pid, system_prompt = select_prompt(decision.decision, decision.domain)
        t2 = time.perf_counter()

        def stream_once(messages) -> tuple[float | None, float, str]:
            """One streamed generation; returns (ttft_s, wall_s, cleaned_text)."""
            t_send = time.perf_counter()
            stream = client.chat.completions.create(
                model=args.model, messages=messages,
                temperature=0.0, max_tokens=max_tokens, stream=True,
            )
            ttft, text_parts = None, []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    if ttft is None:
                        ttft = time.perf_counter() - t_send
                    text_parts.append(chunk.choices[0].delta.content)
            wall = time.perf_counter() - t_send
            return ttft, wall, clean_completion("".join(text_parts))

        messages = build_messages(response_text, prompt_text, decision.decision, system_prompt)
        ttft, gen_wall, text = stream_once(messages)

        # The serving path retries a bare one-liner refusal once with a
        # contextual-refusal prompt; that retry is user-visible latency, so it
        # counts when it fires (rare by design).
        retried = False
        if is_bare_refusal(text):
            retry_msgs = build_refusal_retry_messages(response_text, prompt_text,
                                                      decision.decision)
            _, retry_wall, retry_text = stream_once(retry_msgs)
            if retry_text and not is_bare_refusal(retry_text):
                text = retry_text
            gen_wall += retry_wall
            retried = True

        return {
            "index": i,
            "decision": decision.decision,
            "domain": decision.domain,
            "prompt_id": pid,
            "route_s": t1 - t0,
            "select_s": t2 - t1,
            "ttft_s": ttft,
            "gen_s": gen_wall,
            "e2e_s": (t1 - t0) + (t2 - t1) + gen_wall,
            "bare_refusal_retried": retried,
            "out_chars": len(text),
        }

    for k, i in enumerate(warmup_idx):
        r = run_one(i)
        print(f"warm-up {k + 1}/{len(warmup_idx)}: e2e {r['e2e_s']:.2f}s "
              f"(ttft {r['ttft_s']:.3f}s)", flush=True)

    records = []
    for k, i in enumerate(measure_idx):
        r = run_one(i)
        records.append(r)
        print(f"measure {k + 1}/{n_measure}: idx={i} {r['decision']}/{r['domain']} "
              f"route {r['route_s']:.3f}s ttft {r['ttft_s']:.3f}s gen {r['gen_s']:.2f}s "
              f"e2e {r['e2e_s']:.2f}s ({r['out_chars']} chars)", flush=True)

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    out = {
        "methodology": {
            "server": "dedicated OpenAI-compatible server (vLLM), serial dispatch, streaming",
            "model": args.model,
            "n_warmup": len(warmup_idx),
            "n_measure": n_measure,
            "max_tokens": max_tokens,
            "seed": SEED,
            "hardware": gpu,
            "smoke": args.smoke,
            "n_bare_refusal_retries": sum(1 for r in records if r["bare_refusal_retried"]),
        },
        "stage_stats": {
            "route_s": stats([r["route_s"] for r in records]),
            "select_s": stats([r["select_s"] for r in records]),
            "ttft_s": stats([r["ttft_s"] for r in records if r["ttft_s"] is not None]),
            "gen_s": stats([r["gen_s"] for r in records]),
            "e2e_s": stats([r["e2e_s"] for r in records]),
        },
        "decision_split": {
            d: sum(1 for r in records if r["decision"] == d)
            for d in {r["decision"] for r in records}
        },
        "records": records,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    e = out["stage_stats"]["e2e_s"]
    print(f"E2E mean {e['mean']:.3f}s median {e['median']:.3f}s "
          f"P90 {e['p90']:.3f}s P95 {e['p95']:.3f}s", flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
