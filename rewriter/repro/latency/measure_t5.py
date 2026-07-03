#!/usr/bin/env python3
"""Amortized batched latency of the ToxicChat T5 filter (batch 32).

Times the filter side of the pipeline with the exact inference recipe the
accuracy numbers came from (``moderation/inference_field.py``: "ToxicChat: "
prefix, ``t5-large`` tokenizer, ``max_length=512`` truncation,
``generate(max_new_tokens=5)``), in fp32 -- the recipe's default; fp16 would
understate the latency of the configuration that produced the accuracy cells.

Reports amortized per-sample seconds (batched throughput divided by rows)
for the prompt side (T5 over ``user_input``), the response side (T5 over
``model_output_gpt5``), and their sum -- the three filter scenarios of
``metrics/calculate_metrics.py``. Warm-up batches run first and are excluded;
CUDA is synchronized around every timed batch.

Usage::

    python rewriter/repro/latency/measure_t5.py -o t5_latency.json
    # --limit 64: smoke run over the first 64 rows only
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

MODEL_CHECKPOINT = "lmsys/toxicchat-t5-large-v1.0"
TOKENIZER_NAME = "t5-large"
PREFIX = "ToxicChat: "
BATCH_SIZE = 32


def time_side(texts: list[str], model, tokenizer, device: str,
              warmup_batches: int = 3) -> dict:
    """Batched inference over ``texts``; per-batch wall clock with CUDA sync."""
    import torch

    for start in range(0, min(warmup_batches * BATCH_SIZE, len(texts)), BATCH_SIZE):
        batch = [PREFIX + t for t in texts[start:start + BATCH_SIZE]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=5)
    if device == "cuda":
        torch.cuda.synchronize()

    batch_times, n_flagged = [], 0
    t_total0 = time.perf_counter()
    for start in range(0, len(texts), BATCH_SIZE):
        batch = [PREFIX + t for t in texts[start:start + BATCH_SIZE]]
        t0 = time.perf_counter()
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5)
        if device == "cuda":
            torch.cuda.synchronize()
        batch_times.append(time.perf_counter() - t0)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        n_flagged += sum(1 for d in decoded if d.strip().lower() == "positive")
        done = min(start + BATCH_SIZE, len(texts))
        if done % 640 == 0 or done == len(texts):
            print(f"  {done}/{len(texts)}", flush=True)
    total = time.perf_counter() - t_total0

    return {
        "n_texts": len(texts),
        "batch_size": BATCH_SIZE,
        "total_s": total,
        "amortized_per_sample_s": total / len(texts),
        "per_batch_mean_s": statistics.mean(batch_times),
        "per_batch_p90_s": sorted(batch_times)[int(0.9 * (len(batch_times) - 1))],
        "n_flagged": n_flagged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="T5 filter batched latency")
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "toxicchat_with_GPT5Response.jsonl"))
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Smoke: first N rows only")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    prompts = [str(r["user_input"]) for r in rows]
    responses = [str(r["model_output_gpt5"]) for r in rows]
    print(f"{len(rows)} rows", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT).to(device)
    model.eval()
    if model.dtype != torch.float32:
        raise RuntimeError(f"expected fp32 to match the accuracy recipe, got {model.dtype}")
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"Model loaded on {device} ({gpu})", flush=True)

    res_prompt = time_side(prompts, model, tokenizer, device)
    print(f"prompt side: {res_prompt['amortized_per_sample_s'] * 1000:.1f} ms/sample "
          f"({res_prompt['n_flagged']} flagged)", flush=True)
    res_response = time_side(responses, model, tokenizer, device)
    print(f"response side: {res_response['amortized_per_sample_s'] * 1000:.1f} ms/sample "
          f"({res_response['n_flagged']} flagged)", flush=True)

    out = {
        "model": MODEL_CHECKPOINT,
        "hardware": gpu,
        "batch_size": BATCH_SIZE,
        "prompt": res_prompt,
        "response": res_response,
        "both_amortized_per_sample_s": (
            res_prompt["amortized_per_sample_s"] + res_response["amortized_per_sample_s"]
        ),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
