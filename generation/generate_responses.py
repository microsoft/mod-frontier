#!/usr/bin/env python3
"""Generate model responses for user_input prompts in a JSONL dataset.

Uses Azure OpenAI with identity-based auth (DefaultAzureCredential).
Supports resuming: skips rows that already have the output field populated.

Usage:
    python generate_responses.py \
        --input ../data/toxicchat_with_relevance.jsonl \
        --output ../data/toxicchat_with_relevance.jsonl \
        --model gpt-5 \
        --workers 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path

from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI, RateLimitError, APIError, APITimeoutError, APIConnectionError

# ── Model configs ──────────────────────────────────────────────────────────────
# Azure OpenAI endpoint. Set AZURE_OPENAI_ENDPOINT, or edit the placeholder.
AZURE_OPENAI_ENDPOINT = os.environ.get(
    "AZURE_OPENAI_ENDPOINT", "https://<your-resource>.openai.azure.com/"
)

MODEL_CONFIGS: dict[str, dict] = {
    "gpt-5": {
        "endpoint": AZURE_OPENAI_ENDPOINT,
        "deployment": "gpt-5",
        "api_version": "2024-12-01-preview",
        "output_field": "model_output_gpt5",
    },
    # Add more models here, e.g.:
    # "gpt-4o": {
    #     "endpoint": AZURE_OPENAI_ENDPOINT,
    #     "deployment": "gpt-4o",
    #     "api_version": "2024-12-01-preview",
    #     "output_field": "model_output_gpt4o",
    # },
}

SCOPE = "https://cognitiveservices.azure.com/.default"


def is_reasoning_model(model: str) -> bool:
    """Reasoning models (o*/gpt-5) reject temperature and use max_completion_tokens."""
    return model.startswith("o") or model.startswith("gpt-5")


async def generate_one(
    client: AsyncAzureOpenAI,
    deployment: str,
    model: str,
    prompt: str,
    sem: asyncio.Semaphore,
    max_retries: int = 6,
) -> str:
    """Call the model with retry logic."""
    messages = [{"role": "user", "content": prompt}]

    call_kwargs: dict = {"model": deployment, "messages": messages}
    if not is_reasoning_model(model):
        call_kwargs["temperature"] = 0.7

    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            async with sem:
                resp = await client.chat.completions.create(**call_kwargs)
            return resp.choices[0].message.content or ""
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            last_exc = exc
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
        except APIError as exc:
            if exc.status_code and exc.status_code >= 500:
                last_exc = exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries: {last_exc}")


async def run(
    input_path: Path,
    output_path: Path,
    model: str,
    workers: int,
) -> None:
    cfg = MODEL_CONFIGS[model]
    output_field = cfg["output_field"]

    # Load all rows
    rows = [json.loads(line) for line in open(input_path)]
    total = len(rows)

    # Find rows that need generation
    todo_indices = [i for i, r in enumerate(rows) if not r.get(output_field)]
    print(f"Total rows: {total}, already done: {total - len(todo_indices)}, to generate: {len(todo_indices)}")

    if not todo_indices:
        print("Nothing to do.")
        return

    # Set up client — prefer API key from env, fall back to DefaultAzureCredential
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    credential = None
    if api_key:
        print("Using API key authentication")
        client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=cfg["api_version"],
            azure_endpoint=cfg["endpoint"],
            max_retries=4,
        )
    else:
        print("Using DefaultAzureCredential authentication")
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, SCOPE)
        client = AsyncAzureOpenAI(
            azure_ad_token_provider=token_provider,
            api_version=cfg["api_version"],
            azure_endpoint=cfg["endpoint"],
            max_retries=4,
        )
    sem = asyncio.Semaphore(workers)

    done = 0
    failed = 0

    async def process(idx: int) -> None:
        nonlocal done, failed
        prompt = rows[idx].get("user_input", "")
        try:
            result = await generate_one(client, cfg["deployment"], model, prompt, sem)
            rows[idx][output_field] = result
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] row {idx}: {exc}", file=sys.stderr)
            rows[idx][output_field] = f"ERROR: {exc}"
        done += 1
        if done % 100 == 0 or done == len(todo_indices):
            print(f"  progress: {done}/{len(todo_indices)} (failed: {failed})")

    # Run with concurrency
    tasks = [asyncio.create_task(process(i)) for i in todo_indices]
    await asyncio.gather(*tasks)

    # Atomic write: write to temp file then move
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent, suffix=".jsonl.tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        shutil.move(tmp_path, output_path)
    except BaseException:
        os.unlink(tmp_path)
        raise

    if credential:
        await credential.close()
    await client.close()

    print(f"Done. Wrote {total} rows to {output_path} ({failed} failures)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model responses for JSONL dataset")
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file")
    parser.add_argument("-o", "--output", help="Output JSONL file (default: same as input)")
    parser.add_argument("-m", "--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use")
    parser.add_argument("-w", "--workers", type=int, default=16, help="Concurrent requests")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    asyncio.run(run(input_path, output_path, args.model, args.workers))


if __name__ == "__main__":
    main()
