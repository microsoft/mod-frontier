"""Run ToxicChat T5-Large on a custom field in a JSONL file.

Usage:
    python inference_field.py --input data.jsonl --output data.jsonl \
        --field model_output_gpt5 --label T5_model_output_gpt5
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_CHECKPOINT = "lmsys/toxicchat-t5-large-v1.0"
TOKENIZER_NAME = "t5-large"
PREFIX = "ToxicChat: "


def run_t5_on_texts(texts, model, tokenizer, device, batch_size=32):
    """Run T5 inference, return list of 0/1 predictions."""
    preds = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            [PREFIX + t for t in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for d in decoded:
            preds.append(1 if d.strip().lower() == "positive" else 0)
        done = min(start + batch_size, len(texts))
        if done % 320 == 0 or done == len(texts):
            print(f"  {done}/{len(texts)} ({100 * done / len(texts):.1f}%)")
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--field", required=True, help="JSONL field to classify")
    parser.add_argument("--label", required=True, help="Output column name for prediction")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT).to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    print(f"Loaded {len(rows)} rows")

    texts = [str(r.get(args.field, "")) for r in rows]

    print(f"\n=== Running T5 on '{args.field}' ===")
    t0 = time.time()
    preds = run_t5_on_texts(texts, model, tokenizer, device, args.batch_size)
    print(f"Done in {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row, p in zip(rows, preds):
            row[args.label] = p
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} rows to {args.output}")
    print(f"{args.label} toxic: {sum(preds)}/{len(preds)}")


if __name__ == "__main__":
    main()
