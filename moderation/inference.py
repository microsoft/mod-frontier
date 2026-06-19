"""Run ToxicChat T5-Large inference on the full dataset.

This script is designed to run on an AML compute node with GPU.
It downloads the model from HuggingFace, loads local data, runs
inference, and saves predictions.
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


def load_data(data_path: str) -> list[dict]:
    rows = []
    with open(data_path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def run_inference(
    rows: list[dict],
    model,
    tokenizer,
    device: str,
    batch_size: int = 32,
) -> list[dict]:
    """Run T5 inference on all rows and return rows with predictions."""
    results = []
    total = len(rows)

    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        texts = [PREFIX + r["user_input"] for r in batch]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for row, pred_text in zip(batch, decoded):
            pred_label = 1 if pred_text.strip().lower() == "positive" else 0
            result = {
                "conv_id": row.get("conv_id", ""),
                "user_input": row["user_input"],
                "toxicity_label": row["toxicity"],
                "human_annotation": row.get("human_annotation", False),
                "jailbreaking": row.get("jailbreaking", 0),
                "t5_prediction": pred_label,
                "t5_raw_output": pred_text.strip(),
            }
            results.append(result)

        done = min(start + batch_size, total)
        print(f"  Processed {done}/{total} ({100*done/total:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory containing toxicchat0124_*.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "outputs"),
        help="Directory to write prediction results",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Which split to run inference on (default: test)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"Loading model: {MODEL_CHECKPOINT}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT).to(device)
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    data_path = os.path.join(args.data_dir, f"toxicchat0124_{args.split}.jsonl")
    print(f"\nLoading data from {data_path}")
    rows = load_data(data_path)
    print(f"Loaded {len(rows)} rows")

    print("\nRunning inference...")
    t0 = time.time()
    results = run_inference(rows, model, tokenizer, device, args.batch_size)
    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.1f}s ({len(results)/elapsed:.1f} rows/s)")

    out_path = os.path.join(args.output_dir, f"predictions_{args.split}.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nPredictions saved to {out_path}")

    # Quick summary
    correct = sum(1 for r in results if r["t5_prediction"] == r["toxicity_label"])
    print(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.2f}%)")


if __name__ == "__main__":
    main()
