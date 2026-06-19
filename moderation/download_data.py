"""Download ToxicChat dataset (version 0124) from HuggingFace."""
import json
import os
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading ToxicChat (toxicchat0124) from HuggingFace...")
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")

    for split_name, split_data in dataset.items():
        out_path = os.path.join(DATA_DIR, f"toxicchat0124_{split_name}.jsonl")
        with open(out_path, "w") as f:
            for row in split_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(split_data)} rows -> {out_path}")

    # Print basic stats
    for split_name, split_data in dataset.items():
        total = len(split_data)
        toxic = sum(1 for r in split_data if r["toxicity"] == 1)
        human = sum(1 for r in split_data if r["human_annotation"])
        print(f"\n{split_name} stats:")
        print(f"  Total: {total}")
        print(f"  Toxic: {toxic} ({100*toxic/total:.1f}%)")
        print(f"  Human annotated: {human} ({100*human/total:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
