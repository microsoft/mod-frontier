"""Evaluate T5 predictions against ToxicChat paper-reported metrics.

Paper-reported results on toxicchat0124 test set:
  ToxicChat-T5-large: Precision=0.7983, Recall=0.8475, F1=0.8221, AUPRC=0.8850

This script:
1. Compares reproduced metrics against paper claims
2. Extracts performance on human-labelled subset
"""
import argparse
import json
import os

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)


PAPER_METRICS = {
    "precision": 0.7983,
    "recall": 0.8475,
    "f1": 0.8221,
    "auprc": 0.8850,
}


def load_predictions(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def compute_metrics(rows: list[dict]) -> dict:
    y_true = [r["toxicity_label"] for r in rows]
    y_pred = [r["t5_prediction"] for r in rows]

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # AUPRC requires probability scores; since we only have binary predictions,
    # we use the binary prediction as the score (this gives a lower-bound AUPRC).
    # For exact AUPRC comparison, model logits would be needed.
    metrics["auprc_binary"] = average_precision_score(y_true, y_pred)

    return metrics


def print_comparison(label: str, metrics: dict, reference: dict | None = None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'Metric':<12} {'Reproduced':>12}", end="")
    if reference:
        print(f" {'Paper':>10} {'Delta':>10}")
    else:
        print()

    for key in ["precision", "recall", "f1"]:
        val = metrics[key]
        line = f"  {key:<12} {val:>12.4f}"
        if reference and key in reference:
            ref = reference[key]
            delta = val - ref
            line += f" {ref:>10.4f} {delta:>+10.4f}"
        print(line)

    if "auprc_binary" in metrics:
        val = metrics["auprc_binary"]
        line = f"  {'auprc*':<12} {val:>12.4f}"
        if reference and "auprc" in reference:
            ref = reference["auprc"]
            line += f" {ref:>10.4f} {val - ref:>+10.4f}"
        print(line)
        print("  * AUPRC from binary predictions (lower bound; exact needs logits)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=os.path.join(os.path.dirname(__file__), "outputs", "predictions_test.jsonl"),
        help="Path to predictions JSONL file",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "outputs", "evaluation_report.json"),
        help="Path to save evaluation report",
    )
    args = parser.parse_args()

    print("Loading predictions...")
    rows = load_predictions(args.predictions)
    print(f"Loaded {len(rows)} predictions")

    # --- Full test set ---
    full_metrics = compute_metrics(rows)
    print_comparison("Full Test Set (vs Paper)", full_metrics, PAPER_METRICS)

    y_true = [r["toxicity_label"] for r in rows]
    y_pred = [r["t5_prediction"] for r in rows]
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    # --- Human-annotated subset ---
    human_rows = [r for r in rows if r.get("human_annotation")]
    report = {"full_test_set": {"n": len(rows), **full_metrics}}

    if human_rows:
        human_metrics = compute_metrics(human_rows)
        print_comparison("Human-Annotated Subset", human_metrics)

        y_true_h = [r["toxicity_label"] for r in human_rows]
        y_pred_h = [r["t5_prediction"] for r in human_rows]
        cm_h = confusion_matrix(y_true_h, y_pred_h)
        toxic_h = sum(y_true_h)
        print(f"\n  Total: {len(human_rows)}, Toxic: {toxic_h} ({100*toxic_h/len(human_rows):.1f}%)")
        print(f"  Confusion Matrix:")
        print(f"    TN={cm_h[0][0]}  FP={cm_h[0][1]}")
        print(f"    FN={cm_h[1][0]}  TP={cm_h[1][1]}")

        report["human_annotated"] = {"n": len(human_rows), **human_metrics}
    else:
        print("\n  No human-annotated rows found in predictions.")

    # --- Non-human-annotated (model-annotated) subset ---
    model_rows = [r for r in rows if not r.get("human_annotation")]
    if model_rows:
        model_metrics = compute_metrics(model_rows)
        print_comparison("Model-Annotated Subset (non-human)", model_metrics)
        report["model_annotated"] = {"n": len(model_rows), **model_metrics}

    # Match check
    print(f"\n{'='*60}")
    print("  Paper Reproduction Check")
    print(f"{'='*60}")
    for key in ["precision", "recall", "f1"]:
        repro = full_metrics[key]
        paper = PAPER_METRICS[key]
        close = abs(repro - paper) < 0.02  # within 2% tolerance
        status = "MATCH" if close else "MISMATCH"
        print(f"  {key:<12} {status} (reproduced={repro:.4f}, paper={paper:.4f})")

    # Save report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
