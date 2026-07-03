"""Classification metrics for routing heads.

``compute_metrics`` is the per-head binary metric (matches experiment #4). The
headline domain metric is the macro-average of the per-OvR-head ``macro_f1`` /
``balanced_acc`` across the eight domains (see ``aggregate_domain_metrics``).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Binary classification metrics at a decision ``threshold`` on prob >= thr.

    Returns dict with ``accuracy``, ``balanced_acc``, ``macro_f1``, ``roc_auc``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def aggregate_domain_metrics(per_head: dict[str, dict]) -> dict:
    """Macro-average per-head metrics across domains (the headline domain metric).

    Args:
        per_head: ``{domain_name: {"macro_f1": ..., "balanced_acc": ..., ...}}``.

    Returns:
        ``{"macro_f1": mean_over_heads, "balanced_acc": mean_over_heads, ...}``.
    """
    keys = ["accuracy", "balanced_acc", "macro_f1", "roc_auc"]
    out = {}
    for k in keys:
        vals = [m[k] for m in per_head.values() if k in m and not np.isnan(m[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out
