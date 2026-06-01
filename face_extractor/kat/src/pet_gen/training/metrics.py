"""Evaluation metrics for multi-label attribute prediction."""

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score


def compute_metrics(
    logits: np.ndarray, targets: np.ndarray, attr_names: list[str] | None = None
) -> dict:
    """Compute per-attribute and mean metrics.

    Args:
        logits: Raw logits array of shape (N, num_attrs)
        targets: Binary targets array of shape (N, num_attrs)
        attr_names: Optional attribute names for per-attribute breakdown

    Returns:
        Dictionary with mean and per-attribute metrics
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.float32)
    num_attrs = targets.shape[1]

    per_attr_acc = []
    per_attr_f1 = []
    per_attr_ap = []

    for i in range(num_attrs):
        per_attr_acc.append(accuracy_score(targets[:, i], preds[:, i]))
        per_attr_f1.append(f1_score(targets[:, i], preds[:, i], zero_division=0))
        try:
            per_attr_ap.append(average_precision_score(targets[:, i], probs[:, i]))
        except ValueError:
            per_attr_ap.append(0.0)

    results = {
        "mean_accuracy": float(np.mean(per_attr_acc)),
        "mean_f1": float(np.mean(per_attr_f1)),
        "mAP": float(np.mean(per_attr_ap)),
    }

    if attr_names is not None:
        for i, name in enumerate(attr_names):
            results[f"{name}/accuracy"] = per_attr_acc[i]
            results[f"{name}/f1"] = per_attr_f1[i]
            results[f"{name}/ap"] = per_attr_ap[i]

    return results
