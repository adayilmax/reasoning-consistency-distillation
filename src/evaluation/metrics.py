"""Evaluation metrics: Expected Calibration Error (ECE).

ECE measures how well a model's predicted confidence aligns with its
actual accuracy.  We partition predictions into *n_bins* equal-width
confidence bins and compute:

    ECE = sum_{m=1}^{M} (|B_m| / N) * |acc(B_m) - conf(B_m)|

The per-bin statistics are returned alongside the scalar ECE so that
reliability diagrams can be plotted directly from the output.
"""

import numpy as np
import torch


def compute_ece(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> dict:
    """Compute Expected Calibration Error.

    Args:
        probs: (N, C) predicted class probabilities (softmax outputs).
        targets: (N,) ground-truth labels.
        n_bins: Number of equal-width confidence bins.

    Returns:
        Dictionary with scalar ``ece`` and a ``bin_stats`` list (one entry
        per bin) containing average confidence, accuracy, and sample count.
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(targets)

    confidences_np = confidences.numpy()
    accuracies_np = accuracies.numpy().astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_stats: list[dict] = []

    for lower, upper in zip(bin_lowers, bin_uppers):
        # First bin includes the lower boundary
        if lower == 0.0:
            in_bin = (confidences_np >= lower) & (confidences_np <= upper)
        else:
            in_bin = (confidences_np > lower) & (confidences_np <= upper)

        count = int(in_bin.sum())
        proportion = count / len(confidences_np)

        if count > 0:
            avg_conf = float(confidences_np[in_bin].mean())
            avg_acc = float(accuracies_np[in_bin].mean())
            ece += abs(avg_acc - avg_conf) * proportion
        else:
            avg_conf = float((lower + upper) / 2)
            avg_acc = 0.0

        bin_stats.append({
            "lower": float(lower),
            "upper": float(upper),
            "avg_confidence": avg_conf,
            "avg_accuracy": avg_acc,
            "count": count,
            "proportion": proportion,
        })

    return {"ece": float(ece), "n_bins": n_bins, "bin_stats": bin_stats}


# ---------------------------------------------------------------------------
# Per-class classification metrics
# ---------------------------------------------------------------------------

def compute_per_class_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    class_names: tuple[str, ...] | None = None,
) -> dict:
    """Compute per-class accuracy, precision, recall, and F1.

    Uses the one-vs-rest (macro) formulation standard in multi-class problems.

    Args:
        probs: (N, C) softmax probabilities.
        targets: (N,) ground-truth class indices.
        class_names: Optional tuple of C class name strings.

    Returns:
        Dict with:
            "per_class": list of per-class dicts (name, accuracy, precision,
                         recall, f1, support)
            "macro_f1": unweighted mean F1 across classes
            "weighted_f1": support-weighted mean F1
    """
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    preds = probs.argmax(dim=1).numpy()
    targets_np = targets.numpy()
    n_classes = probs.shape[1]

    if class_names is None:
        class_names = tuple(str(i) for i in range(n_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        targets_np, preds, labels=list(range(n_classes)), zero_division=0,
    )

    per_class = []
    for i in range(n_classes):
        mask = targets_np == i
        cls_acc = float(accuracy_score(targets_np[mask], preds[mask])) if mask.sum() > 0 else 0.0
        per_class.append({
            "class_idx": i,
            "name": class_names[i],
            "accuracy": cls_acc,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })

    macro_f1 = float(f1.mean())
    weighted_f1 = float(
        (f1 * support).sum() / support.sum() if support.sum() > 0 else 0.0
    )

    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
