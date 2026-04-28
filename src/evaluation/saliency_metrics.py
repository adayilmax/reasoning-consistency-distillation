"""Overlap metrics between pairs of GradCAM saliency maps.

Two complementary measures:

  1. Spearman rank correlation  — captures global rank ordering of activations
     across all pixels; robust to monotonic rescaling.

  2. IoU on top-20% pixels — binarises each heatmap at its 80th percentile
     and measures intersection-over-union of the two "hot" regions; captures
     spatial co-localisation of the most salient pixels.

Both metrics are computed on the *flattened* heatmap (H×W → H*W vector).
Input heatmaps must be normalised to [0, 1] (as returned by GradCAM.generate).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Per-pair metrics
# ---------------------------------------------------------------------------

def spearman_correlation(
    cam_a: np.ndarray,
    cam_b: np.ndarray,
) -> float:
    """Spearman rank correlation between two flattened saliency maps.

    Args:
        cam_a: (H, W) heatmap, values in [0, 1].
        cam_b: (H, W) heatmap, values in [0, 1].

    Returns:
        Spearman ρ ∈ [-1, 1].  Returns 0.0 for degenerate (constant) maps.
    """
    a = cam_a.ravel().astype(np.float64)
    b = cam_b.ravel().astype(np.float64)

    # Guard: constant maps produce undefined correlation
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0

    rho, _ = spearmanr(a, b)
    return float(rho) if np.isfinite(rho) else 0.0


def iou_top_k(
    cam_a: np.ndarray,
    cam_b: np.ndarray,
    top_fraction: float = 0.20,
) -> float:
    """IoU between the top-k% most activated pixels in each heatmap.

    Args:
        cam_a: (H, W) heatmap, values in [0, 1].
        cam_b: (H, W) heatmap, values in [0, 1].
        top_fraction: Fraction of pixels to include in each mask (default 0.20).

    Returns:
        IoU ∈ [0, 1].
    """
    a = cam_a.ravel().astype(np.float64)
    b = cam_b.ravel().astype(np.float64)

    k = max(1, int(len(a) * top_fraction))

    # np.partition is O(n) — faster than sorting the full array
    thresh_a = np.partition(a, -k)[-k]
    thresh_b = np.partition(b, -k)[-k]

    mask_a = a >= thresh_a
    mask_b = b >= thresh_b

    intersection = int((mask_a & mask_b).sum())
    union = int((mask_a | mask_b).sum())

    return float(intersection / union) if union > 0 else 0.0


def is_degenerate(cam: np.ndarray, tol: float = 1e-6) -> bool:
    """True if the heatmap is effectively constant (GradCAM failure mode)."""
    return float(cam.max() - cam.min()) < tol


# ---------------------------------------------------------------------------
# Batch aggregation
# ---------------------------------------------------------------------------

def compute_batch_saliency_metrics(
    cams_a: np.ndarray,
    cams_b: np.ndarray,
    top_fraction: float = 0.20,
) -> list[dict]:
    """Compute Spearman + IoU for a batch of heatmap pairs.

    Args:
        cams_a: (N, H, W) array of heatmaps from model A.
        cams_b: (N, H, W) array of heatmaps from model B.
        top_fraction: IoU threshold fraction.

    Returns:
        List of N dicts, each with keys "spearman", "iou", "degenerate_a",
        "degenerate_b".  Degenerate maps get spearman=0.0 and iou=0.0.
    """
    results = []
    for cam_a, cam_b in zip(cams_a, cams_b):
        deg_a = is_degenerate(cam_a)
        deg_b = is_degenerate(cam_b)
        if deg_a or deg_b:
            results.append({
                "spearman": 0.0,
                "iou": 0.0,
                "degenerate_a": deg_a,
                "degenerate_b": deg_b,
            })
        else:
            results.append({
                "spearman": spearman_correlation(cam_a, cam_b),
                "iou": iou_top_k(cam_a, cam_b, top_fraction),
                "degenerate_a": False,
                "degenerate_b": False,
            })
    return results
