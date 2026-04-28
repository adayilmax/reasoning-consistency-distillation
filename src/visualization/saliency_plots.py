"""Visualization helpers for GradCAM saliency maps.

Functions:
  overlay_cam_on_image   — blend a heatmap over a CIFAR-10 image
  plot_divergence_grid   — headline figure: both-correct, high-divergence examples
  plot_saliency_examples — random sample of teacher vs. student saliency maps
  plot_cka_heatmap       — teacher × student CKA matrix
  plot_class_breakdown   — per-class Spearman and IoU bar chart
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# CIFAR-10 normalisation constants (must match data/cifar10.py)
_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _unnormalize(img_tensor: np.ndarray) -> np.ndarray:
    """Convert a (C, H, W) normalised float tensor to (H, W, 3) uint8."""
    img = img_tensor.transpose(1, 2, 0)   # (H, W, C)
    img = img * _STD + _MEAN
    img = np.clip(img, 0.0, 1.0)
    return img


def overlay_cam_on_image(
    img_chw: np.ndarray,
    cam_hw: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a GradCAM heatmap over a CIFAR-10 image.

    Args:
        img_chw: (C, H, W) normalised float tensor (as returned by dataloader).
        cam_hw:  (H, W) heatmap in [0, 1].
        alpha:   Heatmap opacity (0 = image only, 1 = heatmap only).
        colormap: Matplotlib colourmap name.

    Returns:
        (H, W, 3) float32 array in [0, 1] — ready for imshow.
    """
    img_hwc = _unnormalize(img_chw)            # (H, W, 3) float32 in [0, 1]
    cmap = plt.get_cmap(colormap)
    heat = cmap(cam_hw)[:, :, :3].astype(np.float32)  # (H, W, 3)
    blended = (1.0 - alpha) * img_hwc + alpha * heat
    return np.clip(blended, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Divergence grid  (headline result figure)
# ---------------------------------------------------------------------------

def plot_divergence_grid(
    examples: list[dict],
    save_path: str,
    class_names: tuple[str, ...] | None = None,
) -> None:
    """Plot a grid showing teacher vs. student saliency for divergent examples.

    Each row: [Original image | Teacher GradCAM | Student GradCAM]
    Rows are sorted by ascending Spearman correlation (most divergent first).

    Args:
        examples: List of dicts, each with keys:
            "image"       : (C, H, W) float tensor (normalised)
            "cam_teacher" : (H, W) heatmap in [0, 1]
            "cam_student" : (H, W) heatmap in [0, 1]
            "true_label"  : int
            "teacher_pred": int
            "student_pred": int
            "spearman"    : float
            "iou"         : float
        save_path: Output PNG path.
        class_names: Optional CIFAR-10 class name tuple.
    """
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.6 * n))
    if n == 1:
        axes = axes[None, :]  # ensure 2-D indexing

    col_titles = ["Original", "Teacher GradCAM", "Student GradCAM"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, fontweight="bold", pad=4)

    for i, ex in enumerate(examples):
        img = ex["image"]
        cam_t = ex["cam_teacher"]
        cam_s = ex["cam_student"]

        label_str = (
            class_names[ex["true_label"]] if class_names else str(ex["true_label"])
        )
        row_label = (
            f"label: {label_str}\n"
            f"ρ={ex['spearman']:.3f}  IoU={ex['iou']:.3f}"
        )

        # Original
        axes[i, 0].imshow(_unnormalize(img))
        axes[i, 0].set_ylabel(row_label, fontsize=7.5, labelpad=4)

        # Teacher overlay
        axes[i, 1].imshow(overlay_cam_on_image(img, cam_t))

        # Student overlay
        axes[i, 2].imshow(overlay_cam_on_image(img, cam_s))

        for j in range(3):
            axes[i, j].axis("off")

    plt.suptitle(
        "High-Divergence Examples: Both Correct, Saliency Maps Disagree",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Random saliency examples
# ---------------------------------------------------------------------------

def plot_saliency_examples(
    examples: list[dict],
    save_path: str,
    class_names: tuple[str, ...] | None = None,
    n_cols: int = 4,
) -> None:
    """Plot a random sample of teacher/student saliency pairs.

    Layout: groups of 3 (original | teacher | student) across n_cols groups.
    """
    n = len(examples)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows * 3, n_cols, figsize=(3 * n_cols, 3 * n_rows * 3 / 3))

    # Flatten to 1-D for easy indexing
    all_ax = axes.ravel()
    used = 0

    for i, ex in enumerate(examples):
        row_base = (i // n_cols) * 3
        col = i % n_cols
        idx = row_base * n_cols + col

        label_str = (
            class_names[ex["true_label"]] if class_names else str(ex["true_label"])
        )

        all_ax[idx].imshow(_unnormalize(ex["image"]))
        all_ax[idx].set_title(f"{label_str}\nρ={ex['spearman']:.2f}", fontsize=7)
        all_ax[idx].axis("off")

        all_ax[idx + n_cols].imshow(overlay_cam_on_image(ex["image"], ex["cam_teacher"]))
        all_ax[idx + n_cols].set_title("Teacher", fontsize=7)
        all_ax[idx + n_cols].axis("off")

        all_ax[idx + 2 * n_cols].imshow(overlay_cam_on_image(ex["image"], ex["cam_student"]))
        all_ax[idx + 2 * n_cols].set_title("Student", fontsize=7)
        all_ax[idx + 2 * n_cols].axis("off")
        used += 1

    # Hide unused axes
    for ax in all_ax[used * 3:]:
        ax.axis("off")

    plt.suptitle("Teacher vs. Student GradCAM (random sample)", fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CKA heatmap
# ---------------------------------------------------------------------------

def plot_cka_heatmap(
    cka_matrix: dict[str, dict[str, float]],
    save_path: str,
) -> None:
    """Plot a teacher × student CKA similarity matrix as a colour heatmap.

    Args:
        cka_matrix: Nested dict [teacher_layer][student_layer] → float.
        save_path: Output PNG path.
    """
    teacher_layers = list(cka_matrix.keys())
    student_layers = list(next(iter(cka_matrix.values())).keys())

    matrix = np.array(
        [[cka_matrix[t][s] for s in student_layers] for t in teacher_layers],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(max(5, len(student_layers) * 1.1),
                                    max(4, len(teacher_layers) * 0.9)))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(student_layers)))
    ax.set_xticklabels(student_layers, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(teacher_layers)))
    ax.set_yticklabels(teacher_layers, fontsize=9)
    ax.set_xlabel("Student layer", fontsize=10)
    ax.set_ylabel("Teacher layer", fontsize=10)
    ax.set_title("CKA Representation Similarity\n(Teacher vs. Student)", fontsize=11)

    # Annotate each cell
    for i in range(len(teacher_layers)):
        for j in range(len(student_layers)):
            val = matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="CKA similarity")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-class breakdown
# ---------------------------------------------------------------------------

def plot_class_breakdown(
    per_class_saliency: list[dict],
    class_names: tuple[str, ...],
    save_path: str,
) -> None:
    """Grouped bar chart: per-class mean Spearman ρ and IoU.

    Args:
        per_class_saliency: List of dicts (one per class) with keys:
            "name", "mean_spearman", "mean_iou", "n_both_correct".
        class_names: Ordered CIFAR-10 class name tuple.
        save_path: Output PNG path.
    """
    names = [c["name"] for c in per_class_saliency]
    spearmans = [c["mean_spearman"] for c in per_class_saliency]
    ious = [c["mean_iou"] for c in per_class_saliency]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars1 = ax.bar(x - width / 2, spearmans, width, label="Mean Spearman ρ",
                   color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, ious, width, label="Mean IoU (top-20%)",
                   color="darkorange", alpha=0.80, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Overlap metric", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=np.mean(spearmans), color="steelblue", linestyle="--",
               linewidth=1.0, alpha=0.6, label=f"Avg ρ={np.mean(spearmans):.3f}")
    ax.axhline(y=np.mean(ious), color="darkorange", linestyle="--",
               linewidth=1.0, alpha=0.6, label=f"Avg IoU={np.mean(ious):.3f}")

    ax.set_title(
        "Per-Class Teacher–Student Saliency Alignment\n(both models correct)",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 4 — Compression curve
# ---------------------------------------------------------------------------

def plot_compression_curve(
    records: list[dict],
    save_path: str,
) -> None:
    """Line plot: saliency alignment and accuracy vs. compression ratio.

    Args:
        records: List of dicts (one per variant), each with keys:
            "label", "n_params", "compression_ratio",
            "accuracy", "ece", "mean_spearman", "mean_iou".
            Must be sorted by compression_ratio ascending (least compressed first).
        save_path: Output PNG path.
    """
    labels      = [r["label"] for r in records]
    n_params    = [r["n_params"] / 1e6 for r in records]  # in millions
    spearmans   = [r["mean_spearman"] for r in records]
    ious        = [r["mean_iou"] for r in records]
    accs        = [r["accuracy"] for r in records]
    eces        = [r["ece"] for r in records]

    x = np.arange(len(records))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ---- Left: Saliency overlap ----
    ax = axes[0]
    ax.plot(x, spearmans, "o-", color="steelblue",  linewidth=2, markersize=7,
            label="Mean Spearman rho")
    ax.plot(x, ious,      "s--", color="darkorange", linewidth=2, markersize=7,
            label="Mean IoU (top-20%)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['label']}\n({r['n_params']/1e3:.0f}k params)" for r in records],
        fontsize=9,
    )
    ax.set_ylabel("Saliency overlap", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title("Reasoning Consistency vs. Compression", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Arrow annotation: more compressed = rightward → degrades
    ax.annotate(
        "More compressed  -->",
        xy=(0.98, 0.03), xycoords="axes fraction",
        ha="right", fontsize=8, color="gray",
        style="italic",
    )

    # ---- Right: Accuracy & ECE ----
    ax2 = axes[1]
    color_acc = "steelblue"
    color_ece = "crimson"
    ax2.plot(x, accs, "o-",  color=color_acc, linewidth=2, markersize=7, label="Accuracy")
    ax2r = ax2.twinx()
    ax2r.plot(x, eces, "s--", color=color_ece, linewidth=2, markersize=7, label="ECE")

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"{r['label']}\n({r['n_params']/1e3:.0f}k params)" for r in records],
        fontsize=9,
    )
    ax2.set_ylabel("Test Accuracy", fontsize=10, color=color_acc)
    ax2r.set_ylabel("ECE", fontsize=10, color=color_ece)
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2r.tick_params(axis="y", labelcolor=color_ece)
    ax2.set_title("Accuracy & Calibration vs. Compression", fontsize=11)

    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="lower left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 4 — Temperature effect
# ---------------------------------------------------------------------------

def plot_temperature_effect(
    records: list[dict],
    save_path: str,
) -> None:
    """Multi-panel line plot: effect of distillation temperature.

    Args:
        records: List of dicts sorted by temperature ascending, each with:
            "temperature", "accuracy", "ece", "mean_spearman", "mean_iou".
        save_path: Output PNG path.
    """
    temps     = [r["temperature"] for r in records]
    spearmans = [r["mean_spearman"] for r in records]
    ious      = [r["mean_iou"] for r in records]
    accs      = [r["accuracy"] for r in records]
    eces      = [r["ece"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- Left: Saliency overlap vs T ----
    ax = axes[0]
    ax.plot(temps, spearmans, "o-",  color="steelblue",  lw=2, ms=8,
            label="Mean Spearman rho")
    ax.plot(temps, ious,      "s--", color="darkorange", lw=2, ms=8,
            label="Mean IoU (top-20%)")
    for t, s, u in zip(temps, spearmans, ious):
        ax.annotate(f"T={t}", (t, s), textcoords="offset points",
                    xytext=(4, 4), fontsize=8, color="steelblue")
    ax.set_xlabel("Distillation Temperature T", fontsize=10)
    ax.set_ylabel("Saliency overlap", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title("Effect of Temperature on Reasoning Alignment", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ---- Right: Accuracy & ECE vs T ----
    ax2  = axes[1]
    ax2r = ax2.twinx()
    color_acc = "steelblue"
    color_ece = "crimson"
    ax2.plot(temps,  accs, "o-",  color=color_acc, lw=2, ms=8, label="Accuracy")
    ax2r.plot(temps, eces, "s--", color=color_ece, lw=2, ms=8, label="ECE")
    ax2.set_xlabel("Distillation Temperature T", fontsize=10)
    ax2.set_ylabel("Test Accuracy", fontsize=10, color=color_acc)
    ax2r.set_ylabel("ECE", fontsize=10, color=color_ece)
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2r.tick_params(axis="y", labelcolor=color_ece)
    ax2.set_title("Effect of Temperature on Accuracy & Calibration", fontsize=11)
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="best")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 4 — Per-class reasoning gap heatmap
# ---------------------------------------------------------------------------

def plot_per_class_gap_heatmap(
    class_names: tuple[str, ...],
    variant_labels: list[str],
    spearman_matrix: np.ndarray,
    save_path: str,
) -> None:
    """Heatmap of per-class mean Spearman rho for each model variant.

    Args:
        class_names: CIFAR-10 class names (10 entries).
        variant_labels: One label per model variant (columns).
        spearman_matrix: (10, n_variants) array of mean Spearman rho values.
        save_path: Output PNG path.
    """
    n_classes, n_variants = spearman_matrix.shape
    fig, ax = plt.subplots(figsize=(max(7, n_variants * 1.5), 5))

    im = ax.imshow(
        spearman_matrix, vmin=0.0, vmax=1.0,
        cmap="RdYlGn", aspect="auto",
    )

    ax.set_xticks(range(n_variants))
    ax.set_xticklabels(variant_labels, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Model variant", fontsize=10)
    ax.set_title(
        "Per-Class Teacher-Student Saliency Alignment (Mean Spearman rho)\n"
        "Green = well-aligned, Red = divergent",
        fontsize=11,
    )

    for i in range(n_classes):
        for j in range(n_variants):
            val = spearman_matrix[i, j]
            color = "black" if 0.3 < val < 0.8 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Mean Spearman rho", shrink=0.8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
