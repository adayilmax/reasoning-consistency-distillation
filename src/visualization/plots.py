"""Visualization helpers: reliability diagrams and training curves.

All plot functions save to disk (PNG) and close the figure to avoid
memory leaks during batch evaluation.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless / CI use
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    bin_stats: list[dict],
    model_name: str,
    ece: float,
    accuracy: float,
    save_path: str,
) -> None:
    """Plot a calibration reliability diagram.

    Args:
        bin_stats: Per-bin statistics from ``compute_ece``.
        model_name: Label for the plot title.
        ece: Scalar ECE value.
        accuracy: Overall test accuracy.
        save_path: Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    midpoints = [(b["lower"] + b["upper"]) / 2 for b in bin_stats]
    avg_accs = [b["avg_accuracy"] for b in bin_stats]
    bin_width = bin_stats[0]["upper"] - bin_stats[0]["lower"]

    # Gap bars (shows miscalibration per bin)
    avg_confs = [b["avg_confidence"] for b in bin_stats]
    gaps = [a - c for a, c in zip(avg_accs, avg_confs)]

    ax.bar(
        midpoints, avg_accs, width=bin_width * 0.85,
        alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.5,
        label="Accuracy",
    )
    ax.bar(
        midpoints, gaps, bottom=avg_confs, width=bin_width * 0.85,
        alpha=0.25, color="crimson", edgecolor="crimson", linewidth=0.5,
        label="Gap",
    )

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5,
            label="Perfect calibration")

    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(
        f"{model_name}\nAcc = {accuracy:.3f}   ECE = {ece:.4f}",
        fontsize=12,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: dict,
    model_name: str,
    save_path: str,
) -> None:
    """Plot loss and accuracy curves (train vs. test) over epochs.

    Args:
        history: Dict with keys train_loss, test_loss, train_acc, test_acc.
        model_name: Label for the plot title.
        save_path: Output PNG path.
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Loss ---
    ax1.plot(epochs, history["train_loss"], label="Train", linewidth=1.4)
    ax1.plot(epochs, history["test_loss"], label="Test", linewidth=1.4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"{model_name} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Accuracy ---
    ax2.plot(epochs, history["train_acc"], label="Train", linewidth=1.4)
    ax2.plot(epochs, history["test_acc"], label="Test", linewidth=1.4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Side-by-side reliability comparison (used after Phase 2)
# ---------------------------------------------------------------------------

def plot_reliability_comparison(
    stats_list: list[tuple[str, dict, float, float]],
    save_path: str,
) -> None:
    """Plot reliability diagrams for multiple models side by side.

    Args:
        stats_list: List of (model_name, bin_stats, ece, accuracy) tuples.
        save_path: Output PNG path.
    """
    n = len(stats_list)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5))
    if n == 1:
        axes = [axes]

    for ax, (name, bins, ece, acc) in zip(axes, stats_list):
        midpoints = [(b["lower"] + b["upper"]) / 2 for b in bins]
        avg_accs = [b["avg_accuracy"] for b in bins]
        bin_width = bins[0]["upper"] - bins[0]["lower"]

        ax.bar(midpoints, avg_accs, width=bin_width * 0.85,
               alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.5)
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5)
        ax.set_xlabel("Confidence", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{name}\nAcc={acc:.3f}  ECE={ece:.4f}", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
