#!/usr/bin/env python3
"""Phase 2 — Compare teacher, baseline student, and distilled student.

Loads the best checkpoint for each of the three models, runs evaluation,
and produces:
  - A JSON summary table (accuracy + ECE for all three)
  - A side-by-side reliability diagram (3 panels)
  - A grouped bar chart comparing accuracy and ECE

Requires:
    results/checkpoints/teacher_resnet50_best.pth
    results/checkpoints/baseline_student_best.pth
    results/checkpoints/distilled_student_T{T}_a{alpha}_best.pth

Run:
    uv run python scripts/04_compare_phase2.py
    uv run python scripts/04_compare_phase2.py --temperature 4 --alpha 0.9
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cifar10 import get_dataloaders
from src.models.teacher import create_teacher
from src.models.student import create_student
from src.training.trainer import evaluate
from src.evaluation.metrics import compute_ece
from src.visualization.plots import plot_reliability_comparison


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2: Model comparison")
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


def load_and_eval(
    model: nn.Module,
    ckpt_path: Path,
    test_loader,
    device: torch.device,
    label: str,
) -> dict:
    """Load checkpoint, evaluate, and compute ECE."""
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    ev = evaluate(model, test_loader, criterion, device)
    cal = compute_ece(ev["probs"], ev["targets"])
    print(f"  {label:<35s}  acc={ev['accuracy']:.4f}  ECE={cal['ece']:.4f}")
    return {
        "label": label,
        "accuracy": ev["accuracy"],
        "loss": ev["loss"],
        "ece": cal["ece"],
        "bin_stats": cal["bin_stats"],
    }


def plot_comparison_bars(
    records: list[dict],
    save_path: str,
) -> None:
    """Grouped bar chart: accuracy and ECE for each model."""
    labels = [r["label"] for r in records]
    accs = [r["accuracy"] for r in records]
    eces = [r["ece"] for r in records]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, accs, width, label="Accuracy",
                    color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars2 = ax2.bar(x + width / 2, eces, width, label="ECE",
                    color="crimson", alpha=0.75, edgecolor="black", linewidth=0.5)

    # Annotate bars
    for bar in bars1:
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
        )
    for bar in bars2:
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9,
        )

    ax1.set_xlabel("Model", fontsize=11)
    ax1.set_ylabel("Test Accuracy", fontsize=11, color="steelblue")
    ax2.set_ylabel("ECE (lower is better)", fontsize=11, color="crimson")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0, 1.08)
    ax2.set_ylim(0, max(eces) * 2.2 if max(eces) > 0 else 0.1)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="crimson")

    # Combined legend
    lines = [bars1, bars2]
    ax1.legend(lines, ["Accuracy", "ECE"], loc="upper left", fontsize=9)
    ax1.set_title("Phase 2 — Model Comparison: Accuracy & Calibration", fontsize=12)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = args.temperature
    alpha = args.alpha
    run_tag = f"T{T}_a{alpha}"

    results_dir = Path("results")
    ckpt_dir = results_dir / "checkpoints"

    print("=" * 65)
    print(f"Phase 2: Comparison  (distilled with T={T}, α={alpha})")
    print(f"Device : {device}")
    print("=" * 65)

    _, test_loader = get_dataloaders(batch_size=args.batch_size)

    records = []

    # ---- Teacher ----
    teacher_ckpt = ckpt_dir / "teacher_resnet50_best.pth"
    if not teacher_ckpt.exists():
        raise FileNotFoundError(f"Missing: {teacher_ckpt}")
    records.append(load_and_eval(
        create_teacher(pretrained=False), teacher_ckpt,
        test_loader, device, "Teacher (ResNet-50)",
    ))

    # ---- Baseline student ----
    baseline_ckpt = ckpt_dir / "baseline_student_best.pth"
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"Missing: {baseline_ckpt}")
    records.append(load_and_eval(
        create_student(), baseline_ckpt,
        test_loader, device, "Baseline Student (CE only)",
    ))

    # ---- Distilled student ----
    distill_ckpt = ckpt_dir / f"distilled_student_{run_tag}_best.pth"
    if not distill_ckpt.exists():
        raise FileNotFoundError(f"Missing: {distill_ckpt}")
    records.append(load_and_eval(
        create_student(), distill_ckpt,
        test_loader, device, f"Distilled Student (T={T}, α={alpha})",
    ))

    # ---- Comparison bar chart ----
    plot_comparison_bars(
        records,
        str(results_dir / "plots" / f"phase2_comparison_{run_tag}.png"),
    )

    # ---- Side-by-side reliability diagrams ----
    plot_reliability_comparison(
        [(r["label"], r["bin_stats"], r["ece"], r["accuracy"]) for r in records],
        str(results_dir / "plots" / f"phase2_reliability_{run_tag}.png"),
    )

    # ---- JSON summary ----
    summary = {
        "distillation_config": {"temperature": T, "alpha": alpha},
        "models": [
            {k: v for k, v in r.items() if k != "bin_stats"}
            for r in records
        ],
        "delta_accuracy_distilled_vs_baseline": (
            records[2]["accuracy"] - records[1]["accuracy"]
        ),
        "delta_ece_distilled_vs_baseline": (
            records[2]["ece"] - records[1]["ece"]
        ),
    }
    out_path = results_dir / "metrics" / f"phase2_summary_{run_tag}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Console summary ----
    print("\n" + "=" * 65)
    print("PHASE 2 SUMMARY")
    for r in records:
        print(f"  {r['label']:<40s}  acc={r['accuracy']:.4f}  ECE={r['ece']:.4f}")
    print(f"\n  Δ Accuracy  (distilled − baseline): "
          f"{summary['delta_accuracy_distilled_vs_baseline']:+.4f}")
    print(f"  Δ ECE       (distilled − baseline): "
          f"{summary['delta_ece_distilled_vs_baseline']:+.4f}")
    print(f"\n  Results saved to {results_dir / 'metrics' / out_path.name}")
    print("=" * 65)


if __name__ == "__main__":
    main()
