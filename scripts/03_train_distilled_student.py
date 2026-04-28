#!/usr/bin/env python3
"""Phase 2 — Train a student with Hinton-style Knowledge Distillation.

The teacher (ResNet-50) is frozen. The student is trained from scratch using
a weighted combination of soft-target KL divergence and hard-label CE loss.

Run:
    uv run python scripts/03_train_distilled_student.py
    uv run python scripts/03_train_distilled_student.py --temperature 4 --alpha 0.9

CLI args (all optional — defaults match Hinton et al.):
    --temperature   Softening temperature T  (default: 4)
    --alpha         Weight for soft-target term (default: 0.9)
    --epochs        Training epochs (default: 80)
    --lr            Initial learning rate (default: 0.1)

Outputs:
    results/checkpoints/distilled_student_T{T}_a{alpha}_best.pth
    results/checkpoints/distilled_student_T{T}_a{alpha}_final.pth
    results/metrics/distilled_student_T{T}_a{alpha}_metrics.json
    results/metrics/distilled_student_T{T}_a{alpha}_calibration.json
    results/plots/distilled_student_T{T}_a{alpha}_training_curves.png
    results/plots/distilled_student_T{T}_a{alpha}_reliability_diagram.png
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cifar10 import get_dataloaders
from src.models.teacher import create_teacher
from src.models.student import create_student
from src.training.distillation import train_with_distillation
from src.evaluation.metrics import compute_ece
from src.visualization.plots import plot_reliability_diagram, plot_training_curves

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2: Knowledge Distillation")
    p.add_argument("--temperature", type=float, default=4.0,
                   help="Distillation temperature T (default: 4)")
    p.add_argument("--alpha", type=float, default=0.9,
                   help="Soft-target loss weight α (default: 0.9)")
    p.add_argument("--epochs", type=int, default=80,
                   help="Number of training epochs (default: 80)")
    p.add_argument("--lr", type=float, default=0.1,
                   help="Initial SGD learning rate (default: 0.1)")
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = args.temperature
    alpha = args.alpha
    run_tag = f"T{T}_a{alpha}"
    model_name = f"distilled_student_{run_tag}"

    results_dir = Path("results")
    for sub in ("metrics", "plots", "checkpoints"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"Phase 2: Knowledge Distillation  (T={T}, α={alpha})")
    print(f"Device : {device}")
    print(f"Epochs : {args.epochs}   LR : {args.lr}   Batch : {args.batch_size}")
    print("=" * 65)

    # ---- data ----
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # ---- teacher (frozen) ----
    teacher_ckpt = results_dir / "checkpoints" / "teacher_resnet50_best.pth"
    if not teacher_ckpt.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {teacher_ckpt}\n"
            "Run scripts/01_train_teacher.py first."
        )
    teacher = create_teacher(pretrained=False)
    teacher.load_state_dict(
        torch.load(teacher_ckpt, map_location="cpu", weights_only=True)
    )
    print(f"Loaded teacher from {teacher_ckpt}")

    # ---- student (fresh init) ----
    student = create_student()

    # ---- distillation training ----
    results = train_with_distillation(
        student,
        teacher,
        train_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.lr,
        temperature=T,
        alpha=alpha,
        device=device,
        model_name=model_name,
    )

    # ---- calibration ----
    final = results["final_eval"]
    cal = compute_ece(final["probs"], final["targets"])

    # ---- persist metrics ----
    metrics = {
        "model": "StudentCNN (distilled from ResNet-50)",
        "parameters": results["n_params"],
        "temperature": T,
        "alpha": alpha,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "best_test_accuracy": results["best_accuracy"],
        "final_test_accuracy": final["accuracy"],
        "final_test_loss": final["loss"],
        "ece": cal["ece"],
        "training_time_seconds": results["training_time_seconds"],
    }
    with open(results_dir / "metrics" / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(results_dir / "metrics" / f"{model_name}_calibration.json", "w") as f:
        json.dump(cal, f, indent=2)

    # ---- plots ----
    plot_training_curves(
        results["history"],
        f"Distilled Student (T={T}, α={alpha})",
        str(results_dir / "plots" / f"{model_name}_training_curves.png"),
    )
    plot_reliability_diagram(
        cal["bin_stats"],
        f"Distilled Student (T={T}, α={alpha})",
        cal["ece"],
        final["accuracy"],
        str(results_dir / "plots" / f"{model_name}_reliability_diagram.png"),
    )

    # ---- summary ----
    print("\n" + "=" * 65)
    print(f"RESULTS — Distilled Student (T={T}, α={alpha})")
    print(f"  Parameters        : {results['n_params']:,}")
    print(f"  Best test accuracy: {results['best_accuracy']:.4f}")
    print(f"  ECE               : {cal['ece']:.4f}")
    print(f"  Training time     : {results['training_time_seconds'] / 60:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    main()
