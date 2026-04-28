#!/usr/bin/env python3
"""Phase 4 — Train student variants for sensitivity analysis.

Trains the following checkpoints (skips any that already exist):

  Compression axis  (fixed T=4, α=0.9):
    student_tiny_T4.0_a0.9      — 3-block CNN, ~95 k params
    student_small_T4.0_a0.9     — 4-block CNN, ~242 k params
    [student_medium_T4.0_a0.9 already exists as distilled_student_T4.0_a0.9]

  Temperature axis  (fixed variant=medium, α=0.9):
    student_medium_T2.0_a0.9    — lower temperature, sharper soft targets
    student_medium_T8.0_a0.9    — higher temperature, softer soft targets
    [student_medium_T4.0_a0.9 already exists as distilled_student_T4.0_a0.9]

Run:
    uv run python scripts/06_train_variants.py
    uv run python scripts/06_train_variants.py --epochs 80 --alpha 0.9

All outputs follow the naming convention:
    results/checkpoints/student_{variant}_T{T}_a{alpha}_best.pth
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
from src.models.student import VARIANT_REGISTRY
from src.training.distillation import train_with_distillation
from src.evaluation.metrics import compute_ece
from src.visualization.plots import plot_training_curves, plot_reliability_diagram

SEED = 42
DEFAULT_EPOCHS = 80
DEFAULT_ALPHA = 0.9
DEFAULT_LR = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4: Train student variants")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


def checkpoint_exists(save_dir: Path, model_name: str) -> bool:
    return (save_dir / f"{model_name}_best.pth").exists()


def train_one_variant(
    model_name: str,
    student,
    teacher,
    train_loader,
    test_loader,
    *,
    epochs: int,
    lr: float,
    temperature: float,
    alpha: float,
    device: torch.device,
    results_dir: Path,
) -> None:
    """Train a single variant and save all outputs."""
    save_dir = results_dir / "checkpoints"

    results = train_with_distillation(
        student,
        teacher,
        train_loader,
        test_loader,
        epochs=epochs,
        lr=lr,
        temperature=temperature,
        alpha=alpha,
        device=device,
        save_dir=str(save_dir),
        model_name=model_name,
    )

    final = results["final_eval"]
    cal = compute_ece(final["probs"], final["targets"])

    metrics = {
        "model_name": model_name,
        "parameters": results["n_params"],
        "temperature": temperature,
        "alpha": alpha,
        "epochs": epochs,
        "best_test_accuracy": results["best_accuracy"],
        "final_test_accuracy": final["accuracy"],
        "ece": cal["ece"],
        "training_time_seconds": results["training_time_seconds"],
    }
    with open(results_dir / "metrics" / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(results_dir / "metrics" / f"{model_name}_calibration.json", "w") as f:
        json.dump(cal, f, indent=2)

    plot_training_curves(
        results["history"],
        model_name.replace("_", " "),
        str(results_dir / "plots" / f"{model_name}_training_curves.png"),
    )
    plot_reliability_diagram(
        cal["bin_stats"],
        model_name.replace("_", " "),
        cal["ece"],
        final["accuracy"],
        str(results_dir / "plots" / f"{model_name}_reliability_diagram.png"),
    )

    print(
        f"  {model_name:<40s}  "
        f"params={results['n_params']:>8,}  "
        f"acc={results['best_accuracy']:.4f}  "
        f"ECE={cal['ece']:.4f}"
    )


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path("results")
    for sub in ("metrics", "plots", "checkpoints"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)

    # ---- Load teacher (frozen throughout) ----
    teacher_ckpt = results_dir / "checkpoints" / "teacher_resnet50_best.pth"
    if not teacher_ckpt.exists():
        raise FileNotFoundError(f"Missing: {teacher_ckpt}. Run 01_train_teacher.py first.")
    teacher = create_teacher(pretrained=False)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location="cpu", weights_only=True))
    print(f"Loaded teacher from {teacher_ckpt}")

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    save_dir = results_dir / "checkpoints"

    # ------------------------------------------------------------------
    # Define all runs
    # ------------------------------------------------------------------
    # (variant, temperature) — medium T=4 is already done under a different name
    runs: list[tuple[str, float]] = [
        ("tiny",   4.0),
        ("small",  4.0),
        ("medium", 2.0),
        ("medium", 8.0),
    ]

    print("\n" + "=" * 65)
    print("Phase 4: Training student variants")
    print(f"Device: {device}   Epochs: {args.epochs}   alpha: {args.alpha}")
    print("=" * 65)

    completed = []
    skipped = []

    for variant, T in runs:
        model_name = f"student_{variant}_T{T}_a{args.alpha}"

        if checkpoint_exists(save_dir, model_name):
            print(f"  SKIP  {model_name}  (checkpoint exists)")
            skipped.append(model_name)
            continue

        print(f"\n--- Training {model_name} ---")
        factory = VARIANT_REGISTRY[variant]["factory"]
        student = factory()

        train_one_variant(
            model_name,
            student,
            teacher,
            train_loader,
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
            temperature=T,
            alpha=args.alpha,
            device=device,
            results_dir=results_dir,
        )
        completed.append(model_name)

    print("\n" + "=" * 65)
    print(f"Trained : {len(completed)} variants")
    print(f"Skipped : {len(skipped)} (already exist)")
    if completed:
        print("New checkpoints:")
        for name in completed:
            print(f"  results/checkpoints/{name}_best.pth")
    print("=" * 65)
    print("\nNow run:  uv run python scripts/07_analyze_phase4.py")


if __name__ == "__main__":
    main()
