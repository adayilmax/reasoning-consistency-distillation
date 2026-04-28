#!/usr/bin/env python3
"""Phase 1b — Train a lightweight student CNN from scratch on CIFAR-10.

This serves as the *naive baseline*: cross-entropy only, no knowledge
distillation.  Phase 2 will compare the distilled student against this.

Run:
    uv run python scripts/02_train_baseline_student.py

Outputs:
    results/checkpoints/baseline_student_best.pth
    results/checkpoints/baseline_student_final.pth
    results/metrics/baseline_student_metrics.json
    results/metrics/baseline_student_calibration.json
    results/plots/baseline_student_training_curves.png
    results/plots/baseline_student_reliability_diagram.png
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cifar10 import get_dataloaders
from src.models.student import create_student
from src.training.trainer import train_model
from src.evaluation.metrics import compute_ece
from src.visualization.plots import plot_reliability_diagram, plot_training_curves

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
EPOCHS = 80
BATCH_SIZE = 128
LR = 0.1            # standard LR for training from scratch
WEIGHT_DECAY = 5e-4
MODEL_NAME = "baseline_student"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path("results")
    for sub in ("metrics", "plots", "checkpoints"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Phase 1b: Baseline Student (5-layer CNN, from scratch)")
    print(f"Device : {device}")
    print(f"Epochs : {EPOCHS}   LR : {LR}   Batch : {BATCH_SIZE}")
    print("=" * 65)

    # ---- data ----
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # ---- model ----
    student = create_student()

    # ---- train ----
    results = train_model(
        student,
        train_loader,
        test_loader,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device=device,
        model_name=MODEL_NAME,
    )

    # ---- calibration ----
    final = results["final_eval"]
    cal = compute_ece(final["probs"], final["targets"])

    # ---- persist metrics (JSON) ----
    metrics = {
        "model": "StudentCNN (5-layer, trained from scratch)",
        "parameters": results["n_params"],
        "epochs": EPOCHS,
        "learning_rate": LR,
        "best_test_accuracy": results["best_accuracy"],
        "final_test_accuracy": final["accuracy"],
        "final_test_loss": final["loss"],
        "ece": cal["ece"],
        "training_time_seconds": results["training_time_seconds"],
    }
    with open(results_dir / "metrics" / "baseline_student_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(results_dir / "metrics" / "baseline_student_calibration.json", "w") as f:
        json.dump(cal, f, indent=2)

    # ---- plots ----
    plot_training_curves(
        results["history"],
        "Baseline Student (5-layer CNN)",
        str(results_dir / "plots" / "baseline_student_training_curves.png"),
    )
    plot_reliability_diagram(
        cal["bin_stats"],
        "Baseline Student (5-layer CNN)",
        cal["ece"],
        final["accuracy"],
        str(results_dir / "plots" / "baseline_student_reliability_diagram.png"),
    )

    # ---- summary ----
    print("\n" + "=" * 65)
    print("RESULTS — Baseline Student (5-layer CNN)")
    print(f"  Parameters        : {results['n_params']:,}")
    print(f"  Best test accuracy: {results['best_accuracy']:.4f}")
    print(f"  ECE               : {cal['ece']:.4f}")
    print(f"  Training time     : {results['training_time_seconds'] / 60:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    main()
