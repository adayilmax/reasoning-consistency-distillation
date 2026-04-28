#!/usr/bin/env python3
"""Phase 1a — Fine-tune a pretrained ResNet-50 teacher on CIFAR-10.

Run:
    uv run python scripts/01_train_teacher.py

Outputs:
    results/checkpoints/teacher_resnet50_best.pth
    results/checkpoints/teacher_resnet50_final.pth
    results/metrics/teacher_metrics.json
    results/metrics/teacher_calibration.json
    results/plots/teacher_training_curves.png
    results/plots/teacher_reliability_diagram.png
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Allow direct invocation without `uv run` (fallback for editable-install)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cifar10 import get_dataloaders
from src.models.teacher import create_teacher
from src.training.trainer import train_model
from src.evaluation.metrics import compute_ece
from src.visualization.plots import plot_reliability_diagram, plot_training_curves

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
EPOCHS = 20
BATCH_SIZE = 128
LR = 0.01          # low LR for fine-tuning pretrained backbone
WEIGHT_DECAY = 5e-4
MODEL_NAME = "teacher_resnet50"


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
    print("Phase 1a: Teacher (ResNet-50, pretrained) on CIFAR-10")
    print(f"Device : {device}")
    print(f"Epochs : {EPOCHS}   LR : {LR}   Batch : {BATCH_SIZE}")
    print("=" * 65)

    # ---- data ----
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # ---- model ----
    teacher = create_teacher(pretrained=True)

    # ---- train ----
    results = train_model(
        teacher,
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
        "model": "ResNet-50 (ImageNet-pretrained, CIFAR-adapted stem)",
        "parameters": results["n_params"],
        "epochs": EPOCHS,
        "learning_rate": LR,
        "best_test_accuracy": results["best_accuracy"],
        "final_test_accuracy": final["accuracy"],
        "final_test_loss": final["loss"],
        "ece": cal["ece"],
        "training_time_seconds": results["training_time_seconds"],
    }
    with open(results_dir / "metrics" / "teacher_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(results_dir / "metrics" / "teacher_calibration.json", "w") as f:
        json.dump(cal, f, indent=2)

    # ---- plots ----
    plot_training_curves(
        results["history"],
        "Teacher (ResNet-50)",
        str(results_dir / "plots" / "teacher_training_curves.png"),
    )
    plot_reliability_diagram(
        cal["bin_stats"],
        "Teacher (ResNet-50)",
        cal["ece"],
        final["accuracy"],
        str(results_dir / "plots" / "teacher_reliability_diagram.png"),
    )

    # ---- summary ----
    print("\n" + "=" * 65)
    print("RESULTS — Teacher (ResNet-50)")
    print(f"  Parameters        : {results['n_params']:,}")
    print(f"  Best test accuracy: {results['best_accuracy']:.4f}")
    print(f"  ECE               : {cal['ece']:.4f}")
    print(f"  Training time     : {results['training_time_seconds'] / 60:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    main()
