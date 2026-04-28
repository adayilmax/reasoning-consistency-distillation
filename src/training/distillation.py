"""Hinton-style Knowledge Distillation loss and training loop.

The KD loss is a weighted combination of two terms:

    L = α · T² · KL(σ(z_s/T) ‖ σ(z_t/T))   ← soft-target loss
      + (1-α) · CE(z_s, y)                   ← hard-label loss

where:
    z_s   — student logits
    z_t   — teacher logits (detached; teacher is frozen during training)
    T     — temperature (higher T → softer probability distribution)
    α     — distillation weight (α=1: pure KD, α=0: pure CE)
    T²    — compensates for the 1/T² gradient scale reduction from softening

Reference: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015).
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# KD loss
# ---------------------------------------------------------------------------

def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    alpha: float,
) -> tuple[torch.Tensor, dict]:
    """Compute the combined KD + CE loss.

    Args:
        student_logits: Raw student outputs, shape (N, C).
        teacher_logits: Raw teacher outputs, shape (N, C). Must be detached.
        targets: Ground-truth class indices, shape (N,).
        temperature: Softening temperature T.
        alpha: Weight for the soft-target term (1-alpha for hard-label term).

    Returns:
        (loss, component_dict) where component_dict has "soft_loss" and
        "hard_loss" scalars for monitoring.
    """
    # Soft-target KL divergence
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
    soft_loss = soft_loss * (temperature ** 2)  # gradient magnitude correction

    # Hard-label cross-entropy
    hard_loss = F.cross_entropy(student_logits, targets)

    loss = alpha * soft_loss + (1.0 - alpha) * hard_loss

    return loss, {
        "soft_loss": soft_loss.item(),
        "hard_loss": hard_loss.item(),
    }


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def distill_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    temperature: float,
    alpha: float,
    device: torch.device,
) -> dict:
    """One epoch of distillation training.

    Teacher is always in eval mode and its gradients are disabled.
    Returns per-epoch averages of total loss, soft loss, hard loss, accuracy.
    """
    student.train()
    teacher.eval()

    running = {"loss": 0.0, "soft": 0.0, "hard": 0.0}
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc="  distill", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            teacher_logits = teacher(inputs)

        optimizer.zero_grad()
        student_logits = student(inputs)

        loss, components = kd_loss(
            student_logits,
            teacher_logits.detach(),
            targets,
            temperature,
            alpha,
        )
        loss.backward()
        optimizer.step()

        n = inputs.size(0)
        running["loss"] += loss.item() * n
        running["soft"] += components["soft_loss"] * n
        running["hard"] += components["hard_loss"] * n

        _, preds = student_logits.max(1)
        total += n
        correct += preds.eq(targets).sum().item()

    return {
        "loss": running["loss"] / total,
        "soft_loss": running["soft"] / total,
        "hard_loss": running["hard"] / total,
        "accuracy": correct / total,
    }


# ---------------------------------------------------------------------------
# Full distillation pipeline
# ---------------------------------------------------------------------------

def train_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    temperature: float,
    alpha: float,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    device: torch.device | None = None,
    save_dir: str = "results/checkpoints",
    model_name: str = "distilled_student",
) -> dict:
    """Train *student* using Hinton-style KD against a frozen *teacher*.

    Returns the same result schema as ``train_model`` in trainer.py, plus
    distillation-specific history fields (soft_loss, hard_loss per epoch).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Evaluate the test set using standard CE (same as baseline for fair compare)
    ce_criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        student.parameters(), lr=lr,
        momentum=momentum, weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    history: dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "soft_loss": [], "hard_loss": [],
        "test_loss": [], "test_acc": [],
        "lr": [],
    }
    best_acc = 0.0
    n_params = sum(p.numel() for p in student.parameters())

    print(f"\nDistilling into {model_name} on {device} for {epochs} epochs")
    print(f"Parameters: {n_params:,}   T={temperature}   α={alpha}")
    print("-" * 65)

    # Import evaluate here to avoid circular import
    from src.training.trainer import evaluate

    start = time.time()

    for epoch in range(1, epochs + 1):
        cur_lr = optimizer.param_groups[0]["lr"]

        train_m = distill_one_epoch(
            student, teacher, train_loader, optimizer,
            temperature, alpha, device,
        )
        test_m = evaluate(student, test_loader, ce_criterion, device)
        scheduler.step()

        history["train_loss"].append(train_m["loss"])
        history["train_acc"].append(train_m["accuracy"])
        history["soft_loss"].append(train_m["soft_loss"])
        history["hard_loss"].append(train_m["hard_loss"])
        history["test_loss"].append(test_m["loss"])
        history["test_acc"].append(test_m["accuracy"])
        history["lr"].append(cur_lr)

        marker = ""
        if test_m["accuracy"] > best_acc:
            best_acc = test_m["accuracy"]
            torch.save(student.state_dict(), save_path / f"{model_name}_best.pth")
            marker = " *"

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"LR {cur_lr:.6f}  "
            f"KD {train_m['soft_loss']:.4f}  "
            f"CE {train_m['hard_loss']:.4f}  "
            f"Test {test_m['accuracy']:.4f}{marker}"
        )

    elapsed = time.time() - start
    print("-" * 65)
    print(f"Done in {elapsed / 60:.1f} min  |  Best test acc: {best_acc:.4f}")

    torch.save(student.state_dict(), save_path / f"{model_name}_final.pth")

    # Reload best for definitive evaluation
    student.load_state_dict(
        torch.load(save_path / f"{model_name}_best.pth", weights_only=True)
    )
    final_eval = evaluate(student, test_loader, ce_criterion, device)

    return {
        "history": history,
        "best_accuracy": best_acc,
        "final_eval": final_eval,
        "training_time_seconds": elapsed,
        "n_params": n_params,
    }
