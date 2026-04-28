"""Generic training loop: SGD with momentum + cosine-annealing LR schedule.

Used identically for the teacher (fine-tuning) and the baseline student
(training from scratch) — the only knobs that change are learning rate
and number of epochs.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict:
    """Run one training epoch. Returns {"loss": ..., "accuracy": ...}."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc="  train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()

    return {"loss": running_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate the model. Returns loss, accuracy, and per-sample softmax
    probabilities (needed downstream for ECE computation)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for inputs, targets in tqdm(loader, desc="  eval ", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()

        all_probs.append(torch.softmax(outputs, dim=1).cpu())
        all_targets.append(targets.cpu())

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "probs": torch.cat(all_probs),
        "targets": torch.cat(all_targets),
    }


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    device: torch.device | None = None,
    save_dir: str = "results/checkpoints",
    model_name: str = "model",
) -> dict:
    """Train *model* with SGD + cosine annealing and return rich results.

    Returns a dict containing:
        history      – per-epoch train/test loss & accuracy, plus LR
        best_accuracy – best test accuracy seen during training
        final_eval   – evaluation dict (with probs) from the best checkpoint
        training_time_seconds
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr,
        momentum=momentum, weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    history: dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
        "lr": [],
    }
    best_acc = 0.0
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\nTraining {model_name} on {device} for {epochs} epochs")
    print(f"Parameters: {n_params:,}")
    print("-" * 65)

    start = time.time()

    for epoch in range(1, epochs + 1):
        cur_lr = optimizer.param_groups[0]["lr"]

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_m = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_m["loss"])
        history["train_acc"].append(train_m["accuracy"])
        history["test_loss"].append(test_m["loss"])
        history["test_acc"].append(test_m["accuracy"])
        history["lr"].append(cur_lr)

        marker = ""
        if test_m["accuracy"] > best_acc:
            best_acc = test_m["accuracy"]
            torch.save(model.state_dict(), save_path / f"{model_name}_best.pth")
            marker = " *"

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"LR {cur_lr:.6f}  "
            f"Train {train_m['loss']:.4f} / {train_m['accuracy']:.4f}  "
            f"Test {test_m['loss']:.4f} / {test_m['accuracy']:.4f}{marker}"
        )

    elapsed = time.time() - start
    print("-" * 65)
    print(f"Done in {elapsed / 60:.1f} min  |  Best test acc: {best_acc:.4f}")

    # Save final-epoch checkpoint
    torch.save(model.state_dict(), save_path / f"{model_name}_final.pth")

    # Reload best checkpoint for the definitive evaluation
    model.load_state_dict(
        torch.load(save_path / f"{model_name}_best.pth", weights_only=True)
    )
    final_eval = evaluate(model, test_loader, criterion, device)

    return {
        "history": history,
        "best_accuracy": best_acc,
        "final_eval": final_eval,
        "training_time_seconds": elapsed,
        "n_params": n_params,
    }
