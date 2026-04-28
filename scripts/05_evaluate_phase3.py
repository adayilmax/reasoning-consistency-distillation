#!/usr/bin/env python3
"""Phase 3 — Reasoning Consistency Evaluation.

Measures whether the distilled student preserves the teacher's *reasoning
process* — not just its accuracy — using three complementary probes:

  1. GradCAM saliency overlap (Spearman ρ, IoU on top-20% pixels)
     Headline result: examples where both models predict correctly but their
     attention maps diverge significantly.

  2. CKA (Centered Kernel Alignment) on intermediate representations
     Tracks how aligned internal features are across the depth of both models.

  3. Per-class accuracy and F1
     Identifies which CIFAR-10 categories show the largest reasoning gap.

Run:
    uv run python scripts/05_evaluate_phase3.py
    uv run python scripts/05_evaluate_phase3.py --temperature 4 --alpha 0.9

CLI args:
    --temperature   Which distilled-student checkpoint to load  (default: 4.0)
    --alpha         Distillation weight used during training    (default: 0.9)
    --batch-size    Mini-batch size for GradCAM passes          (default: 32)
    --n-cka         Number of samples for CKA computation       (default: 1000)
    --n-divergent   Number of divergence examples to save       (default: 12)
    --n-examples    Random examples for saliency overview plot  (default: 12)

Outputs (results/):
    metrics/phase3_saliency_{tag}.json      per-image Spearman + IoU
    metrics/phase3_cka_{tag}.json           CKA matrix
    metrics/phase3_per_class_{tag}.json     per-class accuracy, F1, saliency
    plots/phase3_divergence_{tag}.png       headline divergence grid
    plots/phase3_saliency_examples_{tag}.png overview of random examples
    plots/phase3_cka_heatmap_{tag}.png      CKA matrix heatmap
    plots/phase3_class_breakdown_{tag}.png  per-class alignment bar chart
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cifar10 import get_dataloaders, CLASSES
from src.models.teacher import create_teacher
from src.models.student import create_student
from src.evaluation.gradcam import (
    GradCAM,
    get_gradcam_layer_teacher,
    get_gradcam_layer_student,
)
from src.evaluation.saliency_metrics import compute_batch_saliency_metrics
from src.evaluation.cka import (
    get_teacher_cka_layers,
    get_student_cka_layers,
    extract_features,
    compute_cka_matrix,
)
from src.evaluation.metrics import compute_per_class_metrics
from src.visualization.saliency_plots import (
    plot_divergence_grid,
    plot_saliency_examples,
    plot_cka_heatmap,
    plot_class_breakdown,
)

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3: Reasoning Consistency")
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for GradCAM (smaller = less GPU memory)")
    p.add_argument("--n-cka", type=int, default=1000,
                   help="Samples for CKA (max 10000)")
    p.add_argument("--n-divergent", type=int, default=12,
                   help="Number of divergence examples to save")
    p.add_argument("--n-examples", type=int, default=12,
                   help="Number of random examples for overview plot")
    return p.parse_args()


# ---------------------------------------------------------------------------
# GradCAM pass over the full test set
# ---------------------------------------------------------------------------

def run_gradcam_pass(
    teacher: nn.Module,
    student: nn.Module,
    loader,
    device: torch.device,
) -> list[dict]:
    """Forward+backward GradCAM pass over the entire test loader.

    Returns a list of per-image result dicts (one per sample).
    Each dict contains:
        idx, true_label, teacher_pred, student_pred,
        spearman, iou, degenerate_a, degenerate_b,
        teacher_correct, student_correct, both_correct
    """
    cam_teacher = GradCAM(teacher, get_gradcam_layer_teacher(teacher))
    cam_student  = GradCAM(student,  get_gradcam_layer_student(student))

    teacher.to(device).eval()
    student.to(device).eval()

    records: list[dict] = []
    global_idx = 0

    for inputs, targets in tqdm(loader, desc="GradCAM pass"):
        inputs = inputs.to(device)
        # inputs.requires_grad = False by default — that is fine;
        # GradCAM only needs gradients w.r.t. intermediate activations.

        cams_t, logits_t = cam_teacher.generate(inputs)
        # _activations/_gradients are now None (freed inside generate()).
        # Empty the CUDA cache before the student pass so teacher's
        # intermediate buffers are fully reclaimed.
        teacher.zero_grad()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        cams_s, logits_s = cam_student.generate(inputs)
        student.zero_grad()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        preds_t = logits_t.argmax(dim=1)
        preds_s = logits_s.argmax(dim=1)

        # Compute saliency overlap metrics
        cams_t_np = cams_t.numpy()   # (N, H, W)
        cams_s_np = cams_s.numpy()
        sal_metrics = compute_batch_saliency_metrics(cams_t_np, cams_s_np)

        for i in range(len(targets)):
            tgt = int(targets[i])
            pt = int(preds_t[i])
            ps = int(preds_s[i])
            m = sal_metrics[i]
            records.append({
                "idx": global_idx + i,
                "true_label": tgt,
                "teacher_pred": pt,
                "student_pred": ps,
                "teacher_correct": pt == tgt,
                "student_correct": ps == tgt,
                "both_correct": (pt == tgt) and (ps == tgt),
                "spearman": m["spearman"],
                "iou": m["iou"],
                "degenerate_a": m["degenerate_a"],
                "degenerate_b": m["degenerate_b"],
            })

        global_idx += len(targets)

    cam_teacher.remove_hooks()
    cam_student.remove_hooks()
    return records


# ---------------------------------------------------------------------------
# Second pass: retrieve images + CAMs for specific indices
# ---------------------------------------------------------------------------

def retrieve_examples(
    teacher: nn.Module,
    student: nn.Module,
    loader,
    device: torch.device,
    target_indices: set[int],
) -> dict[int, dict]:
    """Re-run GradCAM on a specific subset of images (by dataset index)."""
    cam_teacher = GradCAM(teacher, get_gradcam_layer_teacher(teacher))
    cam_student  = GradCAM(student,  get_gradcam_layer_student(student))

    teacher.to(device).eval()
    student.to(device).eval()

    retrieved: dict[int, dict] = {}
    global_idx = 0

    for inputs, targets in loader:
        batch_indices = list(range(global_idx, global_idx + len(targets)))
        relevant = [i - global_idx for i in batch_indices if i in target_indices]
        if not relevant:
            global_idx += len(targets)
            continue

        inputs = inputs.to(device)
        cams_t, logits_t = cam_teacher.generate(inputs)
        teacher.zero_grad()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        cams_s, logits_s = cam_student.generate(inputs)
        student.zero_grad()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        preds_t = logits_t.argmax(dim=1)
        preds_s = logits_s.argmax(dim=1)
        sal_metrics = compute_batch_saliency_metrics(
            cams_t.numpy(), cams_s.numpy()
        )

        for local_i in relevant:
            g_i = global_idx + local_i
            retrieved[g_i] = {
                "image": inputs[local_i].cpu().numpy(),
                "cam_teacher": cams_t[local_i].numpy(),
                "cam_student": cams_s[local_i].numpy(),
                "true_label": int(targets[local_i]),
                "teacher_pred": int(preds_t[local_i]),
                "student_pred": int(preds_s[local_i]),
                "spearman": sal_metrics[local_i]["spearman"],
                "iou": sal_metrics[local_i]["iou"],
            }

        global_idx += len(targets)
        if len(retrieved) >= len(target_indices):
            break

    cam_teacher.remove_hooks()
    cam_student.remove_hooks()
    return retrieved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(SEED)

    T = args.temperature
    alpha = args.alpha
    tag = f"T{T}_a{alpha}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path("results")
    for sub in ("metrics", "plots"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"Phase 3: Reasoning Consistency  (distilled T={T}, α={alpha})")
    print(f"Device : {device}")
    print("=" * 65)

    ckpt_dir = results_dir / "checkpoints"

    # ---- Load teacher ----
    teacher_ckpt = ckpt_dir / "teacher_resnet50_best.pth"
    if not teacher_ckpt.exists():
        raise FileNotFoundError(f"Missing: {teacher_ckpt}. Run 01_train_teacher.py first.")
    teacher = create_teacher(pretrained=False)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location="cpu", weights_only=True))
    print(f"Loaded teacher from {teacher_ckpt}")

    # ---- Load distilled student ----
    student_ckpt = ckpt_dir / f"distilled_student_{tag}_best.pth"
    if not student_ckpt.exists():
        raise FileNotFoundError(
            f"Missing: {student_ckpt}. Run 03_train_distilled_student.py first."
        )
    student = create_student()
    student.load_state_dict(torch.load(student_ckpt, map_location="cpu", weights_only=True))
    print(f"Loaded student from {student_ckpt}")

    # ---- Data ----
    # shuffle=False is critical: we need stable global indices
    _, test_loader = get_dataloaders(batch_size=args.batch_size)

    # ====================================================================
    # Part 1 — GradCAM pass over full test set
    # ====================================================================
    print("\n[1/3] Running GradCAM over the full test set...")
    records = run_gradcam_pass(teacher, student, test_loader, device)

    # Aggregate overall saliency metrics (both-correct subset)
    both_correct = [r for r in records if r["both_correct"] and not r["degenerate_a"] and not r["degenerate_b"]]
    n_both = len(both_correct)
    all_rho = [r["spearman"] for r in both_correct]
    all_iou = [r["iou"] for r in both_correct]

    print(f"  Total test samples   : {len(records)}")
    print(f"  Both-correct samples : {n_both}")
    print(f"  Mean Spearman ρ      : {np.mean(all_rho):.4f} ± {np.std(all_rho):.4f}")
    print(f"  Mean IoU (top-20%)   : {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")

    # ---- Per-class saliency breakdown ----
    per_class_saliency: list[dict] = []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_records = [r for r in both_correct if r["true_label"] == cls_idx]
        if cls_records:
            cls_rho = np.mean([r["spearman"] for r in cls_records])
            cls_iou = np.mean([r["iou"] for r in cls_records])
        else:
            cls_rho = cls_iou = 0.0
        per_class_saliency.append({
            "class_idx": cls_idx,
            "name": cls_name,
            "mean_spearman": float(cls_rho),
            "mean_iou": float(cls_iou),
            "n_both_correct": len(cls_records),
        })

    # ---- Per-class accuracy + F1 (from stored predictions) ----
    all_probs_t  = []
    all_probs_s  = []
    all_targets_ = []

    # We need probabilities — re-run a standard eval pass (no gradients needed)
    print("\n  Computing per-class F1 via standard eval pass...")
    teacher.eval(); student.eval()
    teacher.to(device); student.to(device)
    with torch.no_grad():
        for inputs, targets in tqdm(
            get_dataloaders(batch_size=128)[1], desc="  eval", leave=False
        ):
            inputs = inputs.to(device)
            logits_t = teacher(inputs).cpu()
            logits_s = student(inputs).cpu()
            all_probs_t.append(torch.softmax(logits_t, dim=1))
            all_probs_s.append(torch.softmax(logits_s, dim=1))
            all_targets_.append(targets)

    probs_t_all = torch.cat(all_probs_t)
    probs_s_all = torch.cat(all_probs_s)
    targets_all = torch.cat(all_targets_)

    teacher_cls = compute_per_class_metrics(probs_t_all, targets_all, class_names=CLASSES)
    student_cls = compute_per_class_metrics(probs_s_all, targets_all, class_names=CLASSES)

    # Attach F1 to per-class saliency records
    for i, cls_entry in enumerate(per_class_saliency):
        cls_entry["teacher_f1"] = teacher_cls["per_class"][i]["f1"]
        cls_entry["student_f1"] = student_cls["per_class"][i]["f1"]
        cls_entry["teacher_accuracy"] = teacher_cls["per_class"][i]["accuracy"]
        cls_entry["student_accuracy"] = student_cls["per_class"][i]["accuracy"]

    # ---- Save saliency metrics JSON ----
    saliency_out = {
        "distillation_config": {"temperature": T, "alpha": alpha},
        "n_test_samples": len(records),
        "n_both_correct": n_both,
        "mean_spearman": float(np.mean(all_rho)),
        "std_spearman": float(np.std(all_rho)),
        "mean_iou": float(np.mean(all_iou)),
        "std_iou": float(np.std(all_iou)),
        "per_class": per_class_saliency,
        "per_sample": [
            {k: v for k, v in r.items()}
            for r in records
        ],
    }
    sal_path = results_dir / "metrics" / f"phase3_saliency_{tag}.json"
    with open(sal_path, "w") as f:
        json.dump(saliency_out, f, indent=2)

    # ---- Save per-class JSON ----
    per_class_out = {
        "teacher": teacher_cls,
        "student": student_cls,
        "saliency": per_class_saliency,
    }
    cls_path = results_dir / "metrics" / f"phase3_per_class_{tag}.json"
    with open(cls_path, "w") as f:
        json.dump(per_class_out, f, indent=2)

    # ====================================================================
    # Part 2 — Divergence examples: second GradCAM pass for images + CAMs
    # ====================================================================
    print(f"\n[2/3] Retrieving top-{args.n_divergent} divergence examples...")

    # Sort both-correct by Spearman ascending (most divergent first)
    both_correct_sorted = sorted(both_correct, key=lambda r: r["spearman"])
    divergent_indices = {r["idx"] for r in both_correct_sorted[:args.n_divergent]}

    # Also pick random examples for the overview plot
    random_pool = [r for r in both_correct if r["idx"] not in divergent_indices]
    random.shuffle(random_pool)
    random_indices = {r["idx"] for r in random_pool[:args.n_examples]}

    all_needed = divergent_indices | random_indices
    retrieved = retrieve_examples(
        teacher, student, test_loader, device, all_needed
    )

    # ---- Divergence grid (headline figure) ----
    div_examples = [
        retrieved[idx] for idx in
        sorted(divergent_indices, key=lambda i: retrieved[i]["spearman"])
        if idx in retrieved
    ]
    plot_divergence_grid(
        div_examples[:args.n_divergent],
        str(results_dir / "plots" / f"phase3_divergence_{tag}.png"),
        class_names=CLASSES,
    )
    print(f"  Divergence grid saved (lowest ρ = {div_examples[0]['spearman']:.4f})")

    # ---- Random examples overview ----
    rnd_examples = [retrieved[idx] for idx in random_indices if idx in retrieved]
    plot_saliency_examples(
        rnd_examples[:args.n_examples],
        str(results_dir / "plots" / f"phase3_saliency_examples_{tag}.png"),
        class_names=CLASSES,
    )

    # ---- Per-class breakdown plot ----
    plot_class_breakdown(
        per_class_saliency,
        class_names=CLASSES,
        save_path=str(results_dir / "plots" / f"phase3_class_breakdown_{tag}.png"),
    )

    # ====================================================================
    # Part 3 — CKA
    # ====================================================================
    print(f"\n[3/3] Computing CKA on {args.n_cka} test samples...")
    # Use a fresh loader with larger batch for speed (no gradients needed)
    _, cka_loader = get_dataloaders(batch_size=128)

    teacher_feats = extract_features(
        teacher,
        get_teacher_cka_layers(teacher),
        cka_loader,
        device,
        n_samples=args.n_cka,
    )
    student_feats = extract_features(
        student,
        get_student_cka_layers(student),
        cka_loader,
        device,
        n_samples=args.n_cka,
    )

    cka_matrix = compute_cka_matrix(teacher_feats, student_feats)

    # Print CKA diagonal (same-resolution layer pairs)
    print("\n  CKA — diagonal layer pairs (same spatial resolution):")
    diagonal_pairs = [
        ("layer1", "block1"), ("layer2", "block2"),
        ("layer3", "block3"), ("layer4", "block5"),
        ("pre_fc", "pre_cls"),
    ]
    for t_name, s_name in diagonal_pairs:
        if t_name in cka_matrix and s_name in cka_matrix[t_name]:
            print(f"    {t_name:8s} ↔ {s_name:8s} : {cka_matrix[t_name][s_name]:.4f}")

    # Save CKA JSON
    cka_out = {
        "n_samples": args.n_cka,
        "teacher_layers": list(teacher_feats.keys()),
        "student_layers": list(student_feats.keys()),
        "matrix": cka_matrix,
        "diagonal_pairs": [
            {
                "teacher_layer": t, "student_layer": s,
                "cka": cka_matrix[t][s],
            }
            for t, s in diagonal_pairs
            if t in cka_matrix and s in cka_matrix[t]
        ],
    }
    cka_path = results_dir / "metrics" / f"phase3_cka_{tag}.json"
    with open(cka_path, "w") as f:
        json.dump(cka_out, f, indent=2)

    # CKA heatmap plot
    plot_cka_heatmap(
        cka_matrix,
        str(results_dir / "plots" / f"phase3_cka_heatmap_{tag}.png"),
    )

    # ====================================================================
    # Final summary
    # ====================================================================
    print("\n" + "=" * 65)
    print("PHASE 3 SUMMARY")
    print(f"  Both-correct samples : {n_both} / {len(records)}")
    print(f"  Mean Spearman ρ      : {np.mean(all_rho):.4f} ± {np.std(all_rho):.4f}")
    print(f"  Mean IoU (top-20%)   : {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")

    # Worst class by Spearman
    worst = min(per_class_saliency, key=lambda c: c["mean_spearman"])
    best  = max(per_class_saliency, key=lambda c: c["mean_spearman"])
    print(f"  Worst class (ρ)      : {worst['name']} ({worst['mean_spearman']:.4f})")
    print(f"  Best  class (ρ)      : {best['name']}  ({best['mean_spearman']:.4f})")

    # Diagonal CKA
    diag_cka = [
        cka_matrix[t][s]
        for t, s in diagonal_pairs
        if t in cka_matrix and s in cka_matrix.get(t, {})
    ]
    if diag_cka:
        print(f"  Mean diagonal CKA    : {np.mean(diag_cka):.4f}")

    print(f"\n  Outputs saved to {results_dir}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
