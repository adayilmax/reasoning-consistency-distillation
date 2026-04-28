#!/usr/bin/env python3
"""Phase 4 — Sensitivity analysis: compression depth and temperature.

Loads all trained variant checkpoints, runs GradCAM saliency evaluation
for each against the frozen teacher, and produces four result artefacts:

  Compression analysis  (T=4 fixed, variant = tiny / small / medium)
    plots/phase4_compression_curve.png
    metrics/phase4_compression.json

  Temperature analysis  (medium fixed, T = 2 / 4 / 8)
    plots/phase4_temperature_effect.png
    metrics/phase4_temperature.json

  Per-class reasoning gap heatmap  (all variants × all classes)
    plots/phase4_class_gap_heatmap.png
    metrics/phase4_class_gap.json

  Full summary
    metrics/phase4_summary.json

Run:
    uv run python scripts/07_analyze_phase4.py

Optional flags:
    --alpha       0.9          distillation weight used during training
    --batch-size  32           GradCAM batch size (reduce if OOM)
    --n-samples   2000         max test images per GradCAM run (speed vs. precision)
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
from src.models.student import VARIANT_REGISTRY, create_student
from src.evaluation.gradcam import GradCAM, get_gradcam_layer_teacher, get_gradcam_layer_student
from src.evaluation.saliency_metrics import compute_batch_saliency_metrics
from src.evaluation.metrics import compute_ece
from src.training.trainer import evaluate
from src.visualization.saliency_plots import (
    plot_compression_curve,
    plot_temperature_effect,
    plot_per_class_gap_heatmap,
)

SEED = 42


def _load_student_state_dict(model: nn.Module, ckpt_path) -> None:
    """Load checkpoint, stripping legacy 'gradcam_target.*' alias keys if present.

    Old checkpoints saved with ``self.gradcam_target = self.block_N`` registered
    a duplicate submodule, adding redundant keys to the state_dict.  Strip them
    before loading so both old and new checkpoints are compatible.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if not k.startswith("gradcam_target.")}
    model.load_state_dict(sd)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4: Sensitivity analysis")
    p.add_argument("--alpha",     type=float, default=0.9)
    p.add_argument("--batch-size",type=int,   default=32)
    p.add_argument("--n-samples", type=int,   default=0,
                   help="Max test images per GradCAM run (0 = full 10k, default)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def resolve_checkpoint(results_dir: Path, variant: str, T: float, alpha: float) -> Path:
    """Return the best-checkpoint path for a given (variant, T, alpha) combo.

    The medium T=4 checkpoint was saved under the Phase 3 naming convention
    ('distilled_student_T4.0_a{alpha}'); all Phase 4 runs use the new
    'student_{variant}_T{T}_a{alpha}' convention.
    """
    if variant == "medium" and T == 4.0:
        # Reuse the Phase 2/3 checkpoint
        return results_dir / "checkpoints" / f"distilled_student_T{T}_a{alpha}_best.pth"
    return results_dir / "checkpoints" / f"student_{variant}_T{T}_a{alpha}_best.pth"


# ---------------------------------------------------------------------------
# GradCAM evaluation for one student checkpoint
# ---------------------------------------------------------------------------

def eval_one_variant(
    teacher: nn.Module,
    student: nn.Module,
    loader,
    device: torch.device,
    n_samples: int,
    label: str,
) -> dict:
    """Run GradCAM + saliency metrics for one student against the teacher.

    Returns aggregated metrics (overall + per-class).
    """
    cam_t = GradCAM(teacher, get_gradcam_layer_teacher(teacher))
    cam_s = GradCAM(student, get_gradcam_layer_student(student))

    teacher.to(device).eval()
    student.to(device).eval()

    per_sample: list[dict] = []
    total = 0

    for inputs, targets in tqdm(loader, desc=f"  {label}", leave=False):
        if n_samples > 0 and total >= n_samples:
            break
        n = min(inputs.size(0), n_samples - total) if n_samples > 0 else inputs.size(0)
        inputs  = inputs[:n].to(device)
        targets = targets[:n]

        cams_t, logits_t = cam_t.generate(inputs)
        teacher.zero_grad()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        cams_s, logits_s = cam_s.generate(inputs)
        student.zero_grad()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        preds_t = logits_t.argmax(1)
        preds_s = logits_s.argmax(1)
        sal = compute_batch_saliency_metrics(cams_t.numpy(), cams_s.numpy())

        for i in range(n):
            tgt = int(targets[i])
            pt  = int(preds_t[i])
            ps  = int(preds_s[i])
            per_sample.append({
                "true_label":      tgt,
                "teacher_correct": pt == tgt,
                "student_correct": ps == tgt,
                "both_correct":    (pt == tgt) and (ps == tgt),
                "spearman":        sal[i]["spearman"],
                "iou":             sal[i]["iou"],
                "degenerate":      sal[i]["degenerate_a"] or sal[i]["degenerate_b"],
            })
        total += n

    cam_t.remove_hooks()
    cam_s.remove_hooks()

    # Aggregate: both-correct, non-degenerate subset
    valid = [r for r in per_sample if r["both_correct"] and not r["degenerate"]]
    rhos  = [r["spearman"] for r in valid]
    ious_ = [r["iou"] for r in valid]

    # Per-class breakdown
    per_class: list[dict] = []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_valid = [r for r in valid if r["true_label"] == cls_idx]
        per_class.append({
            "class_idx":    cls_idx,
            "name":         cls_name,
            "mean_spearman": float(np.mean([r["spearman"] for r in cls_valid])) if cls_valid else 0.0,
            "mean_iou":      float(np.mean([r["iou"]      for r in cls_valid])) if cls_valid else 0.0,
            "n_both_correct": len(cls_valid),
        })

    return {
        "label":           label,
        "n_evaluated":     total,
        "n_both_correct":  len(valid),
        "mean_spearman":   float(np.mean(rhos))  if rhos  else 0.0,
        "std_spearman":    float(np.std(rhos))   if rhos  else 0.0,
        "mean_iou":        float(np.mean(ious_)) if ious_ else 0.0,
        "std_iou":         float(np.std(ious_))  if ious_ else 0.0,
        "per_class":       per_class,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(SEED)
    alpha = args.alpha
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path("results")
    for sub in ("metrics", "plots"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Phase 4: Sensitivity Analysis")
    print(f"Device    : {device}")
    print(f"N samples : {args.n_samples if args.n_samples > 0 else 'full (10k)'}")
    print("=" * 65)

    # ---- Load teacher (frozen) ----
    teacher_ckpt = results_dir / "checkpoints" / "teacher_resnet50_best.pth"
    teacher = create_teacher(pretrained=False)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location="cpu", weights_only=True))
    print(f"Loaded teacher from {teacher_ckpt}\n")

    _, test_loader = get_dataloaders(batch_size=args.batch_size)
    ce = nn.CrossEntropyLoss()

    # ====================================================================
    # Define the two analysis axes
    # ====================================================================

    # Compression axis: T=4 fixed, three sizes.
    # (medium T=4 reuses the Phase 2/3 checkpoint)
    COMPRESSION_RUNS: list[tuple[str, str, float]] = [
        # (variant, display_label, T)
        ("tiny",   "Tiny (~95k)",   4.0),
        ("small",  "Small (~242k)", 4.0),
        ("medium", "Medium (~982k)", 4.0),
    ]

    # Temperature axis: medium fixed, three temperatures.
    TEMPERATURE_RUNS: list[tuple[str, str, float]] = [
        ("medium", "T=2", 2.0),
        ("medium", "T=4", 4.0),   # same checkpoint as above
        ("medium", "T=8", 8.0),
    ]

    # ====================================================================
    # 1 — Compression analysis
    # ====================================================================
    print("[1/3] Compression analysis  (T=4, alpha=0.9)")
    compression_results: list[dict] = []

    for variant, label, T in COMPRESSION_RUNS:
        ckpt = resolve_checkpoint(results_dir, variant, T, alpha)
        if not ckpt.exists():
            print(f"  SKIP  {label}: checkpoint not found ({ckpt})")
            continue

        factory = VARIANT_REGISTRY[variant]["factory"]
        student = factory()
        _load_student_state_dict(student, ckpt)
        n_params = sum(p.numel() for p in student.parameters())

        # Accuracy + ECE via standard no-grad eval
        student.to(device)
        with torch.no_grad():
            ev = evaluate(student, test_loader, ce, device)
        cal = compute_ece(ev["probs"], ev["targets"])

        # GradCAM saliency metrics
        sal = eval_one_variant(
            teacher, student, test_loader, device, args.n_samples, label
        )

        entry = {
            "variant":          variant,
            "label":            label,
            "temperature":      T,
            "n_params":         n_params,
            "compression_ratio": round(23_521_802 / n_params),  # teacher params
            "accuracy":         ev["accuracy"],
            "ece":              cal["ece"],
            "mean_spearman":    sal["mean_spearman"],
            "std_spearman":     sal["std_spearman"],
            "mean_iou":         sal["mean_iou"],
            "per_class":        sal["per_class"],
        }
        compression_results.append(entry)
        print(
            f"  {label:<18s}  params={n_params:>8,}  "
            f"acc={ev['accuracy']:.4f}  ECE={cal['ece']:.4f}  "
            f"rho={sal['mean_spearman']:.4f}  IoU={sal['mean_iou']:.4f}"
        )

    with open(results_dir / "metrics" / "phase4_compression.json", "w") as f:
        json.dump(compression_results, f, indent=2)

    if len(compression_results) >= 2:
        plot_compression_curve(
            compression_results,
            str(results_dir / "plots" / "phase4_compression_curve.png"),
        )
        print("  -> phase4_compression_curve.png saved")

    # ====================================================================
    # 2 — Temperature analysis
    # ====================================================================
    print("\n[2/3] Temperature analysis  (medium, alpha=0.9)")

    # Cache the medium T=4 result if already computed
    medium_t4_cached = next(
        (r for r in compression_results if r["variant"] == "medium"), None
    )

    temperature_results: list[dict] = []

    for variant, label, T in TEMPERATURE_RUNS:
        # Reuse cached result for medium T=4 to avoid redundant GradCAM pass
        if variant == "medium" and T == 4.0 and medium_t4_cached is not None:
            entry = {**medium_t4_cached, "label": label, "temperature": T}
            temperature_results.append(entry)
            print(f"  {label}  (reused from compression run)")
            continue

        ckpt = resolve_checkpoint(results_dir, variant, T, alpha)
        if not ckpt.exists():
            print(f"  SKIP  {label}: checkpoint not found ({ckpt})")
            continue

        factory = VARIANT_REGISTRY[variant]["factory"]
        student = factory()
        _load_student_state_dict(student, ckpt)
        n_params = sum(p.numel() for p in student.parameters())

        student.to(device)
        with torch.no_grad():
            ev = evaluate(student, test_loader, ce, device)
        cal = compute_ece(ev["probs"], ev["targets"])

        sal = eval_one_variant(
            teacher, student, test_loader, device, args.n_samples, label
        )

        entry = {
            "variant":       variant,
            "label":         label,
            "temperature":   T,
            "n_params":      n_params,
            "accuracy":      ev["accuracy"],
            "ece":           cal["ece"],
            "mean_spearman": sal["mean_spearman"],
            "std_spearman":  sal["std_spearman"],
            "mean_iou":      sal["mean_iou"],
            "per_class":     sal["per_class"],
        }
        temperature_results.append(entry)
        print(
            f"  {label:<8s}  "
            f"acc={ev['accuracy']:.4f}  ECE={cal['ece']:.4f}  "
            f"rho={sal['mean_spearman']:.4f}  IoU={sal['mean_iou']:.4f}"
        )

    # Sort by temperature for the plot
    temperature_results.sort(key=lambda r: r["temperature"])

    with open(results_dir / "metrics" / "phase4_temperature.json", "w") as f:
        json.dump(temperature_results, f, indent=2)

    if len(temperature_results) >= 2:
        plot_temperature_effect(
            temperature_results,
            str(results_dir / "plots" / "phase4_temperature_effect.png"),
        )
        print("  -> phase4_temperature_effect.png saved")

    # ====================================================================
    # 3 — Per-class reasoning gap heatmap
    # ====================================================================
    print("\n[3/3] Per-class reasoning gap heatmap")

    # Collect all unique variants (compression + temperature, deduped)
    seen_labels: set[str] = set()
    all_results: list[dict] = []
    for r in compression_results + temperature_results:
        if r["label"] not in seen_labels:
            all_results.append(r)
            seen_labels.add(r["label"])

    variant_labels = [r["label"] for r in all_results]
    n_variants = len(variant_labels)

    # Build (10, n_variants) spearman matrix
    spearman_matrix = np.zeros((10, n_variants), dtype=np.float64)
    for j, r in enumerate(all_results):
        cls_map = {c["class_idx"]: c["mean_spearman"] for c in r["per_class"]}
        for i in range(10):
            spearman_matrix[i, j] = cls_map.get(i, 0.0)

    per_class_gap = {
        "variant_labels": variant_labels,
        "class_names":    list(CLASSES),
        "spearman_matrix": spearman_matrix.tolist(),
    }
    with open(results_dir / "metrics" / "phase4_class_gap.json", "w") as f:
        json.dump(per_class_gap, f, indent=2)

    plot_per_class_gap_heatmap(
        CLASSES,
        variant_labels,
        spearman_matrix,
        str(results_dir / "plots" / "phase4_class_gap_heatmap.png"),
    )
    print("  -> phase4_class_gap_heatmap.png saved")

    # ====================================================================
    # Summary
    # ====================================================================
    summary = {
        "compression_analysis": [
            {k: v for k, v in r.items() if k != "per_class"}
            for r in compression_results
        ],
        "temperature_analysis": [
            {k: v for k, v in r.items() if k != "per_class"}
            for r in temperature_results
        ],
    }
    with open(results_dir / "metrics" / "phase4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("PHASE 4 SUMMARY")

    if compression_results:
        print("\n  Compression axis (T=4):")
        print(f"  {'Variant':<18s}  {'Params':>8s}  {'Acc':>6s}  {'ECE':>6s}  {'rho':>6s}  {'IoU':>6s}")
        print("  " + "-" * 58)
        for r in compression_results:
            print(
                f"  {r['label']:<18s}  {r['n_params']:>8,}  "
                f"{r['accuracy']:>6.4f}  {r['ece']:>6.4f}  "
                f"{r['mean_spearman']:>6.4f}  {r['mean_iou']:>6.4f}"
            )

    if temperature_results:
        print("\n  Temperature axis (medium):")
        print(f"  {'T':>4s}  {'Acc':>6s}  {'ECE':>6s}  {'rho':>6s}  {'IoU':>6s}")
        print("  " + "-" * 38)
        for r in sorted(temperature_results, key=lambda x: x["temperature"]):
            print(
                f"  T={r['temperature']:<3.0f}  "
                f"{r['accuracy']:>6.4f}  {r['ece']:>6.4f}  "
                f"{r['mean_spearman']:>6.4f}  {r['mean_iou']:>6.4f}"
            )

    print(f"\n  All outputs in {results_dir}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
