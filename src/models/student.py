"""Student model variants for Phase 1 / Phase 4 sensitivity analysis.

Three architectures at increasing capacity:

  Tiny   (~95 k params, ~247x compression vs teacher)
    3 conv blocks: 3→32→64→128, each with MaxPool → 4x4 spatial at last block
    GradCAM target: block3

  Small  (~242 k params, ~97x compression)
    4 conv blocks: 3→32→64→128→128, pool after blocks 2-4
    GradCAM target: block4

  Medium (~982 k params, ~24x compression)  ← default, used in Phases 1-3
    5 conv blocks: 3→32→64→128→256→256
    GradCAM target: block5

All variants:
  - Use 3×3 convolutions with BN+ReLU, no pretraining.
  - Store each block as a named nn.Module attribute (needed for CKA).
  - Expose a ``gradcam_target`` attribute pointing to their last conv block
    so that generic code can call GradCAM without hardcoding block names.
  - Expose a ``variant`` string tag used to name checkpoints and plots.
"""

import torch
import torch.nn as nn


def _conv_block(in_c: int, out_c: int, pool: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Tiny  (~95 k params)
# ---------------------------------------------------------------------------

class TinyStudentCNN(nn.Module):
    """3-block student (~95 k params, ~247x compressed vs ResNet-50 teacher)."""

    variant: str = "tiny"

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = _conv_block(3,  32, pool=True)   # 32x32 → 16x16
        self.block2 = _conv_block(32, 64, pool=True)   # 16x16 → 8x8
        self.block3 = _conv_block(64, 128, pool=True)  # 8x8   → 4x4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

        # GradCAM target — plain Python ref, NOT a registered submodule
        object.__setattr__(self, "gradcam_target", self.block3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Small  (~242 k params)
# ---------------------------------------------------------------------------

class SmallStudentCNN(nn.Module):
    """4-block student (~242 k params, ~97x compressed vs ResNet-50 teacher)."""

    variant: str = "small"

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = _conv_block(3,   32, pool=False)  # 32x32
        self.block2 = _conv_block(32,  64, pool=True)   # → 16x16
        self.block3 = _conv_block(64,  128, pool=True)  # → 8x8
        self.block4 = _conv_block(128, 128, pool=True)  # → 4x4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

        object.__setattr__(self, "gradcam_target", self.block4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Medium  (~982 k params)  — original student from Phases 1-3
# ---------------------------------------------------------------------------

class StudentCNN(nn.Module):
    """5-block student (~982 k params, ~24x compressed vs ResNet-50 teacher)."""

    variant: str = "medium"

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = _conv_block(3,   32,  pool=False)  # 32x32
        self.block2 = _conv_block(32,  64,  pool=True)   # → 16x16
        self.block3 = _conv_block(64,  128, pool=True)   # → 8x8
        self.block4 = _conv_block(128, 256, pool=True)   # → 4x4
        self.block5 = _conv_block(256, 256, pool=False)  # 4x4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

        object.__setattr__(self, "gradcam_target", self.block5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_student(num_classes: int = 10) -> StudentCNN:
    """Create the medium (5-block) student — default for Phases 1-3."""
    return StudentCNN(num_classes=num_classes)


def create_student_tiny(num_classes: int = 10) -> TinyStudentCNN:
    """Create the tiny (3-block) student for Phase 4 sensitivity analysis."""
    return TinyStudentCNN(num_classes=num_classes)


def create_student_small(num_classes: int = 10) -> SmallStudentCNN:
    """Create the small (4-block) student for Phase 4 sensitivity analysis."""
    return SmallStudentCNN(num_classes=num_classes)


# Variant registry — maps string tag to factory function and param count (approx)
VARIANT_REGISTRY: dict[str, dict] = {
    "tiny":   {"factory": create_student_tiny,  "approx_params": 95_000,  "compression_vs_teacher": 247},
    "small":  {"factory": create_student_small, "approx_params": 242_000, "compression_vs_teacher": 97},
    "medium": {"factory": create_student,       "approx_params": 982_000, "compression_vs_teacher": 24},
}
