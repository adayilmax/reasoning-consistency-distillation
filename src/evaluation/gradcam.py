"""Gradient-weighted Class Activation Mapping (GradCAM).

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization" (ICCV 2017).

Design:
  - One GradCAM instance per model.  Hooks are registered at construction
    and must be removed via .remove_hooks() when done to prevent memory leaks.
  - Batch-aware: a single .generate() call handles a full mini-batch.
    The gradient trick is to construct a one-hot matrix that selects each
    sample's target class logit; because the logits are independent across
    samples, the batch backward pass yields per-sample gradients correctly.
  - Models must NOT be inside a torch.no_grad() context during .generate()
    (we need the computation graph for the backward pass).
  - Models should be in .eval() mode so BatchNorm uses running statistics.

Typical usage:
    cam_teacher = GradCAM(teacher, teacher.layer4)
    cam_student  = GradCAM(student, student.block5)
    ...
    cams_t, logits_t = cam_teacher.generate(inputs)
    cams_s, logits_s = cam_student.generate(inputs)
    ...
    cam_teacher.remove_hooks()
    cam_student.remove_hooks()
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Batch GradCAM for an arbitrary layer of an arbitrary model."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
        Args:
            model: The model to explain.
            target_layer: The layer whose activations and gradients are used.
                - For ResNet-50 (CIFAR-adapted): model.layer4
                - For StudentCNN:                model.block5
        """
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------
    # Private hook callbacks
    # ------------------------------------------------------------------

    def _save_activation(
        self,
        module: nn.Module,
        input: tuple,
        output: torch.Tensor,
    ) -> None:
        # Detach so the stored tensor doesn't hold the computation graph
        self._activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        # grad_output[0]: gradient of the loss w.r.t. this layer's output
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        inputs: torch.Tensor,
        class_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GradCAM heatmaps for a batch.

        Args:
            inputs: (N, C, H, W) — normalised image tensor on the model's device.
                    Must NOT be inside a torch.no_grad() context.
            class_indices: (N,) — class to explain for each sample.
                           Defaults to the model's argmax prediction.

        Returns:
            cams:   (N, H, W)    — per-image heatmaps normalised to [0, 1].
            logits: (N, num_cls) — raw model outputs (before softmax).
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(inputs)  # forward pass → triggers _save_activation

        if class_indices is None:
            class_indices = logits.argmax(dim=1)

        # Build a one-hot gradient mask: selects one logit per sample.
        # Backpropagating with this mask is equivalent to summing
        #   d logit[i, class_indices[i]] / d activations
        # for all i simultaneously (they are independent across samples).
        N = inputs.shape[0]
        one_hot = torch.zeros_like(logits)
        one_hot[torch.arange(N), class_indices] = 1.0
        logits.backward(gradient=one_hot)  # backward → triggers _save_gradient

        # GradCAM weights: global-average pooling of gradients over spatial dims
        # Shape: (N, C, h, w) → (N, C, 1, 1)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps; ReLU keeps positive influences
        # Shape: (N, C, h, w) → sum → (N, 1, h, w)
        cam = F.relu(
            (weights * self._activations).sum(dim=1, keepdim=True)
        )

        # Upsample to original input resolution (bilinear)
        cam = F.interpolate(
            cam, size=inputs.shape[-2:],
            mode="bilinear", align_corners=False,
        )  # (N, 1, H, W)

        # Per-image min-max normalization → [0, 1]
        cam = cam.squeeze(1)           # (N, H, W)
        flat = cam.view(N, -1)
        cam_min = flat.min(dim=1)[0].view(N, 1, 1)
        cam_max = flat.max(dim=1)[0].view(N, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Move results to CPU before freeing GPU tensors
        cam_cpu = cam.detach().cpu()
        logits_cpu = logits.detach().cpu()

        # Explicitly release stored GPU tensors so they don't accumulate
        # across batches — critical for avoiding OOM on long evaluation loops.
        self._activations = None
        self._gradients = None

        return cam_cpu, logits_cpu

    def remove_hooks(self) -> None:
        """Deregister hooks. Always call this when done."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ---------------------------------------------------------------------------
# Layer accessors for our two architectures
# ---------------------------------------------------------------------------

def get_gradcam_layer_teacher(model: nn.Module) -> nn.Module:
    """Return the GradCAM target layer for the CIFAR-adapted ResNet-50.

    We use layer4 (output: N×2048×4×4 for 32x32 CIFAR inputs).
    """
    return model.layer4


def get_gradcam_layer_student(model: nn.Module) -> nn.Module:
    """Return the GradCAM target layer for any StudentCNN variant.

    Each variant exposes a ``gradcam_target`` attribute pointing to its last
    conv block (block3 for tiny, block4 for small, block5 for medium).
    Falls back to ``block5`` for backwards-compatibility if the attribute
    is absent (e.g. a manually instantiated StudentCNN from Phase 1).
    """
    # getattr's third argument is evaluated eagerly, so we use a conditional
    # to avoid AttributeError on tiny/small variants that have no block5.
    return model.gradcam_target if hasattr(model, "gradcam_target") else model.block5
