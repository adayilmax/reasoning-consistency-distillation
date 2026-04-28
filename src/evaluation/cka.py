"""Centered Kernel Alignment (CKA) for comparing layer representations.

Kornblith et al., "Similarity of Neural Network Representations Revisited"
(ICML 2019).

We use the *linear* kernel variant:

    K = X Xᵀ,   L = Y Yᵀ
    CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) · HSIC(L, L))

where HSIC uses the unbiased Gram-centering operator H:

    HSIC(K, L) = ‖H K H ⊙ H L H‖_F / (N-1)²

CKA is invariant to orthogonal transformations and isotropic scaling, so
differences in channel count between teacher and student don't matter — the
metric captures alignment of the *geometry* of representations, not their
dimensionality.

Feature extraction strategy
----------------------------
Spatial feature maps (H×W) are collapsed to a single vector per sample via
global average pooling before the Gram matrix is computed.  This focuses the
comparison on *what* information each layer encodes (channel-wise statistics)
rather than *where* exactly it is encoded — appropriate for comparing networks
with the same spatial resolution but very different channel counts.

Named layer correspondences (same spatial resolution):
    Teacher layer1  (32×32) ↔ Student block1  (32×32)
    Teacher layer2  (16×16) ↔ Student block2  (16×16)
    Teacher layer3  ( 8×8 ) ↔ Student block3  ( 8×8 )
    Teacher layer4  ( 4×4 ) ↔ Student block4  ( 4×4 )
                             ↔ Student block5  ( 4×4 )
    Teacher pre-fc  (pool)  ↔ Student pre-cls (pool)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------

def _center_gram(K: torch.Tensor) -> torch.Tensor:
    """Double-center a Gram matrix: H K H where H = I - 1/N · 11ᵀ."""
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - 1.0 / n
    return H @ K @ H


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two feature matrices.

    Args:
        X: (N, p) feature matrix from model / layer A.
        Y: (N, q) feature matrix from model / layer B.

    Returns:
        CKA ∈ [0, 1].  0 = no alignment, 1 = identical geometry.
    """
    # Use float64 for numerical stability (Gram products can be large)
    X = X.double()
    Y = Y.double()

    K = X @ X.T         # (N, N)
    L = Y @ Y.T         # (N, N)

    K_c = _center_gram(K)
    L_c = _center_gram(L)

    # HSIC = Frobenius inner product / (N-1)² — but the (N-1)² cancels in CKA
    hsic_xy = (K_c * L_c).sum()
    hsic_xx = (K_c * K_c).sum()
    hsic_yy = (L_c * L_c).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


# ---------------------------------------------------------------------------
# Feature extractor with forward hooks
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Collect intermediate activations from named layers via forward hooks.

    Usage:
        extractor = FeatureExtractor(model, {"layer4": model.layer4})
        with torch.no_grad():
            _ = model(inputs)
        feats = extractor.features  # {"layer4": Tensor(N, C)}
        extractor.remove_hooks()
    """

    def __init__(
        self,
        model: nn.Module,
        layers: dict[str, nn.Module],
        pool: bool = True,
    ) -> None:
        """
        Args:
            model: The model to extract features from.
            layers: Mapping of name → nn.Module to hook.
            pool: If True, apply global average pooling to collapse spatial
                  dims before storing.  Recommended for CKA.
        """
        self.model = model
        self.pool = pool
        self.features: dict[str, torch.Tensor] = {}
        self._hooks: list = []

        for name, layer in layers.items():
            handle = layer.register_forward_hook(self._make_hook(name))
            self._hooks.append(handle)

    def _make_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            feat = output.detach()
            if self.pool and feat.dim() == 4:
                # (N, C, H, W) → (N, C) via global average pool
                feat = feat.mean(dim=(2, 3))
            elif feat.dim() > 2:
                # Fallback: flatten
                feat = feat.view(feat.size(0), -1)
            self.features[name] = feat.cpu()
        return hook

    def clear(self) -> None:
        """Clear stored features (call between batches)."""
        self.features = {}

    def remove_hooks(self) -> None:
        """Deregister all hooks."""
        for h in self._hooks:
            h.remove()


# ---------------------------------------------------------------------------
# Layer definitions for each architecture
# ---------------------------------------------------------------------------

def get_teacher_cka_layers(model: nn.Module) -> dict[str, nn.Module]:
    """Named CKA layers for the CIFAR-adapted ResNet-50."""
    return OrderedDict([
        ("layer1", model.layer1),
        ("layer2", model.layer2),
        ("layer3", model.layer3),
        ("layer4", model.layer4),
        # avgpool collapses 4×4 → 1×1; hook placed on avgpool output
        ("pre_fc", model.avgpool),
    ])


def get_student_cka_layers(model: nn.Module) -> dict[str, nn.Module]:
    """Named CKA layers for StudentCNN."""
    return OrderedDict([
        ("block1", model.block1),
        ("block2", model.block2),
        ("block3", model.block3),
        ("block4", model.block4),
        ("block5", model.block5),
        ("pre_cls", model.pool),
    ])


# ---------------------------------------------------------------------------
# Batch-accumulating CKA computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    model: nn.Module,
    layers: dict[str, nn.Module],
    loader,
    device: torch.device,
    n_samples: int = 1000,
) -> dict[str, torch.Tensor]:
    """Run forward passes and collect pooled features for up to n_samples images.

    Returns a dict mapping layer_name → (N, C) feature tensor.
    """
    extractor = FeatureExtractor(model, layers, pool=True)
    model.eval()
    model.to(device)

    collected: dict[str, list[torch.Tensor]] = {k: [] for k in layers}
    total = 0

    for inputs, _ in loader:
        if total >= n_samples:
            break
        inputs = inputs.to(device)
        n = min(inputs.size(0), n_samples - total)
        inputs = inputs[:n]

        _ = model(inputs)  # populates extractor.features

        for k in layers:
            collected[k].append(extractor.features[k])
        extractor.clear()
        total += n

    extractor.remove_hooks()

    return {k: torch.cat(v, dim=0) for k, v in collected.items()}


def compute_cka_matrix(
    teacher_feats: dict[str, torch.Tensor],
    student_feats: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    """Compute pairwise CKA between all teacher × student layer pairs.

    Returns a nested dict: result[teacher_layer][student_layer] = cka_value.
    """
    results: dict[str, dict[str, float]] = {}
    for t_name, T in teacher_feats.items():
        results[t_name] = {}
        for s_name, S in student_feats.items():
            assert T.shape[0] == S.shape[0], "Sample count mismatch"
            results[t_name][s_name] = linear_cka(T, S)
    return results
