"""Teacher model: ResNet-50 adapted for CIFAR-10 (32x32 inputs).

Modifications from the standard ImageNet ResNet-50:
  1. conv1 replaced with 3x3 / stride-1 (instead of 7x7 / stride-2)
     to preserve spatial resolution at 32x32.
  2. Initial max-pool removed (identity pass-through).
  3. Final FC replaced with a 10-class head.

All layers except conv1 and fc are initialised from ImageNet-pretrained
weights (ResNet50_Weights.IMAGENET1K_V2), giving a strong starting point
for fine-tuning on CIFAR-10.
"""

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def create_teacher(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Build a CIFAR-adapted ResNet-50 teacher.

    Args:
        num_classes: Number of output classes.
        pretrained: If True, load ImageNet-pretrained weights for compatible layers.

    Returns:
        Modified ResNet-50 model.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)

    # Replace 7x7 stem conv with 3x3 for 32x32 inputs (randomly initialised)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Skip the max-pool — spatial resolution is already small
    model.maxpool = nn.Identity()

    # New classification head
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
