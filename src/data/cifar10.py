"""CIFAR-10 data loading and preprocessing.

Standard augmentation pipeline used across all experiments:
- Training: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
- Test: Normalize only

Both teacher and student operate on native 32x32 resolution to avoid
confounding the saliency-map comparison in Phase 3.
"""

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return the standard CIFAR-10 transform pipeline."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and test data loaders.

    Args:
        data_dir: Root directory for dataset download/cache.
        batch_size: Mini-batch size.
        num_workers: Dataloader workers. Defaults to 0 on Windows, 4 otherwise.

    Returns:
        (train_loader, test_loader) tuple.
    """
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 4

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=get_transforms(train=True),
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
