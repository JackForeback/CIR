"""MNIST loading.

A copy of MNIST is committed at the repository root under ``data/`` so runs work
on compute nodes without network access. ``download`` therefore defaults to
``False``: a missing dataset should fail loudly rather than silently reach for
the network on a cluster where that hangs.
"""

from __future__ import annotations

import os
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

__all__ = ["default_data_root", "mnist_dataloaders"]

# cir/data/mnist.py -> cir/data -> cir -> refactor -> repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def default_data_root() -> str:
    """Return the directory holding ``MNIST/``.

    Returns:
        ``$CIR_DATA_ROOT`` when set, otherwise the repository's ``data/``.
    """
    return os.environ.get("CIR_DATA_ROOT", os.path.join(_REPO_ROOT, "data"))


def mnist_dataloaders(
    root: str | None = None,
    batch_size: int = 64,
    download: bool = False,
    num_workers: int = 0,
    train_subset: int | None = None,
    test_subset: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build MNIST train and test loaders.

    Images are converted to tensors and scaled to ``[0, 1]``. They are
    deliberately *not* standardized: the decoders end in a sigmoid, so targets
    must stay in the sigmoid's range for the reconstruction loss to be
    achievable.

    Args:
        root: Directory containing ``MNIST/``. Defaults to
            :func:`default_data_root`.
        batch_size: Batch size for both loaders.
        download: Fetch the dataset if absent. Off by default, see module docs.
        num_workers: Dataloader worker processes.
        train_subset: If set, use only the first N training examples. Used by the
            smoke tests to keep runs to a few seconds.
        test_subset: The same for the test split.

    Returns:
        ``(train_loader, test_loader)``.
    """
    root = root or default_data_root()
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=download, transform=transform)

    if train_subset:
        train_dataset = _head(train_dataset, train_subset)
    if test_subset:
        test_dataset = _head(test_dataset, test_subset)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def _head(dataset, n: int):
    """Take the first ``n`` examples of a dataset, clamped to its length."""
    from torch.utils.data import Subset

    return Subset(dataset, range(min(n, len(dataset))))
