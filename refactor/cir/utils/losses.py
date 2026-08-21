"""Fairness-regularized losses.

Each loss here is standard MSE plus a penalty on the *spread* between classes.
The intent is the same as the geometric projection in :mod:`cir.utils.geometry`
— equalize how well the model does across classes — but applied to the objective
rather than to the data.

Two notions of spread are provided:

``per_class_gap``
    The gap between the worst-fit and best-fit class, measured as per-class MSE.
``soft_accuracy_gap``
    The gap in mean softmax confidence assigned to the correct class. This
    tracks *confidence* rather than error magnitude, which is closer to the
    per-class accuracy the experiments actually plot.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = ["loss_with_per_class_gap", "loss_with_soft_accuracy_gap", "FAIRNESS_LOSSES"]


def _per_class_values(values: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Stack one scalar per class, using 0.0 for classes absent from the batch."""
    out = []
    for k in range(num_classes):
        mask = labels == k
        out.append(values[mask].mean() if mask.any() else values.new_zeros(()))
    return torch.stack(out)


def loss_with_per_class_gap(
    pred: torch.Tensor, target: torch.Tensor, lambda_fair: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE plus a penalty on the worst-minus-best per-class MSE.

    Args:
        pred: Raw model outputs, shape ``(N, C)``.
        target: One-hot labels, shape ``(N, C)``.
        lambda_fair: Weight on the fairness penalty. The experiments pass a
            decaying value so the penalty fades as the model converges.

    Returns:
        ``(total_loss, mse_loss, fairness_loss)``.
    """
    num_classes = target.size(1)
    labels = target.argmax(dim=1)

    mse_loss = F.mse_loss(pred, target)

    # Squared error per sample, averaged within each class.
    sample_errors = ((pred - target) ** 2).mean(dim=1)
    per_class_losses = _per_class_values(sample_errors, labels, num_classes)

    fairness_loss = per_class_losses.max() - per_class_losses.min()
    return mse_loss + lambda_fair * fairness_loss, mse_loss, fairness_loss


def loss_with_soft_accuracy_gap(
    pred: torch.Tensor, target: torch.Tensor, lambda_fair: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE plus a penalty on the spread in per-class softmax confidence.

    Args:
        pred: Raw model outputs, shape ``(N, C)``.
        target: One-hot labels, shape ``(N, C)``.
        lambda_fair: Weight on the fairness penalty.

    Returns:
        ``(total_loss, mse_loss, fairness_loss)``.
    """
    num_classes = target.size(1)
    labels = target.argmax(dim=1)

    mse_loss = F.mse_loss(pred, target)

    # Confidence each sample assigns to its own true class.
    probs = F.softmax(pred, dim=1)
    true_class_conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    class_confidences = _per_class_values(true_class_conf, labels, num_classes)

    fairness_loss = class_confidences.max() - class_confidences.min()
    return mse_loss + lambda_fair * fairness_loss, mse_loss, fairness_loss


#: Config-name -> callable, so ``flags.fairness_loss`` in YAML selects one.
FAIRNESS_LOSSES = {
    "per_class_gap": loss_with_per_class_gap,
    "soft_accuracy_gap": loss_with_soft_accuracy_gap,
}
