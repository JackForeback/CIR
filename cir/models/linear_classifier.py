"""The linear classifier under study.

A single `nn.Linear` producing one score per class, trained against one-hot
targets with MSE. Deliberately minimal: the research question is about the
*geometry of the data*, so the model is kept as simple as possible to avoid
confounding the result.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["LinearClassifier"]


class LinearClassifier(nn.Module):
    """One linear layer, zero-initialized bias.

    The bias starts at zero so every class's decision boundary passes through
    the origin at step 0. Any early asymmetry between classes then comes from
    the data geometry rather than from the initialization, which is the effect
    the experiments are trying to isolate.

    Args:
        input_dim: Number of input features.
        num_classes: Number of output scores.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch.

        Args:
            x: Shape ``(batch, input_dim)``.

        Returns:
            Raw scores of shape ``(batch, num_classes)``.
        """
        return self.linear(x)
