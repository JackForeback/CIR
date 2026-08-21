"""A configurable MLP variational autoencoder.

Layer widths, depth, and activation all come from the config, so the same class
covers the small MNIST autoencoder the experiments use and anything wider.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

__all__ = ["get_activation", "build_mlp", "Encoder", "Decoder", "VAE"]

ACTIVATIONS = {
    "relu": lambda: nn.ReLU(),
    "leakyrelu": lambda: nn.LeakyReLU(0.2),
    "gelu": lambda: nn.GELU(),
    "tanh": lambda: nn.Tanh(),
    "sigmoid": lambda: nn.Sigmoid(),
}


def get_activation(name: str) -> nn.Module:
    """Look up an activation module by name.

    Args:
        name: One of ``relu``, ``leakyrelu``, ``gelu``, ``tanh``, ``sigmoid``
            (case-insensitive).

    Returns:
        A fresh activation module.

    Raises:
        ValueError: If ``name`` is not a supported activation.
    """
    key = name.lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation {name!r}; expected one of {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[key]()


def build_mlp(input_dim: int, hidden_sizes: Sequence[int], activation: str) -> Tuple[nn.Sequential, int]:
    """Stack ``Linear -> activation`` blocks.

    A fresh activation module is created per layer so no module is shared
    between two points in the graph.

    Args:
        input_dim: Width entering the stack.
        hidden_sizes: Width of each hidden layer, in order. May be empty.
        activation: Activation name, see :func:`get_activation`.

    Returns:
        ``(stack, output_width)`` — the width is ``input_dim`` for an empty stack.
    """
    layers: list = []
    prev_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden_size))
        layers.append(get_activation(activation))
        prev_dim = hidden_size
    return nn.Sequential(*layers), prev_dim


class Encoder(nn.Module):
    """Maps an input to the parameters of a diagonal Gaussian posterior.

    Args:
        input_dim: Dimensionality of the input.
        hidden_layer_sizes: Hidden widths, in order.
        latent_dim: Size of the latent space.
        activation: Activation name, see :func:`get_activation`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: Sequence[int],
        latent_dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.hidden_layers, prev_dim = build_mlp(input_dim, hidden_layer_sizes, activation)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch.

        Args:
            x: Shape ``(batch, input_dim)``.

        Returns:
            ``(mu, log_var)``, each of shape ``(batch, latent_dim)``.
        """
        hidden = self.hidden_layers(x)
        return self.fc_mu(hidden), self.fc_logvar(hidden)


class Decoder(nn.Module):
    """Maps a latent code back to a reconstruction in ``[0, 1]``.

    Args:
        latent_dim: Size of the latent space.
        hidden_layer_sizes: Hidden widths, in order.
        output_dim: Dimensionality of the reconstruction.
        activation: Activation name, see :func:`get_activation`.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_layer_sizes: Sequence[int],
        output_dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.hidden_layers, prev_dim = build_mlp(latent_dim, hidden_layer_sizes, activation)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latent codes.

        Args:
            z: Shape ``(batch, latent_dim)``.

        Returns:
            Reconstruction of shape ``(batch, output_dim)``, squashed to ``[0, 1]``.
        """
        return torch.sigmoid(self.output_layer(self.hidden_layers(z)))


class VAE(nn.Module):
    """Encoder, decoder, reparameterization, and the KL term.

    Args:
        input_dim: Dimensionality of the data.
        latent_dim: Size of the latent space.
        encoder_layers: Encoder hidden widths.
        decoder_layers: Decoder hidden widths.
        activation: Activation name, see :func:`get_activation`.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_layers: Sequence[int],
        decoder_layers: Sequence[int],
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, encoder_layers, latent_dim, activation)
        self.decoder = Decoder(latent_dim, decoder_layers, input_dim, activation)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample the posterior through the reparameterization trick.

        Args:
            mu: Posterior means, shape ``(batch, latent_dim)``.
            log_var: Posterior log-variances, same shape.

        Returns:
            A differentiable sample, same shape.
        """
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def get_kl_loss(
        self, mu: torch.Tensor, log_var: torch.Tensor, reduction: str = "batchmean"
    ) -> torch.Tensor:
        """KL divergence from the posterior to a standard normal prior.

        Args:
            mu: Posterior means, shape ``(batch, latent_dim)``.
            log_var: Posterior log-variances, same shape.
            reduction: ``"sum"``, ``"mean"``, ``"batchmean"`` (sum over latent
                dims, mean over the batch — the correct pairing for an MSE
                reconstruction term), or ``"none"``.

        Returns:
            A scalar, or the unreduced tensor when ``reduction="none"``.

        Raises:
            ValueError: If ``reduction`` is unrecognized.
        """
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        if reduction == "sum":
            return kl.sum()
        if reduction == "mean":
            return kl.mean()
        if reduction == "batchmean":
            return kl.sum() / mu.size(0)
        if reduction == "none":
            return kl
        raise ValueError(f"Unsupported kl reduction {reduction!r}")

    def forward(self, x: torch.Tensor, kl_reduction: str = "batchmean") -> dict:
        """Run a full encode/sample/decode pass.

        Args:
            x: Shape ``(batch, input_dim)``.
            kl_reduction: Passed to :meth:`get_kl_loss`.

        Returns:
            A dict with ``x_hat``, ``kl_loss``, ``mu``, ``log_var``, and ``z``.
            Subclasses add their own keys (see
            :class:`cir.models.alvae.ALVAE`), which is why this is a dict rather
            than a fixed-width tuple.
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return {
            "x_hat": x_hat,
            "kl_loss": self.get_kl_loss(mu, log_var, kl_reduction),
            "mu": mu,
            "log_var": log_var,
            "z": z,
        }
