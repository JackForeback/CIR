"""ALVAE — the Added-Loss Variational Autoencoder.

A :class:`~cir.models.vae.VAE` carrying one extra term in its objective. The
pre-refactor sketches (``FOLVAE``, ``LAVAE``, ``ALVAE`` in ``original/models.py``)
all circled the same idea: hold part of the decoder *fixed* — an analytically
known basis rather than learned weights — and penalize the model for the part of
its reconstruction that the fixed basis cannot express.

Concretely, the reconstruction is projected onto a fixed orthonormal basis
(:class:`~cir.models.basis.BasisLinear`) and back. Anything lost in that round
trip is energy outside the basis's span, and the auxiliary loss is the size of
that residual. Minimizing it pushes reconstructions toward the smooth, low-order
subspace the basis describes, which is what makes them solvable by
:func:`cir.utils.solvers.iterative_chebyshev_ls`.

.. note::
   The *mechanism* — a fixed basis, a residual penalty, a config-driven weight —
   is what the original code specified. The precise regularizer was only ever
   sketched in comments. Override :meth:`ALVAE.auxiliary_loss` to substitute a
   different definition; nothing else in the pipeline needs to change.
"""

from __future__ import annotations

from typing import Sequence

import torch

from cir.models.basis import BasisLinear
from cir.models.vae import VAE

__all__ = ["ALVAE"]


class ALVAE(VAE):
    """VAE plus a fixed-basis residual penalty.

    Args:
        input_dim: Dimensionality of the data.
        latent_dim: Size of the latent space.
        encoder_layers: Encoder hidden widths.
        decoder_layers: Decoder hidden widths.
        activation: Activation name, see :func:`cir.models.vae.get_activation`.
        basis: Basis name or callable for the fixed projection; see
            :class:`~cir.models.basis.BasisLinear`.
        num_basis: Number of basis functions to keep. Fewer means a stronger
            constraint. Defaults to a quarter of ``input_dim``, at least one.
        orthonormalize: Orthonormalize the basis rows. Required for the
            round-trip projection to be a true orthogonal projection, so this
            defaults on.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_layers: Sequence[int],
        decoder_layers: Sequence[int],
        activation: str = "relu",
        basis: str = "dct",
        num_basis: int | None = None,
        orthonormalize: bool = True,
    ):
        super().__init__(input_dim, latent_dim, encoder_layers, decoder_layers, activation)
        self.num_basis = max(1, input_dim // 4) if num_basis is None else int(num_basis)
        self.projection = BasisLinear(
            input_dim=input_dim,
            output_dim=self.num_basis,
            basis=basis,
            orthonormalize=orthonormalize,
        )

    def auxiliary_loss(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Mean squared energy of the reconstruction outside the basis span.

        With orthonormal rows ``W``, ``x_hat @ Wᵀ @ W`` is the orthogonal
        projection of ``x_hat`` onto the span, so the residual measures exactly
        the component the fixed basis cannot represent.

        Args:
            x_hat: Reconstructions, shape ``(batch, input_dim)``.

        Returns:
            A scalar loss, zero when ``x_hat`` lies entirely in the span.
        """
        coefficients = self.projection(x_hat)                    # (batch, num_basis)
        reprojected = coefficients.matmul(self.projection.weight)  # (batch, input_dim)
        return (x_hat - reprojected).pow(2).mean()

    def forward(self, x: torch.Tensor, kl_reduction: str = "batchmean") -> dict:
        """Run the VAE pass and attach the auxiliary loss.

        Args:
            x: Shape ``(batch, input_dim)``.
            kl_reduction: Passed to :meth:`~cir.models.vae.VAE.get_kl_loss`.

        Returns:
            The base VAE dict plus ``aux_loss``.
        """
        outputs = super().forward(x, kl_reduction)
        outputs["aux_loss"] = self.auxiliary_loss(outputs["x_hat"])
        return outputs
