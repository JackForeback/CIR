"""ALVAE reconstruction on MNIST.

Identical to :class:`~cir.experiments.vae.VAEExperiment` except for the model and
the extra ``aux_weight * aux_loss`` term in the objective — so it inherits
everything else rather than restating it.
"""

from __future__ import annotations

from typing import Any

import torch

from cir.experiments.vae import VAEExperiment
from cir.models.alvae import ALVAE

__all__ = ["ALVAEExperiment"]


class ALVAEExperiment(VAEExperiment):
    """Train an :class:`~cir.models.alvae.ALVAE`.

    Adds the config keys ``basis``, ``num_basis``, ``orthonormalize``, and
    ``aux_weight`` on top of the VAE experiment's.
    """

    def build_model(self) -> ALVAE:
        """Construct the ALVAE described by the config."""
        return ALVAE(
            input_dim=self.cfg["input_dim"],
            latent_dim=self.cfg["latent_dim"],
            encoder_layers=self.cfg["encoder_layers"],
            decoder_layers=self.cfg["decoder_layers"],
            activation=self.cfg.get("activation", "relu"),
            basis=self.cfg.get("basis", "dct"),
            num_basis=self.cfg.get("num_basis"),
            orthonormalize=bool(self.cfg.get("orthonormalize", True)),
        )

    def compute_loss(self, batch: Any) -> torch.Tensor:
        """Reconstruction + KL + the fixed-basis residual penalty.

        Args:
            batch: An ``(images, labels)`` pair from the loader.

        Returns:
            The scalar total loss.
        """
        images, _ = batch
        x = images.view(images.size(0), -1).to(self.device)

        outputs = self.model(x, self.cfg.get("kl_reduction", "batchmean"))
        reconstruction = self.loss_function(outputs["x_hat"], x)
        return (
            reconstruction
            + float(self.cfg.get("kl_weight", 1e-3)) * outputs["kl_loss"]
            + float(self.cfg.get("aux_weight", 1e-2)) * outputs["aux_loss"]
        )
