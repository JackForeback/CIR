"""VAE reconstruction on MNIST."""

from __future__ import annotations

from typing import Any, Tuple

import torch

from cir.data.mnist import mnist_dataloaders
from cir.experiments.base import BaseExperiment
from cir.models.vae import VAE

__all__ = ["VAEExperiment"]


class VAEExperiment(BaseExperiment):
    """Train a :class:`~cir.models.vae.VAE` to reconstruct MNIST digits.

    Config keys: ``input_dim``, ``latent_dim``, ``encoder_layers``,
    ``decoder_layers``, ``activation``, ``batch_size``, ``kl_weight``,
    ``kl_reduction``, plus everything :class:`~cir.experiments.base.BaseExperiment`
    reads.
    """

    def build_model(self) -> VAE:
        """Construct the VAE described by the config."""
        return VAE(
            input_dim=self.cfg["input_dim"],
            latent_dim=self.cfg["latent_dim"],
            encoder_layers=self.cfg["encoder_layers"],
            decoder_layers=self.cfg["decoder_layers"],
            activation=self.cfg.get("activation", "relu"),
        )

    def get_dataloaders(self) -> Tuple[Any, Any]:
        """Return MNIST train and test loaders."""
        return mnist_dataloaders(
            root=self.cfg.get("data_root"),
            batch_size=int(self.cfg.get("batch_size", 64)),
            download=bool(self.cfg.get("download", False)),
            num_workers=int(self.cfg.get("num_workers", 0)),
            train_subset=self.cfg.get("train_subset"),
            test_subset=self.cfg.get("test_subset"),
        )

    def compute_loss(self, batch: Any) -> torch.Tensor:
        """Reconstruction loss plus a weighted KL term.

        Args:
            batch: An ``(images, labels)`` pair from the loader; labels are
                unused because the task is unsupervised reconstruction.

        Returns:
            The scalar total loss.
        """
        images, _ = batch
        x = images.view(images.size(0), -1).to(self.device)

        outputs = self.model(x, self.cfg.get("kl_reduction", "batchmean"))
        reconstruction = self.loss_function(outputs["x_hat"], x)
        return reconstruction + float(self.cfg.get("kl_weight", 1e-3)) * outputs["kl_loss"]

    def on_run_end(self) -> None:
        """Report where the run's artifacts landed."""
        print(f"Run finished. Metrics written to {self.logger.path}")
