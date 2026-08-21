"""Alternating-decoder VAE reconstruction on MNIST.

Runs any of the three variants in :mod:`cir.models.alternating`, selected by the
config's ``variant`` key. Everything except the model and the extra loss term is
inherited from :class:`~cir.experiments.vae.VAEExperiment`.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from cir.experiments.vae import VAEExperiment
from cir.models.alternating import VARIANTS, AlternatingVAE

__all__ = ["AlternatingVAEExperiment"]


class AlternatingVAEExperiment(VAEExperiment):
    """Train an :class:`~cir.models.alternating.AlternatingVAE`.

    Adds these config keys on top of the VAE experiment's:

    ==================  =========================================================
    ``variant``         ``lavae``, ``folvae``, or ``added_loss``.
    ``linear_layers``   Hidden widths of the purely linear decode path.
    ``alternate_every`` Use the linear path on every *n*-th training step.
    ``aux_weight``      Weight on the added-loss term (``added_loss`` only).
    ==================  =========================================================

    The alternation schedule advances on training steps only. Validation always
    decodes through the learned nonlinear decoder, so the reported validation
    loss compares like with like from epoch to epoch instead of jumping between
    two paths.

    Attributes:
        global_step: Training steps taken so far; drives the schedule.
    """

    def __init__(self, cfg: Dict[str, Any], logger=None):
        super().__init__(cfg, logger=logger)

        name = str(cfg.get("variant", "lavae")).lower()
        if name not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {name!r}")
        self.variant = name
        self.global_step = 0

    def build_model(self) -> AlternatingVAE:
        """Construct the variant named by ``cfg["variant"]``."""
        kwargs = dict(
            input_dim=self.cfg["input_dim"],
            latent_dim=self.cfg["latent_dim"],
            encoder_layers=self.cfg["encoder_layers"],
            decoder_layers=self.cfg["decoder_layers"],
            activation=self.cfg.get("activation", "relu"),
            linear_layers=self.cfg.get("linear_layers", [128]),
            alternate_every=int(self.cfg.get("alternate_every", 2)),
        )
        return VARIANTS[self.variant](**kwargs)

    def compute_loss(self, batch: Any) -> torch.Tensor:
        """Reconstruction + KL + the weighted added-loss term.

        Advances :attr:`global_step` on training batches only, so evaluation
        neither shifts the schedule nor decodes through the linear path.

        Args:
            batch: An ``(images, labels)`` pair from the loader.

        Returns:
            The scalar total loss.
        """
        images, _ = batch
        x = images.view(images.size(0), -1).to(self.device)

        # Step 1 is a nonlinear step for every schedule, which is what we want
        # for evaluation.
        step = self.global_step if self.model.training else 1
        outputs = self.model(x, self.cfg.get("kl_reduction", "batchmean"), step=step)
        if self.model.training:
            self.global_step += 1

        reconstruction = self.loss_function(outputs["x_hat"], x)
        return (
            reconstruction
            + float(self.cfg.get("kl_weight", 1e-3)) * outputs["kl_loss"]
            + float(self.cfg.get("aux_weight", 1e-2)) * outputs["aux_loss"]
        )

    def on_run_end(self) -> None:
        """Report the variant that ran and where its metrics landed."""
        print(f"Variant {self.variant!r} finished. Metrics written to {self.logger.path}")
