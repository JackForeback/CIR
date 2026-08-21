"""Alternating-decoder VAEs.

These carry forward the three exploratory variants the pre-refactor code
sketched (``FOLVAE``, ``LAVAE``, ``ALVAE`` in the original ``models.py``). All
three circle one idea: give the VAE a second decode path that is *purely
linear* — no activations anywhere — so the reconstruction it produces lives in a
known low-dimensional subspace, something
:func:`cir.utils.solvers.iterative_chebyshev_ls` could solve for directly
instead of learning.

Two ways of using that second path are provided, which is exactly the split the
original made:

*Alternation* (:class:`LAVAE`, :class:`FOLVAE`)
    Decode through the learned nonlinear decoder on some steps and through the
    linear path on the others, so both are trained against the same objective.
*Added loss* (:class:`AddedLossVAE`)
    Always decode through the nonlinear decoder, and penalize how far its output
    sits from what the linear path would have produced. Driving that gap down
    pulls the learned decoder toward a linearly-representable solution.

.. note::
   The original called the added-loss variant ``ALVAE``. That name now belongs
   to :class:`cir.models.alvae.ALVAE`, a *later* and different mechanism — a
   fixed orthonormal basis with a residual penalty rather than a second learned
   decoder. Both are kept because they answer different questions; the class
   here is named :class:`AddedLossVAE` so the two never get confused.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from cir.models.vae import VAE

__all__ = ["build_linear_path", "AlternatingVAE", "LAVAE", "FOLVAE", "AddedLossVAE", "VARIANTS"]


def build_linear_path(
    latent_dim: int, hidden_sizes: Sequence[int], output_dim: int
) -> nn.Sequential:
    """Stack ``Linear`` layers with no activation between them.

    A composition of affine maps is itself one affine map; stacking them merely
    changes how that map is *parameterized*, which is how the original wrote it
    (``latent -> 128 -> input``). The point is that the path stays linear, so its
    output is confined to a subspace a least-squares solve can reach.

    Args:
        latent_dim: Width entering the path.
        hidden_sizes: Intermediate widths, in order. May be empty for a single
            ``latent_dim -> output_dim`` map.
        output_dim: Width leaving the path.

    Returns:
        The stacked path.
    """
    layers: list = []
    prev_dim = latent_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden_size))
        prev_dim = hidden_size
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class AlternatingVAE(VAE):
    """A :class:`~cir.models.vae.VAE` with a second, purely linear decode path.

    On steps where ``step % alternate_every == 0`` the reconstruction comes from
    the linear path; otherwise it comes from the learned nonlinear decoder. The
    default of 2 reproduces the original's "linear on even steps" schedule.

    Unlike :class:`~cir.models.vae.Decoder`, the linear path has **no** output
    sigmoid: squashing it would make it nonlinear and defeat the purpose. Its
    reconstructions are therefore unbounded, which the MSE objective handles
    fine but a Bernoulli likelihood would not.

    Args:
        input_dim: Dimensionality of the data.
        latent_dim: Size of the latent space.
        encoder_layers: Encoder hidden widths.
        decoder_layers: Nonlinear decoder hidden widths.
        activation: Activation name, see :func:`cir.models.vae.get_activation`.
        linear_layers: Hidden widths of the linear path.
        alternate_every: Use the linear path on every *n*-th step.
        freeze_output: Freeze the final layer of the linear path, so only the
            layers feeding it adapt. See :class:`FOLVAE`.

    Raises:
        ValueError: If ``alternate_every`` is not positive.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_layers: Sequence[int],
        decoder_layers: Sequence[int],
        activation: str = "relu",
        linear_layers: Sequence[int] = (128,),
        alternate_every: int = 2,
        freeze_output: bool = False,
    ):
        super().__init__(input_dim, latent_dim, encoder_layers, decoder_layers, activation)
        if alternate_every < 1:
            raise ValueError(f"alternate_every must be >= 1, got {alternate_every}")

        self.alternate_every = int(alternate_every)
        self.linear_decoder = build_linear_path(latent_dim, list(linear_layers), input_dim)

        if freeze_output:
            for parameter in self.linear_decoder[-1].parameters():
                parameter.requires_grad_(False)

    def linear_decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode through the linear path only.

        Args:
            z: Latent codes, shape ``(batch, latent_dim)``.

        Returns:
            Reconstruction of shape ``(batch, input_dim)``, unbounded.
        """
        return self.linear_decoder(z)

    def uses_linear_path(self, step: int) -> bool:
        """Whether ``step`` decodes through the linear path.

        Args:
            step: Training step index.

        Returns:
            ``True`` on the linear steps of the schedule.
        """
        return step % self.alternate_every == 0

    def decode(self, z: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produce the reconstruction for one step, plus any auxiliary loss.

        Args:
            z: Latent codes, shape ``(batch, latent_dim)``.
            step: Training step index, which selects the path.

        Returns:
            ``(x_hat, aux_loss)``. The base schedule has no auxiliary term, so
            ``aux_loss`` is a zero scalar; :class:`AddedLossVAE` overrides it.
        """
        x_hat = self.linear_decode(z) if self.uses_linear_path(step) else self.decoder(z)
        return x_hat, z.new_zeros(())

    def forward(self, x: torch.Tensor, kl_reduction: str = "batchmean", step: int = 1) -> dict:
        """Encode, sample, and decode through whichever path ``step`` selects.

        Args:
            x: Shape ``(batch, input_dim)``.
            kl_reduction: Passed to :meth:`~cir.models.vae.VAE.get_kl_loss`.
            step: Training step index. The default of 1 is a nonlinear step, so
                a caller that does not track steps gets the ordinary VAE path.

        Returns:
            The usual VAE dict, plus ``aux_loss`` and ``used_linear_path``.
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat, aux_loss = self.decode(z, step)
        return {
            "x_hat": x_hat,
            "kl_loss": self.get_kl_loss(mu, log_var, kl_reduction),
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "aux_loss": aux_loss,
            "used_linear_path": self.uses_linear_path(step),
        }


class LAVAE(AlternatingVAE):
    """Linear Alternating VAE: both paths learned, alternating by schedule.

    The plain alternating model, and the baseline the other two variants are
    compared against. It adds nothing to :class:`AlternatingVAE` — the name is
    kept because the original's results are recorded under it.
    """


class FOLVAE(AlternatingVAE):
    """Fixed Output Layer VAE: alternating, with the linear path's last layer frozen.

    Freezing the output map is the cheapest stand-in for the step the original
    never finished — *solving* for that map with
    :func:`cir.utils.solvers.iterative_chebyshev_ls` rather than learning it. On
    linear steps only the layers feeding the frozen output can adapt.

    Args: as :class:`AlternatingVAE`, except ``freeze_output``, which is forced on.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_layers: Sequence[int],
        decoder_layers: Sequence[int],
        activation: str = "relu",
        linear_layers: Sequence[int] = (128,),
        alternate_every: int = 2,
    ):
        super().__init__(
            input_dim,
            latent_dim,
            encoder_layers,
            decoder_layers,
            activation,
            linear_layers=linear_layers,
            alternate_every=alternate_every,
            freeze_output=True,
        )


class AddedLossVAE(AlternatingVAE):
    """Always decode nonlinearly, and penalize disagreement with the linear path.

    This is the original ``ALVAE``: rather than switching between the paths, it
    keeps the learned decoder and reports how far its reconstruction sits from
    what the linear path would produce. The experiment weights that term with
    ``aux_weight``.

    Not to be confused with :class:`cir.models.alvae.ALVAE`, which penalizes the
    residual against a *fixed* basis instead of a second learned decoder.
    """

    def decode(self, z: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode nonlinearly and measure the gap to the linear path.

        Args:
            z: Latent codes, shape ``(batch, latent_dim)``.
            step: Ignored — this variant never switches paths. Accepted so the
                schedule-driven caller does not need to special-case it.

        Returns:
            ``(x_hat, aux_loss)``, where ``aux_loss`` is the mean squared
            difference between the two decoders' outputs.
        """
        del step
        x_hat = self.decoder(z)
        return x_hat, (x_hat - self.linear_decode(z)).pow(2).mean()

    def uses_linear_path(self, step: int) -> bool:
        """Always ``False``: the linear path informs the loss, never the output."""
        del step
        return False


#: Config-name -> class, so ``variant:`` in YAML selects one.
VARIANTS = {"lavae": LAVAE, "folvae": FOLVAE, "added_loss": AddedLossVAE}
