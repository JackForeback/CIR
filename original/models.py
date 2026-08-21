"""Model definitions for the original scripts.

Part of the *original*, pre-refactor codebase; see ``README.md``.

`LinearClassifier` and `VAE` are the two models the original experiments train.
The three variants below (`FOLVAE`, `LAVAE`, `ALVAE`) were exploratory sketches
around one idea: alternate the decoder with a purely *linear* path so the
reconstruction lands in a subspace that could be solved for directly. They are
kept because they record the direction the work was heading. Each one now
constructs and runs -- as written they all called `super(VAE, self).__init__()`
from a sibling class, unpacked the encoder's single output as a pair, and
referenced an undefined `solver`, so none of them could be instantiated.
"""

import torch
import torch.nn as nn


# Model definition
class LinearClassifier(nn.Module):
    """A single linear layer producing one score per class.

    The bias starts at zero so every decision boundary passes through the origin
    at step 0, which keeps early asymmetry between classes attributable to the
    data geometry rather than the initialization.

    Args:
        input_dim (int): Number of input features.
        num_classes (int): Number of output scores.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        nn.init.zeros_(self.linear.bias)  # Set bias to zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class VAE(nn.Module):
    """A one-hidden-layer MLP variational autoencoder.

    Args:
        input_dim (int): Dimensionality of the data (784 for flattened MNIST).
        latent_dim (int): Size of the latent space.
    """

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)
        self.Decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def reparameterization(self, mean, std):
        """Sample the posterior differentiably: z = mean + std * eps."""
        epsilon = torch.randn_like(std)      # sampling epsilon
        return mean + std * epsilon         # reparameterization trick

    def forward(self, x):
        """Encode, sample, decode.

        Returns:
            tuple: (x_hat, kl_loss, mean, log_var). `kl_loss` is summed over the
            batch, so it is on a very different scale from a mean-reduced
            reconstruction loss -- weight it accordingly.
        """
        x = self.Encoder(x)
        mean = self.mu(x)
        log_var = self.log_var(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # log var -> std
        x_hat = self.Decoder(z)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return x_hat, kl_loss, mean, log_var


class AlternatingVAE(VAE):
    """Base class for the three exploratory "alternating decoder" variants.

    All three alternate between the ordinary nonlinear decoder (on odd steps)
    and a single *linear* decode path (on even steps). The premise is that a
    reconstruction produced by a linear map lives in a known, low-dimensional
    subspace, and is therefore something the least-squares routine in
    `method.py` could solve for directly instead of learning.

    That solver was never wired into training -- the original code left it as a
    comment reading "LINEAR SOLVER AND THEN BACKPROP. HOW DO I DO THIS BEST?".
    These classes preserve the structure that was actually written and run
    correctly; `../refactor/cir/models/alvae.py` carries the idea forward with a
    fixed orthonormal basis in place of the unfinished solver step.

    Args:
        input_dim (int): Dimensionality of the data.
        latent_dim (int): Size of the latent space.
    """

    def __init__(self, input_dim, latent_dim):
        super().__init__(input_dim, latent_dim)
        # The linear decode path: latent -> 128 -> input, with no activation.
        self.linear128 = nn.Linear(latent_dim, 128)
        self.output = nn.Linear(128, input_dim)

    def linear_decode(self, z):
        """Decode through the linear path only.

        Args:
            z (Tensor): Latent codes, shape (batch, latent_dim).

        Returns:
            Tensor: Reconstruction of shape (batch, input_dim).
        """
        return self.output(self.linear128(z))

    def forward(self, x, current_step=1):
        """Alternate between the learned decoder and the linear path.

        Args:
            x (Tensor): Input batch, shape (batch, input_dim).
            current_step (int): Training step; odd steps use the learned decoder.

        Returns:
            tuple: (x_hat, mean, log_var).
        """
        hidden = self.Encoder(x)
        mean, log_var = self.mu(hidden), self.log_var(hidden)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        x_hat = self.Decoder(z) if current_step % 2 else self.linear_decode(z)
        return x_hat, mean, log_var


# Fixed Output Layer Variational AutoEncoder
class FOLVAE(AlternatingVAE):
    """Alternating VAE whose linear path ends in a *fixed* output layer.

    The output layer is frozen at initialization, so on linear steps only
    `linear128` can adapt. This is the cheapest stand-in for the intended
    "solve, don't learn, the output map" step.
    """

    def __init__(self, input_dim, latent_dim):
        super().__init__(input_dim, latent_dim)
        for param in self.output.parameters():
            param.requires_grad_(False)


# Linear Alternating Variational AutoEncoder
class LAVAE(AlternatingVAE):
    """Alternating VAE, used unmodified.

    This is the plain alternating model: nonlinear decoder on odd steps, linear
    path on even steps, both fully learned. It is the baseline the other two
    variants are compared against.
    """


# Added Loss Variational AutoEncoder
class ALVAE(AlternatingVAE):
    """Alternating VAE that adds the two decoders' disagreement to the loss.

    Instead of switching between the paths, this variant keeps the nonlinear
    decoder and reports how far its reconstruction is from what the linear path
    would produce. Driving that gap down pulls the learned decoder toward a
    linearly-representable solution -- the "added loss".

    See `../refactor/cir/models/alvae.py` for the completed version of the idea.
    """

    def forward(self, x, current_step=1):
        """Run the pass and attach the added-loss term.

        Args:
            x (Tensor): Input batch, shape (batch, input_dim).
            current_step (int): Training step.

        Returns:
            tuple: (x_hat, mean, log_var, added_loss).
        """
        hidden = self.Encoder(x)
        mean, log_var = self.mu(hidden), self.log_var(hidden)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        x_hat = self.Decoder(z)
        added_loss = (x_hat - self.linear_decode(z)).pow(2).mean()

        return x_hat, mean, log_var, added_loss
