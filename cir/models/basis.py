"""Fixed-basis linear layers.

A :class:`BasisLinear` behaves like ``nn.Linear`` but its weight matrix is *not*
learned: each row is a basis function (DCT-II or Chebyshev) sampled on a grid.
The layer therefore projects its input onto a fixed, well-conditioned subspace,
which is what lets the VAE variants replace a learned output map with a solvable
one — see :mod:`cir.utils.solvers`.

This module supersedes three near-duplicate implementations from the
pre-refactor tree (``dct.py``, ``generalbasislinear.py``, ``gpmethod.py``).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

__all__ = ["dct_basis", "chebyshev_basis", "BasisLinear", "DCTLinear", "ChebyshevLinear"]

BasisSpec = Union[str, Callable[[int, int, torch.Tensor], torch.Tensor]]


def dct_basis(num_basis: int, input_dim: int, points: torch.Tensor) -> torch.Tensor:
    """Build an orthonormal DCT-II basis matrix.

    Row ``k`` is ``alpha_k * cos(k * t)`` evaluated at ``points``, with
    ``alpha_0 = sqrt(1/N)`` and ``alpha_k = sqrt(2/N)`` otherwise — the scaling
    that makes the rows orthonormal on the standard DCT grid.

    Args:
        num_basis: Number of basis rows ``K``.
        input_dim: Length of each row ``N``.
        points: Sample positions, shape ``(N,)``. For a true DCT these are the
            angles ``(n + 0.5) * pi / N``; see :meth:`BasisLinear.default_points`.

    Returns:
        Tensor of shape ``(K, N)``.
    """
    alpha = torch.full((num_basis,), math.sqrt(2.0 / input_dim), dtype=points.dtype)
    if num_basis > 0:
        alpha[0] = math.sqrt(1.0 / input_dim)
    degrees = torch.arange(num_basis, dtype=points.dtype)
    return alpha[:, None] * torch.cos(torch.outer(degrees, points))


def chebyshev_basis(num_basis: int, input_dim: int, points: torch.Tensor) -> torch.Tensor:
    """Build a Chebyshev (first kind) basis matrix.

    Uses the stable recurrence ``T_{k+1} = 2x T_k - T_{k-1}`` rather than
    ``cos(k arccos x)``, which loses precision near ``|x| = 1``.

    Args:
        num_basis: Number of basis rows ``K``.
        input_dim: Length of each row ``N`` (unused; kept for a uniform signature).
        points: Sample positions in ``[-1, 1]``, shape ``(N,)``.

    Returns:
        Tensor of shape ``(K, N)``. Rows are *not* normalized; pass
        ``orthonormalize=True`` to :class:`BasisLinear` if you need that.
    """
    del input_dim  # signature parity with dct_basis
    rows = []
    for k in range(num_basis):
        if k == 0:
            rows.append(torch.ones_like(points))
        elif k == 1:
            rows.append(points.clone())
        else:
            rows.append(2 * points * rows[k - 1] - rows[k - 2])
    return torch.stack(rows, dim=0) if rows else points.new_zeros((0, points.numel()))


BASES = {"dct": dct_basis, "chebyshev": chebyshev_basis}


class BasisLinear(nn.Module):
    """A linear layer whose weights are fixed basis vectors.

    The weight matrix is registered as a buffer, not a parameter, so it moves
    with ``.to(device)`` and is saved in ``state_dict`` but never receives
    gradients.

    Args:
        input_dim: Input feature length ``N``.
        output_dim: Number of basis functions ``K`` (the output width).
        basis: ``"dct"``, ``"chebyshev"``, or a callable
            ``(K, N, points) -> Tensor(K, N)``.
        points: Optional sample grid of length ``N``. Defaults to the DCT angle
            grid for ``basis="dct"`` and to ``linspace(-1, 1, N)`` otherwise.
        orthonormalize: Orthonormalize the rows numerically via QR. Recommended
            for ``"chebyshev"``, which is otherwise badly conditioned for large
            ``K``; a no-op in exact arithmetic for ``"dct"``.
        bias: Add a learnable bias. Off by default — the point of the layer is a
            fixed projection.

    Raises:
        ValueError: If ``basis`` is unrecognized, or ``points`` has the wrong
            length.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        basis: BasisSpec = "dct",
        points: Optional[torch.Tensor] = None,
        orthonormalize: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.basis = basis

        points = (
            self.default_points(self.input_dim, basis)
            if points is None
            else torch.as_tensor(points, dtype=torch.float32)
        )
        if points.numel() != self.input_dim:
            raise ValueError(
                f"points must have length input_dim ({self.input_dim}), got {points.numel()}"
            )
        self.register_buffer("points", points)

        weight = self._build_basis(points)
        if orthonormalize:
            # Thin QR of Wᵀ gives orthonormal columns; transpose back for rows.
            q, _ = torch.linalg.qr(weight.T)
            weight = q.T
        self.register_buffer("weight", weight)

        self.bias = nn.Parameter(torch.zeros(self.output_dim)) if bias else None

    @staticmethod
    def default_points(input_dim: int, basis: BasisSpec) -> torch.Tensor:
        """Return the natural sample grid for a basis.

        Args:
            input_dim: Grid length ``N``.
            basis: Basis name or callable.

        Returns:
            ``(n + 0.5) * pi / N`` for the DCT, otherwise ``linspace(-1, 1, N)``.
        """
        if basis == "dct":
            return (torch.arange(input_dim, dtype=torch.float32) + 0.5) * math.pi / input_dim
        return torch.linspace(-1.0, 1.0, steps=input_dim, dtype=torch.float32)

    def _build_basis(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluate the configured basis into a ``(K, N)`` matrix."""
        if callable(self.basis):
            weight = self.basis(self.output_dim, self.input_dim, points)
        elif isinstance(self.basis, str) and self.basis.lower() in BASES:
            weight = BASES[self.basis.lower()](self.output_dim, self.input_dim, points)
        else:
            raise ValueError(
                f"basis must be one of {sorted(BASES)} or a callable, got {self.basis!r}"
            )

        expected = (self.output_dim, self.input_dim)
        if tuple(weight.shape) != expected:
            raise ValueError(f"basis produced shape {tuple(weight.shape)}, expected {expected}")
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` onto the basis.

        Args:
            x: Shape ``(..., input_dim)``.

        Returns:
            Shape ``(..., output_dim)``.

        Raises:
            ValueError: If the trailing dimension of ``x`` is not ``input_dim``.
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected last dim {self.input_dim}, got {x.shape[-1]}")
        out = x.matmul(self.weight.t())
        return out if self.bias is None else out + self.bias

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"basis={self.basis!r}, bias={self.bias is not None}"
        )


class DCTLinear(BasisLinear):
    """:class:`BasisLinear` fixed to the orthonormal DCT-II basis."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        super().__init__(input_dim, output_dim, basis="dct", bias=bias)


class ChebyshevLinear(BasisLinear):
    """:class:`BasisLinear` fixed to the Chebyshev basis, orthonormalized by default."""

    def __init__(
        self, input_dim: int, output_dim: int, orthonormalize: bool = True, bias: bool = False
    ):
        super().__init__(
            input_dim, output_dim, basis="chebyshev", orthonormalize=orthonormalize, bias=bias
        )
