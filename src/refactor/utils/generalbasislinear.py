import torch
import torch.nn as nn
import math
from typing import Callable, Optional

class GeneralBasisLinear(nn.Module):
    """
    Linear-like layer whose weights are fixed basis vectors evaluated at discrete points.

    Args:
        input_dim (int): N -- input vector length
        output_dim (int): K -- number of basis vectors (rows of W)
        basis: 'dct' | 'chebyshev' | callable (callable(k, points) -> Tensor(len(points),))
               If callable, it will be used to evaluate basis functions.
        points: optional 1D tensor of length input_dim specifying sample positions t_n.
                If None, defaults depend on `basis`:
                  - 'dct' -> t_n = (n + 0.5) * pi / N  (used by closed-form DCT)
                  - otherwise -> linear grid in [-1, 1]: linspace(-1,1,N)
        orthonormalize (bool): if True, orthonormalize the basis rows numerically via QR.
        bias (bool): whether to include a learnable bias (default False).
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 basis='dct',
                 points: Optional[torch.Tensor] = None,
                 orthonormalize: bool = True,
                 bias: bool = False,
                 device=None,
                 dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.N = int(input_dim)
        self.K = int(output_dim)
        self.basis = basis
        self.orthonormalize = bool(orthonormalize)

        # Build sample points if not provided
        if points is None:
            if basis == 'dct':
                # DCT-II uses t_n = (n + 0.5) * pi / N (angles)
                n = torch.arange(self.N, **factory_kwargs)
                points = (n + 0.5) * math.pi / float(self.N)   # angles in [0, pi)
            else:
                # default to linear grid in [-1, 1)
                points = torch.linspace(-1.0, 1.0, steps=self.N, **factory_kwargs)
        else:
            if not isinstance(points, torch.Tensor):
                points = torch.tensor(points, **factory_kwargs)
            if points.numel() != self.N:
                raise ValueError("points must have length equal to input_dim")

        self.register_buffer('_points', points)

        # Build raw basis matrix B (K x N)
        B = self._build_raw_basis(self.K, self.N, points, basis, factory_kwargs)

        # Optional orthonormalization of rows: compute thin QR of B^T,
        # then W = Q^T has orthonormal rows in R^N (Q is N x K with orthonormal columns).
        if self.orthonormalize:
            # B.T shape: (N x K)
            # QR: B.T = Q R  -> Q: (N x K) has orthonormal columns
            Q, R = torch.linalg.qr(B.T)   # Q: N x K
            W = Q.T                       # K x N, rows orthonormal
        else:
            # Optionally scale rows to have unit norm (just a choice)
            # compute row norms and divide, guarding against zero
            row_norms = B.norm(dim=1, keepdim=True)
            row_norms[row_norms == 0] = 1.0
            W = B / row_norms

        # Register W as a buffer (fixed weights)
        self.register_buffer('weight', W)  # shape (K, N)

        # Optional bias (learnable)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.K, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def _build_raw_basis(self, K, N, points: torch.Tensor, basis, factory_kwargs):
        """
        Produce a raw basis matrix B of shape (K x N) where
        B[k, n] = phi_k(points[n]).
        Supports 'dct', 'chebyshev', or a user-provided callable.
        """
        B = torch.empty((K, N), **factory_kwargs)

        if basis == 'dct':
            # closed-form DCT-II basis rows with analytic normalization
            # C_{k,n} = alpha_k * cos( pi/N * (n + 1/2) * k )
            # but note: `points` for DCT we passed are angles ( (n+0.5) * pi / N )
            # So phi_k(points[n]) = alpha_k * cos(k * points[n])
            n_k = torch.arange(K, **factory_kwargs)
            # compute alpha factors
            alpha = torch.empty((K,), **factory_kwargs)
            alpha[0] = math.sqrt(1.0 / N)
            if K > 1:
                alpha[1:] = math.sqrt(2.0 / N)
            # evaluate
            # points shape (N,), we want broadcasting to (K,N): k[:,None] * points[None,:]
            phases = torch.outer(n_k, points)   # (K, N)  (k * angle)
            B = alpha[:, None] * torch.cos(phases)

        elif basis == 'chebyshev':
            # Evaluate Chebyshev polynomials of the first kind: T_k(x) = cos(k * arccos(x))
            # We expect `points` in [-1,1] (angles arccos exist).
            # Alternative nodes (chebyshev nodes) can be provided by the user.
            x = points
            # numerical safety: clamp to [-1,1]
            x_clamped = torch.clamp(x, -1.0, 1.0)
            theta = torch.acos(x_clamped)  # angle in [0, pi]
            for k in range(K):
                # T_k(x) = cos(k * arccos(x))
                B[k, :] = torch.cos(k * theta)

            # Note: these T_k are not normalized; orthonormalization step will fix norms if requested.

        elif callable(basis):
            # basis is a callable: basis(k, points) -> 1D tensor length N
            for k in range(K):
                vec = basis(k, points)
                vec = torch.as_tensor(vec, **factory_kwargs)
                if vec.ndim != 1 or vec.numel() != N:
                    raise ValueError("basis callable must return 1D tensor of length N")
                B[k, :] = vec
        else:
            raise ValueError("basis must be 'dct', 'chebyshev', or a callable")

        return B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (..., N)  (last dim is feature)
        returns: shape (..., K)
        """
        # support batched x with last dim N
        if x.shape[-1] != self.N:
            raise ValueError(f"Expected input with last dim {self.N}, got {x.shape[-1]}")
        # weight is (K x N) -> we want to produce (..., K) = x @ weight.T
        out = x.matmul(self.weight.t())
        if self.bias is not None:
            out = out + self.bias
        return out

