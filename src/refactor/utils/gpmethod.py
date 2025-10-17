# basis_linear.py
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional

# ---------------------------
# Basis generators (torch)
# ---------------------------

def dct2_basis_row(n: int, k: int, N: int, orthonormal: bool = True) -> torch.Tensor:
    """
    Return the k-th DCT-II basis vector sampled at n = 0..N-1 as a 1D tensor length N.
    Uses formula: cos(pi/N * (n + 1/2) * k) with orthonormal scaling.
    """
    n_idx = torch.arange(N, dtype=torch.float32)
    vec = torch.cos(math.pi / N * (n_idx + 0.5) * float(k))
    if orthonormal:
        if k == 0:
            vec = vec * math.sqrt(1.0 / N)
        else:
            vec = vec * math.sqrt(2.0 / N)
    return vec

def dct2_basis_matrix(N: int, K: int, orthonormal: bool = True) -> torch.Tensor:
    """
    Build (K x N) DCT-II orthonormal basis matrix. Row k is basis k.
    """
    rows = [dct2_basis_row(n=..., k=k, N=N, orthonormal=orthonormal) for k in range(K)]
    return torch.stack(rows, dim=0)  # shape (K, N)

def chebyshev_basis_vector(k: int, pts: torch.Tensor) -> torch.Tensor:
    """
    Evaluate Chebyshev polynomial T_k(x) at points pts (torch tensor).
    Stable recurrence:
        T_0(x) = 1
        T_1(x) = x
        T_{k+1} = 2 x T_k - T_{k-1}
    Returns length-pts.shape[0] tensor.
    """
    pts = pts.float()
    if k == 0:
        return torch.ones_like(pts)
    if k == 1:
        return pts
    Tkm1 = pts  # T1
    Tkm2 = torch.ones_like(pts)  # T0
    for _ in range(2, k+1):
        Tk = 2 * pts * Tkm1 - Tkm2
        Tkm2, Tkm1 = Tkm1, Tk
    return Tkm1

def chebyshev_basis_matrix(N: int, K: int, pts: Optional[torch.Tensor] = None, sample_strategy: str = "linspace") -> torch.Tensor:
    """
    Build (K x N) Chebyshev basis matrix where each row k is T_k sampled.
    pts: optional torch tensor length N with sample points in [-1,1]. If None, create according to sample_strategy:
        - "linspace": uniform points in [-1,1], pts = linspace(-1,1,N)
        - "cheb_nodes": Chebyshev nodes cos(pi*(2j+1)/(2N)) (usually better conditioning for interpolation)
    """
    if pts is None:
        if sample_strategy == "cheb_nodes":
            j = torch.arange(N, dtype=torch.float32)
            pts = torch.cos(math.pi*(2*j + 1)/(2.0*N))
        else:
            pts = torch.linspace(-1.0, 1.0, N, dtype=torch.float32)
    rows = [chebyshev_basis_vector(k, pts) for k in range(K)]
    return torch.stack(rows, dim=0)  # (K, N)

# ---------------------------
# Generic BasisLinear module
# ---------------------------

class BasisLinear(nn.Module):
    """
    A fixed linear layer whose rows are basis vectors evaluated at discrete sample points.

    Parameters
    ----------
    input_dim : int
        N, length of input vector
    output_dim : int
        K, number of basis functions to project onto (rows of W)
    basis : str or callable
        If 'dct' -> uses DCT-II (orthonormal by default).
        If 'chebyshev' -> uses Chebyshev polynomials T_k evaluated at pts in [-1,1].
        If callable -> call as basis_matrix = basis(input_dim, output_dim, **basis_kwargs)
            basis must return a torch.Tensor shaped (K, N).
    basis_kwargs : dict
        Extra keyword args forwarded to the basis callable (e.g., sample strategy for Chebyshev).
    """
    def __init__(self, input_dim: int, output_dim: int, basis='dct', basis_kwargs=None, device=None, dtype=torch.float32):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.basis = basis
        self.basis_kwargs = basis_kwargs or {}
        self.device = device
        self.dtype = dtype

        W = self._build_weight_matrix()  # (K, N)
        # register as buffer (fixed, not learnable)
        self.register_buffer("weight", W.to(device=device, dtype=dtype))

    def _build_weight_matrix(self) -> torch.Tensor:
        N = self.input_dim
        K = self.output_dim
        if callable(self.basis):
            # user-supplied builder: must return (K, N) torch tensor
            M = self.basis(N, K, **self.basis_kwargs)
            assert isinstance(M, torch.Tensor) and M.shape == (K, N)
            return M
        elif isinstance(self.basis, str):
            b = self.basis.lower()
            if b == 'dct':
                return dct2_basis_matrix(N, K, orthonormal=self.basis_kwargs.get('orthonormal', True))
            elif b == 'chebyshev':
                pts = self.basis_kwargs.get('pts', None)
                sample_strategy = self.basis_kwargs.get('sample_strategy', 'linspace')
                return chebyshev_basis_matrix(N, K, pts=pts, sample_strategy=sample_strategy)
            else:
                raise ValueError(f"Unknown basis string '{self.basis}'")
        else:
            raise ValueError("basis must be 'dct', 'chebyshev', or a callable builder")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input(s) x of shape (..., input_dim) to (..., output_dim).

        Behavior matches torch.nn.Linear except weights are fixed and bias is zero.
        """
        # flatten trailing dimension and do matmul
        # We do x @ W.T so Row k of W multiplies x
        orig_shape = x.shape[:-1]
        assert x.shape[-1] == self.input_dim, f"expected last dim {self.input_dim}, got {x.shape[-1]}"
        y = x.matmul(self.weight.t())
        return y

# Convenience subclass
class DCTLinear(BasisLinear):
    def __init__(self, input_dim: int, output_dim: int, orthonormal: bool = True, device=None, dtype=torch.float32):
        super().__init__(input_dim=input_dim, output_dim=output_dim, basis='dct', basis_kwargs={'orthonormal': orthonormal}, device=device, dtype=dtype)

# ---------------------------
# Example usage:
# ---------------------------
if __name__ == "__main__":
    # Example: 784 -> 128 via DCT-II
    layer = DCTLinear(784, 128)
    x = torch.randn(32, 784)
    y = layer(x)  # (32, 128)
    print("y shape:", y.shape)

