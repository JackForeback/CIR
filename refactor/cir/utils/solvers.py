"""Iterative least-squares in a Chebyshev basis, via Householder QR.

Solves ``A x ~= b`` by restricting ``x`` to the span of the first ``k`` Chebyshev
polynomials sampled on a grid, growing ``k`` one column at a time until the
relative residual falls below a tolerance. Because each new basis column extends
the previous QR factorization rather than restarting it, the *k*-th iteration
costs one Householder reflection instead of a fresh decomposition.

This is the "linear solver" the VAE variants were sketched around: it produces a
smooth, low-order approximation of a decoder mapping instead of a dense
least-squares fit, which is what makes it usable as a regularizer.

The Householder helpers are also exported on their own — they are the operational
core and are worth testing (and reusing) directly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import eval_chebyt

__all__ = [
    "householder_vector",
    "apply_householder",
    "householder_matrix",
    "chebyshev_basis_column",
    "iterative_chebyshev_ls",
]


def householder_vector(a: np.ndarray) -> np.ndarray:
    """Build the Householder vector that reflects ``a`` onto the first axis.

    The reflector is ``H = I - 2 v vᵀ / (vᵀ v)`` with ``v = a + sign(a₀)‖a‖e₁``.
    The sign choice avoids catastrophic cancellation when ``a`` already points
    close to ``e₁``.

    Args:
        a: Vector to reflect.

    Returns:
        The (unnormalized) Householder vector ``v``. A zero vector is returned
        for a zero input, which :func:`apply_householder` treats as a no-op.
    """
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a

    alpha = np.linalg.norm(a)
    if alpha == 0:
        return a.copy()

    v = a.copy()
    v[0] += math.copysign(alpha, a[0])
    return v


def apply_householder(y: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply the reflector defined by ``v`` to ``y``.

    Computed as ``y - 2 v (vᵀy)/(vᵀv)``, never forming the ``H`` matrix.

    Args:
        y: Vector to transform.
        v: Householder vector from :func:`householder_vector`.

    Returns:
        A new, reflected vector. Returns ``y`` unchanged when ``v`` is empty or
        zero, so degenerate columns cost nothing.
    """
    if v is None or v.size == 0:
        return y
    denom = float(np.dot(v, v))
    if denom == 0:
        return y
    return y - (2.0 / denom) * v * float(np.dot(v, y))


def householder_matrix(v: np.ndarray) -> np.ndarray:
    """Materialize the reflector ``H = I - 2 v vᵀ / (vᵀ v)``.

    Only needed for inspection and tests; the solver uses
    :func:`apply_householder` instead.

    Args:
        v: Householder vector.

    Returns:
        The dense ``(len(v), len(v))`` reflection matrix.
    """
    v = np.asarray(v, dtype=float).reshape(-1, 1)
    denom = float(np.dot(v.ravel(), v.ravel()))
    if denom == 0:
        return np.eye(v.shape[0])
    return np.eye(v.shape[0]) - 2.0 * (v @ v.T) / denom


def chebyshev_basis_column(k: int, nodes: np.ndarray) -> np.ndarray:
    """Evaluate the Chebyshev polynomial ``T_k`` at ``nodes``.

    Args:
        k: Polynomial degree.
        nodes: Evaluation points, expected in ``[-1, 1]``.

    Returns:
        ``T_k(nodes)``, same shape as ``nodes``.
    """
    return eval_chebyt(k, nodes)


def iterative_chebyshev_ls(
    A: np.ndarray,
    b: np.ndarray,
    max_iter: int = 20,
    tol: float = 1e-5,
    nodes: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Approximately solve ``A x ~= b`` over a growing Chebyshev basis.

    Each iteration appends one basis column ``T_k``, maps it through ``A``,
    applies every previous Householder reflection to the result, computes one new
    reflection to zero the entries below the diagonal, and back-substitutes for
    the coefficients. Iteration stops on ``tol`` or when the basis is exhausted.

    Args:
        A: Coefficient matrix, shape ``(m, n)``.
        b: Right-hand side, shape ``(m,)``.
        max_iter: Maximum basis columns to add.
        tol: Stop once the relative residual ``‖b - Ax‖ / ‖b‖`` is at or below
            this value.
        nodes: Length-``n`` evaluation grid for the basis. Defaults to a uniform
            grid on ``[-1, 1]``.
        verbose: Print the residual after each iteration.

    Returns:
        ``(x, diagnostics)``. ``x`` has shape ``(n,)``; ``diagnostics`` carries
        ``k`` (columns used), ``residual``, and — when at least one column was
        added — the coefficients ``z``, basis ``X``, factor ``R``, and the
        Householder vectors ``vvecs``.

    Raises:
        ValueError: If ``A`` and ``b`` have mismatched leading dimensions.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError(f"A has {m} rows but b has {b.shape[0]} entries")

    if nodes is None:
        nodes = np.linspace(-1.0, 1.0, n)

    b_norm = np.linalg.norm(b) + 1e-16
    bcheck = b.copy()      # accumulates Qᵀb as reflections are applied
    X_cols: list = []      # basis columns, each length n
    vvecs: list = []       # Householder vectors, each padded to length m
    R = np.zeros((m, n), dtype=float)
    residual = 1.0
    z = None

    for k in range(min(max_iter, n, m)):
        new_col = np.asarray(chebyshev_basis_column(k, nodes)).reshape(-1)
        X_cols.append(new_col)

        # Map the basis column through A, then into the existing QR frame.
        y = A.dot(new_col)
        for v in vvecs:
            y = apply_householder(y, v)

        # One new reflection zeroes everything below row k in this column.
        v_small = householder_vector(y[k:].copy())
        v_full = (
            np.zeros(m)
            if np.linalg.norm(v_small) == 0
            else np.concatenate([np.zeros(k), v_small])
        )
        vvecs.append(v_full)

        y = apply_householder(y, v_full)
        R[:, k] = y
        bcheck = apply_householder(bcheck, v_full)

        # Back-substitute on the leading (k+1)x(k+1) triangular block.
        R_top, b_top = R[: k + 1, : k + 1], bcheck[: k + 1]
        try:
            z = solve_triangular(R_top, b_top, lower=False)
        except (ValueError, np.linalg.LinAlgError):
            # A (near-)singular block means this column added no rank; fall back
            # to a least-squares solve so the iteration can still make progress.
            z, *_ = np.linalg.lstsq(R_top, b_top, rcond=None)

        X = np.column_stack(X_cols)
        residual = np.linalg.norm(b - A.dot(X.dot(z))) / b_norm

        if verbose:
            print(f"iter {k}: residual={residual:.3e}, columns={k + 1}")
        if residual <= tol:
            break

    if not X_cols:
        # Degenerate input (n or m is 0): defer to numpy.
        x_final, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x_final, {
            "k": 0,
            "residual": float(np.linalg.norm(b - A.dot(x_final)) / b_norm),
        }

    X = np.column_stack(X_cols)
    return X.dot(z), {
        "k": len(X_cols),
        "residual": float(residual),
        "z": z,
        "X": X,
        "R": R,
        "vvecs": vvecs,
    }
