"""Householder reflection primitives used by ``method.py``.

Part of the *original*, pre-refactor codebase; see ``README.md``.
"""

import math
import numpy as np


def compute_H(vector):
    """
    Materialize the Householder reflector H = I - 2 v v^T / (v^T v).

    Args:
        vector (ndarray): The Householder vector v.

    Returns:
        ndarray: The (len(v), len(v)) reflection matrix.
    """
    v = np.asarray(vector, dtype=float).reshape(-1, 1)  # column vector
    denom = float(np.dot(v.ravel(), v.ravel()))
    if denom == 0:
        return np.eye(v.shape[0])
    return np.eye(v.shape[0]) - 2 * ((v @ v.T) / denom)


def apply_H(u, v):
    """
    Apply the reflector defined by v to u, without forming H.

    Args:
        u (ndarray): Vector to transform.
        v (ndarray): Householder vector.

    Returns:
        ndarray: The reflected vector u - 2 v (v.u)/(v.v).
    """
    denom = v @ v
    if denom == 0:
        return u
    scalar = 2 * ((v @ u) / denom)
    return u - (scalar * v)


def find_v(col, k):
    """
    Build the Householder vector that reflects `col` onto the first axis.

    The sign of alpha follows col[0] to avoid cancellation when col already
    points close to e_1.

    Args:
        col (ndarray): The column to reflect.
        k (int): Column index. Unused; kept for the original call signature.

    Returns:
        ndarray: The (unnormalized) Householder vector.
    """
    # two norm of col, same sign for e_i
    alpha = np.linalg.norm(col)
    alpha = math.copysign(alpha, col[0])
    evec = np.zeros(len(col))
    evec[0] = 1
    return col + alpha * evec
