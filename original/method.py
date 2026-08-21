"""Iterative least-squares in a Chebyshev basis, via Householder QR.

Part of the *original*, pre-refactor codebase; see ``README.md``.

Solves ``A x ~= b`` by restricting ``x`` to the span of the first k Chebyshev
polynomials and growing k one column at a time, extending the QR factorization
rather than recomputing it. See ``../refactor/cir/utils/solvers.py`` for the
cleaned-up version.
"""

import copy
import numpy as np

from scipy.linalg import solve_triangular
from scipy.special import eval_chebyt

from funcs import compute_H, apply_H, find_v


def solver(A, b, tol=1e-5, max_iter=20, verbose=True):
    """
    Solve A x ~= b over a growing Chebyshev basis.

    Args:
        A (ndarray): Coefficient matrix, shape (m, n).
        b (ndarray): Right-hand side, shape (m,).
        tol (float): Stop once the relative residual falls to or below this.
        max_iter (int): Maximum basis columns to add.
        verbose (bool): Print the iteration count and residual comparison.

    Returns:
        ndarray: The coefficients z of the solution in the Chebyshev basis. The
        solution in the original variables is X @ z, where X holds the basis
        columns.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    m = len(A)
    n = len(A[0])
    # copy for checking
    bcheck = copy.deepcopy(b)


    # evenly spaced pts on the interval -1, 1 to evaluate at chebychev polynomials
    pts = np.linspace(-1, 1, n)

    # Set tolerance and variable that will be used to check if error is below tolerance
    tol = 1e-5
    check = tol + 1

    k = 0
    vvecs = []

    # Solver loop continues until below tolerance or full least squares solution found (all possible cols added)
    while check > tol and k < n and k < m and k < max_iter:
        new_col = (eval_chebyt(k, pts)) # new X col from chebychev pts
        # new_col = np.zeros(n)
        # new_col[k] = 1.0

        # loop for steps after first
        if k:
            X = np.column_stack((X, new_col))
            Y = np.column_stack((Y, A @ X[:, k]))

            # Add new Y column to R
            R[:, k] = copy.deepcopy(Y[:, k])
            # Apply old transformations to relevant entries in new column
            for v in range(len(vvecs)):
                R[v:, k] = apply_H(R[v:, k], vvecs[v])

            # Find new vvector and apply transformations
            vvecs.append(find_v(R[k:, k], k))
            R[k:, k] = apply_H(R[k:, k], vvecs[k])
            # pad new vvector for computing H and Q (unncessary). Also for bcheck
            padded = np.insert(vvecs[k], 0, np.zeros(k))
            # print(f'padded: {padded}')
            H = compute_H(padded)
            Q = Q @ H
            bcheck = apply_H(bcheck, padded)

        # First step initialization
        else:
            # make X and Y
            X = np.array(new_col)
            # Y = AX
            Y = np.array(A @ X)

            # Find first v vector for Householder
            vvecs.append(find_v(Y, k))

            # Q starts as H. Don't really need Q. R is empty, replace columns as computed
            Q = compute_H(vvecs[k])
            R = np.zeros((m, n))
            R[:, k] = apply_H(Y, vvecs[k])
            # must apply to both sides to maintain least squares solution
            bcheck = apply_H(b, vvecs[k])
            # add dim for doing Y.dot(z) below
            Y = Y[:, np.newaxis]
            
        # Compute QR of current Y using householder. Q is H's multiplied by each other. just right multiply (slides) by new H for each column
        # One new householder vector per column. Apply all of previous transformations to new Y column, THEN get new H for that col (so it will cancel)
        # Solve least squares problem using QR iteration. QRz ~= b, Rz = Q^Tb, Solve Rz = vec upper triangular like RREF.
        # Use numpy. Then substitute z back in and take b - Yz and check less than tol then repeat

        # slice to ensure our matrix is square
        s1 = m - (k + 1)
        s2 = n - (k + 1)
        if k < (n-1) and k < (m-1):
            z = solve_triangular(R[:-s1,:-s2], (bcheck)[:-s1])
        # If reach last row or column, make square and solve for least squares solution (bcheck preserves least squares) bcheck = Q.T @ b
        else:
            if m > n:
                z = solve_triangular(R[:-s1,:], (bcheck)[:-s1])
            elif m < n:
                z = solve_triangular(R[:,:-s2], (bcheck))
            else:
                z = solve_triangular(R, (bcheck))


        # Relative residual of the current approximation, measured in the
        # ORIGINAL system. The first version compared `bcheck` (which is Q^T b)
        # against `Y.dot(z)` (which is not transformed), so the two sides lived
        # in different frames and `check` never fell below tol -- the loop always
        # ran to the iteration cap regardless of how good the fit was.
        # X is 1-D on the very first iteration and 2-D afterwards.
        check = (np.linalg.norm(b - A.dot(X.reshape(n, -1) @ z)) / np.linalg.norm(b))
        k += 1

    if verbose:
        x = X.reshape(n, -1) @ z
        xc, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        print(f'k: {k}')
        print(f'check: {check}')
        print(f'this method: {np.linalg.norm(b - A.dot(x)) / np.linalg.norm(b)}')
        print(f'numpy lstsq: {np.linalg.norm(b - A.dot(xc)) / np.linalg.norm(b)}')

    return z
