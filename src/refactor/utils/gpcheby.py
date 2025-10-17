# chebyshev_iterative_solver.py
import copy
import math
import numpy as np
from scipy.special import eval_chebyt
from scipy.linalg import solve_triangular

def householder_vector(a: np.ndarray) -> np.ndarray:
    """
    Compute the Householder vector v for vector a such that
    H = I - 2 v v^T / (v^T v) zeros all but the first entry of a.

    Returns v (not normalized); applying H to a: a' = a - 2 v (v^T a)/(v^T v)
    This follows the classic construction: v = a + sign(a0) * ||a|| * e1
    """
    a = a.astype(float, copy=False)
    if a.size == 0:
        return a
    alpha = np.linalg.norm(a)
    if alpha == 0:
        # zero vector -> no reflection
        v = a.copy()
        return v
    # sign to avoid cancellation; if a[0] is zero, copysign handles it
    alpha = math.copysign(alpha, a[0])
    v = a.copy()
    v[0] += alpha
    return v

def apply_householder(y: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Apply the Householder transform H = I - 2 v v^T / (v^T v) to vector y.
    v may be all zeros (no-op) — handle stable division.
    """
    if v is None or v.size == 0:
        return y
    denom = np.dot(v, v)
    if denom == 0:
        return y
    tau = 2.0 / denom
    # y <- y - tau * v * (v^T y)
    vy = np.dot(v, y)
    y = y - tau * v * vy
    return y

def chebyshev_basis_column(k: int, n: int, nodes: np.ndarray) -> np.ndarray:
    """
    Evaluate Chebyshev polynomial T_k at nodes array of length n.
    Uses scipy.special.eval_chebyt which expects nodes in [-1,1].
    Returns length-n column vector.
    """
    return eval_chebyt(k, nodes)

def iterative_chebyshev_ls(A: np.ndarray,
                           b: np.ndarray,
                           max_iter: int = 20,
                           tol: float = 1e-5,
                           nodes: np.ndarray = None,
                           verbose: bool = False):
    """
    Iterative least-squares using Chebyshev basis and manual iterative Householder QR.

    A: shape (m, n)
    b: shape (m,)
    nodes: length-n array of evaluation points for Chebyshev polynomials
           default: uniform nodes in [-1,1]
    Returns:
        x_approx: approximate solution in R^n (constructed from X @ z)
        diagnostics: dict with details (k, residual, z, X, R, vvecs)
    Notes:
      - This follows your original algorithm's structure:
          * Build X (n x k) with columns T_k(nodes)
          * Y[:,k] = A @ X[:,k]
          * Apply previous Householders to Y[:,k]
          * Compute new Householder v (to zero below row k)
          * Apply v to Y[:,k] and to bcheck (b transformed by Q^T)
          * Fill R[:,k] with transformed column
          * Solve triangular R_top z = bcheck_top
          * Compute residual ||b - A (X z)|| / ||b|| and stop if below tol
    """
    # shapes
    A = np.asarray(A, dtype=float)
    b  = np.asarray(b, dtype=float).ravel()
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError("A and b dimension mismatch")

    if nodes is None:
        nodes = np.linspace(-1.0, 1.0, n)  # default nodes

    # bookkeeping
    b_orig = b.copy()
    bcheck = b.copy()   # will be Q^T b (updated as we append v's)
    X_cols = []         # list of columns (length-n)
    vvecs = []          # list of Householder vectors padded to length m (top entries zero where appropriate)
    R = np.zeros((m, n), dtype=float)
    k = 0
    residual = np.linalg.norm(b_orig)
    z = None

    # iterate
    while k < max_iter and k < n and k < m:
        # 1) compute new basis column (length n)
        new_col = chebyshev_basis_column(k, n, nodes)  # shape (n,)
        X_cols.append(new_col.reshape(-1,))            # append as length-n

        # 2) compute y = A @ new_col  (length m)
        y = A.dot(new_col)

        # 3) apply all previous Householders to this new y column
        #    (each vvec has zeros for entries < v_index)
        for v in vvecs:
            y = apply_householder(y, v)

        # 4) compute new Householder to zero entries below row k
        #    work on the tail y[k:], create v_small, then pad to full length
        tail = y[k:].copy()
        v_small = householder_vector(tail)
        if np.linalg.norm(v_small) == 0:
            # No new information in this column (it's already zero in tail)
            v_full = np.zeros(m)
        else:
            # pad top k zeros
            v_full = np.concatenate([np.zeros(k), v_small])
        vvecs.append(v_full)

        # 5) apply the new Householder to y (so R[:,k] is the transformed column)
        y = apply_householder(y, v_full)
        R[:, k] = y  # store the transformed column in R (m-length)

        # 6) apply the new Householder to bcheck as well (bcheck = Q^T b)
        bcheck = apply_householder(bcheck, v_full)

        # 7) build X matrix (n x (k+1)) and compute least-squares coefficients z on transformed system
        X = np.column_stack(X_cols)  # shape (n, k+1)
        # We have R[0:k+1, 0:k+1] triangular block; solve that system
        # Extract top-left triangular
        R_top = R[:k+1, :k+1]       # shape (k+1, k+1)
        b_top = bcheck[:k+1]        # transformed RHS top part

        # Solve triangular R_top z = b_top for z (len k+1)
        # Some R_top diagonal entries may be tiny; solve_triangular handles well.
        try:
            z = solve_triangular(R_top, b_top, lower=False)
        except Exception as exc:
            # fallback to least-squares if triangular solve fails numerically
            z, *_ = np.linalg.lstsq(R_top, b_top, rcond=None)

        # 8) compute the residual in original space: r = b - A @ (X @ z)
        x_approx = X.dot(z)         # length-n
        residual = np.linalg.norm(b_orig - A.dot(x_approx)) / (np.linalg.norm(b_orig) + 1e-16)

        if verbose:
            print(f"iter {k}: residual={residual:.3e}, dims k+1={k+1}")

        # check stopping condition
        if residual <= tol:
            k += 1
            break

        k += 1

    # final output
    # If no columns were added (k==0), return least-squares via numpy
    if len(X_cols) == 0:
        x_final, *_ = np.linalg.lstsq(A, b_orig, rcond=None)
        diagnostics = {"k": 0, "residual": np.linalg.norm(b_orig - A.dot(x_final)) / (np.linalg.norm(b_orig) + 1e-16)}
        return x_final, diagnostics

    # final x in original variable space
    X = np.column_stack(X_cols)  # n x K_final
    z_final = z
    x_final = X.dot(z_final)     # n-vector

    diagnostics = {
        "k": len(X_cols),
        "residual": residual,
        "z": z_final,
        "X": X,
        "R": R,
        "vvecs": vvecs
    }
    return x_final, diagnostics


# Example usage (replicating your example sizes):
if __name__ == "__main__":
    import scipy.sparse as sparse
    # build your A and b like in your example
    r1 = np.zeros(100)
    r1[0] = 1
    A = sparse.diags([1, -2, 1], [0, 1, 2], shape=(98,100)).toarray()
    A = np.vstack([r1, A])
    r1[0] = 0
    r1[99] = 1
    A = np.vstack([A, r1])
    A = A * 100.0

    pts = np.linspace(0, 2*np.pi, 100)
    pts = np.sin(pts)
    b = pts.copy()

    x_approx, diag = iterative_chebyshev_ls(A, b, max_iter=20, tol=1e-5, verbose=True)
    print("k:", diag["k"])
    print("residual:", diag["residual"])
    # compare to numpy least-squares
    xc, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print("numpy ls residual:", np.linalg.norm(b - A.dot(xc)) / (np.linalg.norm(b) + 1e-16))

