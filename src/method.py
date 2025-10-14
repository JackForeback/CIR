import copy, math, scipy
import numpy as np

from scipy.special import eval_chebyt
from scipy.linalg import solve_triangular
from funcs import compute_H, apply_H, find_v

from scipy import sparse

# r1 = np.zeros(100)
# r1[0] = 1
# A = sparse.diags([1, -2, 1], [0, 1, 2], shape=(98,100)).toarray()
# A = np.vstack([r1, A])
# r1[0] = 0
# r1[99] = 1
# A = np.vstack([A, r1])
# A = A * 100
# # print(A)
# # print(A.shape)

# pts = np.linspace(0, 2*np.pi, 100)

# pts = np.sin(pts)

# b = np.array(pts)
# print(b)
# print(b.shape)

# input array. harcoded for example
# m=5
# n=3
# A=np.random.rand(m,n)
# b = np.random.rand(m,)

# A = np.array([[1,-1,1],[1,-.5,.25],[1,0,0],[1,.5,.25],[1,1,1]])
# b = np.array([1,.5,0,.5,2])

def solver(A, b):
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
    while check > tol and k < n and k < m and k < 20:
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


        # and plug in z for tol. Solve least squares problem Y_k * z ~= b.
        check = (np.linalg.norm(bcheck - Y.dot(z)) / np.linalg.norm(bcheck))
        k += 1

    print(f'k: {k}')
    print(f'check: {check}')
    # print(f'~z: {z}')
    x = X @ z
    # print(f'~x: {x}')
    print(f'my method least squares: {np.linalg.norm(b - A.dot(x)) / np.linalg.norm(b)}')
    xc, residuals, rank, s = np.linalg.lstsq(A, b)
    print(f'numpy method: {np.linalg.norm(b - A.dot(xc)) / np.linalg.norm(b)}')

    return z
# print(f'xc {xc}')

# print(f'Ax {A.dot(x)}')
# print(f'Axc {A.dot(xc)}')
# print(f'b {b}')


# a = np.array([2,1,2])
# alpha = np.linalg.norm(a)
# alpha = math.copysign(alpha, a[0])
# evec = np.zeros(3)
# evec[0] = 1
# # FIXME it is subtract but same sign add should be same
# v = a + (alpha*evec)

# c = compute_H(v)

# c = apply_H(a, v)

# print(c)

# A = np.array([[1,-1,1],[1,-.5,.25],[1,0,0],[1,.5,.25],[1,1,1]])
# b = np.array([1,.5,0,.5,2])
# trans = copy.deepcopy(A)

# print(f'A, b: {A} {b}')

# k=0
# vvecs = []

# for i in range(3):
#     if not i:
#         vvecs.append(find_v(A[:, i], i))
#         print(f'step 0, vvec: {vvecs}')
#         A[:, i] = apply_H(A[:, i], vvecs[0])
#         print(f'new col after h: {A[:, i]}')
#     else:
#     # print(f"vv{i}{vvecs}")
#         for v in range(len(vvecs)):
#             A[v:, i] = apply_H(A[v:, i], vvecs[v])
#         print(f'new col after prev transforms: {A[:, i]}')
#         vvecs.append(find_v(A[i:, i], i))
#         print(f'new vvec after prev transforms: {vvecs[i]}')
#         A[i:, i] = apply_H(A[i:, i], vvecs[i])
#         print(f'new col after new vvec: {A[i:, i]}')

#     print(f'A after full step: {A}')
#     k += 1


