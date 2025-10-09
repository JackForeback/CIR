import math
import numpy as np

def compute_H(vector):
    v = vector.reshape(-1, 1)  # column vector
    return np.eye(len(vector)) - 2 * ((v @ v.T) / (v.T @ v))

def apply_H(u, v):
    # print(f'u,v: {u, v}, v@u: {v@u}, v@v: {v@v}')
    scalar = 2 * ((v @ u) / (v @ v))
    # print(f'scalar: {scalar}')
    # print(f'v: {v}')
    # print(f'scalarv: {(scalar * v)}')
    u = u - (scalar * v)
    # print(f'newu{u}')
    return u

def find_v(col, k):
    # two norm of col, same sign for e_i
    alpha = np.linalg.norm(col)
    alpha = math.copysign(alpha, col[0])
    evec = np.zeros(len(col))
    evec[0] = 1
    return col + alpha*evec
