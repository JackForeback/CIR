"""Tests for the Householder / Chebyshev least-squares solver."""

import numpy as np
import pytest

from cir.utils.solvers import (
    apply_householder,
    chebyshev_basis_column,
    householder_matrix,
    householder_vector,
    iterative_chebyshev_ls,
)


def test_householder_zeros_all_but_the_first_entry():
    a = np.array([3.0, 4.0, 12.0])
    reflected = apply_householder(a, householder_vector(a))
    assert np.allclose(reflected[1:], 0.0, atol=1e-10)
    # A reflection is norm preserving.
    assert np.isclose(abs(reflected[0]), np.linalg.norm(a))


def test_householder_is_a_no_op_on_a_zero_vector():
    a = np.zeros(4)
    assert np.allclose(apply_householder(a, householder_vector(a)), a)


def test_householder_matrix_is_symmetric_and_involutory():
    H = householder_matrix(np.array([1.0, -2.0, 0.5]))
    assert np.allclose(H, H.T)
    assert np.allclose(H @ H, np.eye(3), atol=1e-10)


def test_matrix_form_matches_the_matrix_free_application():
    v = householder_vector(np.array([2.0, 1.0, 2.0]))
    y = np.array([1.0, -3.0, 0.5])
    assert np.allclose(householder_matrix(v) @ y, apply_householder(y, v))


@pytest.mark.parametrize("k, expected", [(0, 1.0), (1, 0.5), (2, -0.5)])
def test_chebyshev_columns_match_known_values(k, expected):
    # T_0(0.5)=1, T_1(0.5)=0.5, T_2(0.5)=2(0.25)-1=-0.5
    assert np.isclose(chebyshev_basis_column(k, np.array([0.5]))[0], expected)


def test_solver_recovers_a_smooth_signal():
    # A smooth right-hand side is exactly what a low-order Chebyshev basis fits.
    n = 40
    A = np.eye(n)
    b = np.sin(np.linspace(-1.0, 1.0, n) * np.pi)

    x, diagnostics = iterative_chebyshev_ls(A, b, max_iter=20, tol=1e-8)

    residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    assert residual < 1e-6
    assert np.isclose(residual, diagnostics["residual"], atol=1e-9)
    assert 0 < diagnostics["k"] <= 20


def test_solver_is_optimal_within_the_basis_it_has_built():
    """The iterative QR must match a direct least-squares fit over the same basis.

    This is the real correctness property: the solver restricts x to the span of
    k Chebyshev polynomials, so it cannot beat an unrestricted solve — but within
    that span it must be exactly as good as one.
    """
    rng = np.random.default_rng(0)
    n = 40
    A = np.eye(n) + 0.01 * rng.standard_normal((n, n))
    b = np.sin(np.linspace(-1.0, 1.0, n) * np.pi)

    x, diagnostics = iterative_chebyshev_ls(A, b, max_iter=12, tol=0.0)

    # Best possible coefficients over the very same basis columns.
    best_z, *_ = np.linalg.lstsq(A @ diagnostics["X"], b, rcond=None)
    assert np.allclose(x, diagnostics["X"] @ best_z, atol=1e-8)


def test_more_basis_columns_never_hurt():
    rng = np.random.default_rng(2)
    n = 40
    A = np.eye(n) + 0.01 * rng.standard_normal((n, n))
    b = np.cos(np.linspace(-1.0, 1.0, n) * 2 * np.pi)

    _, few = iterative_chebyshev_ls(A, b, max_iter=4, tol=0.0)
    _, many = iterative_chebyshev_ls(A, b, max_iter=16, tol=0.0)
    assert many["residual"] <= few["residual"] + 1e-12


def test_solver_stops_early_once_tolerance_is_met():
    n = 30
    nodes = np.linspace(-1.0, 1.0, n)
    # b is exactly A @ T_0, so one basis column suffices.
    A = np.eye(n)
    b = A @ chebyshev_basis_column(0, nodes)

    _, diagnostics = iterative_chebyshev_ls(A, b, max_iter=20, tol=1e-8)
    assert diagnostics["k"] == 1


def test_solver_honours_its_iteration_cap():
    rng = np.random.default_rng(1)
    A, b = rng.standard_normal((25, 25)), rng.standard_normal(25)
    _, diagnostics = iterative_chebyshev_ls(A, b, max_iter=4, tol=0.0)
    assert diagnostics["k"] == 4


def test_solver_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="rows"):
        iterative_chebyshev_ls(np.eye(5), np.ones(4))
