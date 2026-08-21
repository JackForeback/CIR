"""Tests for the fixed-basis linear layers."""

import pytest
import torch

from cir.models.basis import BasisLinear, ChebyshevLinear, DCTLinear, chebyshev_basis, dct_basis


def test_dct_basis_rows_are_orthonormal():
    N = 32
    W = dct_basis(N, N, BasisLinear.default_points(N, "dct"))
    assert torch.allclose(W @ W.T, torch.eye(N), atol=1e-5)


def test_chebyshev_basis_matches_the_closed_form():
    points = torch.linspace(-1.0, 1.0, 16)
    W = chebyshev_basis(5, 16, points)
    for k in range(5):
        assert torch.allclose(W[k], torch.cos(k * torch.acos(points)), atol=1e-5)


@pytest.mark.parametrize("layer_cls", [DCTLinear, ChebyshevLinear])
def test_layers_produce_the_expected_shape(layer_cls):
    layer = layer_cls(64, 16)
    assert layer(torch.randn(8, 64)).shape == (8, 16)


def test_weights_are_buffers_and_never_learn():
    layer = DCTLinear(32, 8)
    assert "weight" in dict(layer.named_buffers())
    assert not any(p.requires_grad and p is layer.weight for p in layer.parameters())

    # Gradients flow through to the input, but the basis itself is untouched.
    before = layer.weight.clone()
    x = torch.randn(4, 32, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None
    assert layer.weight.grad is None
    assert torch.equal(layer.weight, before)


def test_a_full_dct_projection_is_lossless():
    # With K == N the orthonormal basis spans the whole space, so the round trip
    # x -> coefficients -> x is the identity. This is the property ALVAE's
    # auxiliary loss relies on.
    layer = DCTLinear(24, 24)
    x = torch.randn(5, 24)
    assert torch.allclose(layer(x) @ layer.weight, x, atol=1e-4)


def test_truncated_projection_loses_energy():
    layer = DCTLinear(24, 4)
    x = torch.randn(5, 24)
    assert not torch.allclose(layer(x) @ layer.weight, x, atol=1e-3)


def test_orthonormalized_chebyshev_rows_are_orthonormal():
    layer = ChebyshevLinear(40, 8, orthonormalize=True)
    assert torch.allclose(layer.weight @ layer.weight.T, torch.eye(8), atol=1e-4)


def test_optional_bias_is_learnable():
    layer = BasisLinear(16, 4, basis="dct", bias=True)
    assert layer.bias is not None and layer.bias.requires_grad
    assert BasisLinear(16, 4, basis="dct").bias is None


def test_custom_callable_basis_is_accepted():
    layer = BasisLinear(10, 3, basis=lambda k, n, pts: torch.ones(k, n))
    assert torch.allclose(layer(torch.ones(1, 10)), torch.full((1, 3), 10.0))


def test_rejects_unknown_basis_wrong_points_and_wrong_input_width():
    with pytest.raises(ValueError, match="basis must be"):
        BasisLinear(8, 2, basis="fourier")
    with pytest.raises(ValueError, match="points must have length"):
        BasisLinear(8, 2, points=torch.zeros(3))
    with pytest.raises(ValueError, match="expected last dim"):
        DCTLinear(8, 2)(torch.randn(2, 7))
