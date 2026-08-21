"""Tests for cluster geometry and the ETF projections."""

import math

import pytest
import torch

from cir.utils import geometry


@pytest.mark.parametrize("num_classes", [3, 4, 5, 8])
def test_evenly_spaced_targets_form_a_regular_polygon(num_classes):
    means = geometry.make_evenly_spaced_targets(num_classes, radius=10.0)
    assert means.shape == (num_classes, 2)
    assert geometry.is_regular_polygon(means)
    # Every vertex sits on the requested circle.
    assert torch.allclose(torch.linalg.norm(means, dim=1), torch.full((num_classes,), 10.0), atol=1e-4)


def test_evenly_spaced_targets_are_deterministic():
    a = geometry.make_evenly_spaced_targets(5, 3.0)
    b = geometry.make_evenly_spaced_targets(5, 3.0)
    assert torch.equal(a, b)


def test_is_regular_polygon_rejects_a_distorted_configuration():
    means = geometry.make_evenly_spaced_targets(3, 10.0)
    means[1] *= 2.0
    assert not geometry.is_regular_polygon(means)


def test_is_regular_polygon_needs_three_points():
    with pytest.raises(ValueError, match="at least 3 points"):
        geometry.is_regular_polygon(torch.zeros(2, 2))


def test_shift_transform_lands_exactly_on_the_etf():
    means = geometry.make_evenly_spaced_targets(4, 10.0)
    means[0] *= 3.0
    shifts = geometry.transform_to_even_space(means, mode="shift", ref_mode="median")
    assert geometry.is_regular_polygon(means + shifts)


def test_scale_transform_equalizes_norms():
    means = geometry.make_evenly_spaced_targets(4, 10.0)
    means[0] *= 3.0
    scalars = geometry.transform_to_even_space(means, mode="scale", ref_mode="max")
    norms = torch.linalg.norm(means * scalars[:, None], dim=1)
    assert torch.allclose(norms, norms[0].expand_as(norms), atol=1e-3)


@pytest.mark.parametrize("bad", [{"mode": "nope"}, {"ref_mode": "nope"}])
def test_transform_rejects_unknown_modes(bad):
    means = geometry.make_evenly_spaced_targets(3, 1.0)
    with pytest.raises(ValueError):
        geometry.transform_to_even_space(means, **bad)


def test_rotation_preserves_norms_and_full_turn_is_identity():
    means = geometry.make_evenly_spaced_targets(3, 10.0)
    rotated = geometry.rotate_classes(means, [30.0, 0.0, -45.0])
    assert torch.allclose(
        torch.linalg.norm(rotated, dim=1), torch.linalg.norm(means, dim=1), atol=1e-4
    )
    assert torch.allclose(geometry.rotate_classes(means, [360.0] * 3), means, atol=1e-4)


def test_rotation_length_must_match_class_count():
    with pytest.raises(ValueError, match="rotations has"):
        geometry.rotate_classes(geometry.make_evenly_spaced_targets(3, 1.0), [0.0])


def test_generated_samples_cluster_around_their_means():
    torch.manual_seed(0)
    means = geometry.make_evenly_spaced_targets(3, 10.0)
    covs = [torch.eye(2) for _ in range(3)]
    X = geometry.generate_samples(means, covs, 3, 500)

    assert X.shape == (1500, 2)
    for i in range(3):
        assert torch.allclose(X[i * 500 : (i + 1) * 500].mean(dim=0), means[i], atol=0.3)


def test_labels_align_with_the_sample_layout():
    classes = list(torch.eye(3))
    Y = geometry.create_labels(3, 4, classes)
    assert Y.shape == (12, 3)
    assert torch.equal(Y.argmax(dim=1), torch.tensor([0] * 4 + [1] * 4 + [2] * 4))


def test_projection_decay_interpolates_between_identity_and_full_effect():
    means = geometry.make_evenly_spaced_targets(3, 10.0)
    means[1] *= 2.0
    covs = [torch.eye(2) * 0.01 for _ in range(3)]
    X = geometry.generate_samples(means, covs, 3, 20)
    Y = geometry.create_labels(3, 20, list(torch.eye(3)))
    shifts = geometry.transform_to_even_space(means, mode="shift", ref_mode="median")

    unchanged, _ = geometry.apply_projection(X, Y, means, shifts, "shift", decay=0.0)
    assert torch.allclose(unchanged, X, atol=1e-5)

    full, projected_means = geometry.apply_projection(X, Y, means, shifts, "shift", decay=1.0)
    assert not torch.allclose(full, X)
    assert geometry.is_regular_polygon(projected_means)


def test_apply_projection_rejects_unknown_mode():
    means = geometry.make_evenly_spaced_targets(3, 1.0)
    X = torch.zeros(3, 2)
    Y = torch.eye(3)
    with pytest.raises(ValueError, match="mode must be"):
        geometry.apply_projection(X, Y, means, torch.ones(3), "bogus", 1.0)
