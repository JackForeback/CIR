"""Tests for the fairness losses, accuracy tracking, and evolutionary init."""

import pytest
import torch

from cir.utils.evolution import evolve_weights, mutate, population_entropy, rank_population
from cir.utils.losses import loss_with_per_class_gap, loss_with_soft_accuracy_gap
from cir.utils.metrics import AccuracyTracker, count_samples


# --- losses ------------------------------------------------------------------

@pytest.mark.parametrize("loss_fn", [loss_with_per_class_gap, loss_with_soft_accuracy_gap])
def test_fairness_penalty_vanishes_when_classes_are_treated_equally(loss_fn):
    target = torch.eye(3).repeat(4, 1)
    # Every class gets the same margin, so there is no gap to penalize.
    pred = target * 3.0
    total, mse, fairness = loss_fn(pred, target)
    assert torch.isclose(fairness, torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(total, mse, atol=1e-5)


@pytest.mark.parametrize("loss_fn", [loss_with_per_class_gap, loss_with_soft_accuracy_gap])
def test_fairness_penalty_is_positive_when_one_class_lags(loss_fn):
    target = torch.eye(3).repeat(4, 1)
    pred = target * 3.0
    pred[target[:, 2] == 1] = torch.tensor([1.0, 1.0, 0.0])  # class 2 fits badly
    _, _, fairness = loss_fn(pred, target)
    assert fairness > 0


@pytest.mark.parametrize("loss_fn", [loss_with_per_class_gap, loss_with_soft_accuracy_gap])
def test_lambda_scales_the_penalty_and_the_loss_is_differentiable(loss_fn):
    target = torch.eye(3).repeat(4, 1)
    pred = (target * 3.0).clone().requires_grad_(True)
    pred_detached = pred.detach()
    pred_detached[target[:, 2] == 1] = torch.tensor([1.0, 1.0, 0.0])

    weak, mse, fairness = loss_fn(pred, target, lambda_fair=0.1)
    strong, _, _ = loss_fn(pred, target, lambda_fair=1.0)
    assert torch.isclose(strong - weak, 0.9 * fairness, atol=1e-5)

    weak.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_per_class_gap_tolerates_a_class_missing_from_the_batch():
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # no class 2
    total, _, fairness = loss_with_per_class_gap(target * 2, target)
    assert torch.isfinite(total) and torch.isfinite(fairness)


# --- metrics -----------------------------------------------------------------

def test_count_samples_counts_each_class():
    labels = torch.eye(3)[torch.tensor([0, 0, 1, 2, 2, 2])]
    assert count_samples(labels, 3) == [2, 1, 3]


def test_count_samples_reports_zero_for_absent_classes():
    assert count_samples(torch.eye(4)[torch.tensor([0, 0])], 4) == [2, 0, 0, 0]


def test_tracker_records_perfect_and_imperfect_predictions():
    tracker = AccuracyTracker(num_classes=2, num_steps=1, num_seeds=1)
    targets = torch.eye(2).repeat(2, 1)  # two of each class

    perfect = tracker.update(targets * 5, targets, [2, 2], "train", seed=0, step=0)
    assert perfect == 1.0
    assert tracker.gap_per_seed("train")[0][0] == 0.0

    # Class 1 is always predicted as class 0: class 0 perfect, class 1 at zero.
    wrong = torch.tensor([[5.0, 0.0]]).repeat(4, 1)
    macro = tracker.update(wrong, targets, [2, 2], "train", seed=0, step=0)
    assert macro == pytest.approx(0.5)
    assert tracker.gap_per_seed("train")[0][0] == pytest.approx(1.0)


def test_tracker_averages_over_seeds():
    tracker = AccuracyTracker(num_classes=2, num_steps=1, num_seeds=2)
    targets = torch.eye(2).repeat(2, 1)
    tracker.update(targets * 5, targets, [2, 2], "test", seed=0, step=0)
    tracker.update(torch.tensor([[5.0, 0.0]]).repeat(4, 1), targets, [2, 2], "test", seed=1, step=0)

    assert tracker.mean_per_class("test")[0] == [1.0]      # right under both seeds
    assert tracker.mean_per_class("test")[1] == [0.5]      # right under one
    assert tracker.mean_gap("test") == [pytest.approx(0.5)]


def test_tracker_rejects_an_unknown_split():
    tracker = AccuracyTracker(2, 1, 1)
    with pytest.raises(ValueError, match="split must be"):
        tracker.update(torch.eye(2), torch.eye(2), [1, 1], "validation", 0, 0)


# --- evolution ---------------------------------------------------------------

def test_confident_weights_score_lower_entropy_than_indifferent_ones():
    means = torch.eye(3) * 10.0
    confident = torch.eye(3) * 10.0     # each mean maps to its own class, strongly
    indifferent = torch.zeros(3, 3)     # uniform softmax, maximal entropy
    assert population_entropy(confident, means) < population_entropy(indifferent, means)


def test_rank_population_orders_best_first():
    means = torch.eye(3) * 10.0
    population = [torch.zeros(3, 3), torch.eye(3) * 10.0]
    ranked, scores = rank_population(population, means)
    assert torch.equal(ranked[0], population[1])
    assert scores[0] <= scores[1]


def test_mutate_preserves_shape_leaves_the_input_alone_and_is_reproducible():
    weights = torch.zeros(3, 2)
    a = mutate(weights, rate=1.0, generator=torch.Generator().manual_seed(0))
    b = mutate(weights, rate=1.0, generator=torch.Generator().manual_seed(0))
    assert a.shape == weights.shape
    assert torch.equal(weights, torch.zeros(3, 2))  # not modified in place
    assert torch.equal(a, b)
    assert not torch.equal(a, weights)


def test_mutate_with_zero_rate_changes_nothing():
    weights = torch.randn(3, 2)
    assert torch.equal(mutate(weights, rate=0.0), weights)


def test_evolution_improves_on_the_starting_population():
    means = torch.tensor([[10.0, 0.0], [0.0, 10.0], [-10.0, -10.0]])
    best = evolve_weights(means, num_classes=3, input_dim=2, num_iter=3, pop_size=200,
                          tournament_size=50, seed=0)
    assert best.shape == (3, 2)

    torch.manual_seed(0)
    scores = [population_entropy(torch.randn(3, 2) * 0.7, means) for _ in range(200)]
    assert population_entropy(best, means) < sum(scores) / len(scores)


def test_evolution_is_reproducible_and_validates_its_population_size():
    means = torch.eye(3)[:, :2] * 5
    kwargs = dict(means=means, num_classes=3, input_dim=2, pop_size=100, tournament_size=50)
    assert torch.equal(evolve_weights(seed=7, **kwargs), evolve_weights(seed=7, **kwargs))

    with pytest.raises(ValueError, match="must be a positive multiple"):
        evolve_weights(means=means, num_classes=3, input_dim=2, pop_size=100, tournament_size=30)
