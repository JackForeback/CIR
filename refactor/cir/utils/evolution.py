"""Evolutionary search for a linear classifier's initial weights.

An alternative to random initialization: instead of accepting whatever weights
`nn.Linear` hands out, search for a weight matrix that already separates the
class means confidently. Fitness is the mean Shannon entropy of the softmax over
``W @ mean`` across all class means — low entropy means the initial decision
boundaries already commit to one class per cluster, so no class starts at a
disadvantage.

Selection is a standard tournament: the population is split into groups of
``tournament_size``, the lowest-entropy member of each group survives, and the
survivors are cloned and mutated to refill the population.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = ["population_entropy", "rank_population", "mutate", "evolve_weights"]

_EPS = 1e-12


def population_entropy(candidate: torch.Tensor, means: torch.Tensor) -> float:
    """Mean softmax entropy of one candidate evaluated at every class mean.

    Args:
        candidate: Weight matrix, shape ``(num_classes, input_dim)``.
        means: Class means, shape ``(num_classes, input_dim)``.

    Returns:
        Average Shannon entropy in nats. Lower is a more confident candidate.
    """
    logits = means @ candidate.T                     # (num_classes, num_classes)
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * (probs + _EPS).log()).sum(dim=1)
    return float(entropy.mean())


def rank_population(
    population: Sequence[torch.Tensor], means: torch.Tensor
) -> Tuple[List[torch.Tensor], List[float]]:
    """Sort a population by fitness, best (lowest entropy) first.

    Args:
        population: Candidate weight matrices.
        means: Class means, shape ``(num_classes, input_dim)``.

    Returns:
        ``(sorted_population, sorted_entropies)``.
    """
    scored = sorted(
        ((population_entropy(c, means), i) for i, c in enumerate(population)),
        key=lambda pair: pair[0],
    )
    return [population[i] for _, i in scored], [score for score, _ in scored]


def mutate(
    weights: torch.Tensor,
    rate: float = 1.0 / 6.0,
    scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Add Gaussian noise to a random subset of entries.

    Args:
        weights: Weight matrix to mutate; not modified in place.
        rate: Probability that any individual entry is perturbed.
        scale: Standard deviation of the perturbation.
        generator: Optional RNG for reproducibility.

    Returns:
        A new, mutated weight matrix of the same shape.
    """
    mask = torch.rand(weights.shape, generator=generator) < rate
    noise = torch.randn(weights.shape, generator=generator) * scale
    return weights + mask * noise


def evolve_weights(
    means: torch.Tensor,
    num_classes: int,
    input_dim: int,
    num_iter: int = 1,
    pop_size: int = 1000,
    tournament_size: int = 100,
    mutation_rate: float = 1.0 / 6.0,
    seed: int = 0,
) -> torch.Tensor:
    """Search for a low-entropy initial weight matrix.

    Args:
        means: Class means, shape ``(num_classes, input_dim)``.
        num_classes: Rows of the weight matrix.
        input_dim: Columns of the weight matrix.
        num_iter: Number of generations.
        pop_size: Candidates per generation. Must be divisible by
            ``tournament_size``.
        tournament_size: Candidates per tournament group.
        mutation_rate: Per-entry mutation probability.
        seed: Seed for the dedicated RNG, so the search is reproducible and does
            not disturb global torch RNG state.

    Returns:
        The best weight matrix found, shape ``(num_classes, input_dim)``.

    Raises:
        ValueError: If ``pop_size`` is not a positive multiple of
            ``tournament_size``.
    """
    if tournament_size <= 0 or pop_size <= 0 or pop_size % tournament_size:
        raise ValueError(
            f"pop_size ({pop_size}) must be a positive multiple of "
            f"tournament_size ({tournament_size})"
        )

    generator = torch.Generator().manual_seed(seed)
    num_groups = pop_size // tournament_size

    # Generation 0: uniformly random candidates in the same range nn.Linear uses.
    bound = 1.0 / (input_dim ** 0.5)
    population = [
        (torch.rand((num_classes, input_dim), generator=generator) * 2 - 1) * bound
        for _ in range(pop_size)
    ]

    for _ in range(num_iter):
        # Tournament selection: best of each contiguous group survives.
        winners: List[torch.Tensor] = []
        for g in range(num_groups):
            group = population[g * tournament_size : (g + 1) * tournament_size]
            ranked, _ = rank_population(group, means)
            winners.append(ranked[0])

        # Repopulate: keep the winners, fill the rest with their mutated clones.
        population = list(winners)
        while len(population) < pop_size:
            parent = winners[len(population) % num_groups]
            population.append(mutate(parent, mutation_rate, generator=generator))

    ranked, _ = rank_population(population, means)
    return ranked[0].clone()
