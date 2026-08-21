"""Per-class accuracy tracking.

The quantity this project cares about is not overall accuracy but the *spread*
between classes: whether every class converges at the same rate. :class:`AccuracyTracker`
records per-class accuracy for every (split, seed, step) and exposes both the
seed-averaged curves and the per-seed max-minus-min gap that the plots use.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

__all__ = ["count_samples", "AccuracyTracker"]


def count_samples(labels: torch.Tensor, num_classes: int) -> List[int]:
    """Count how many rows of ``labels`` belong to each class.

    Args:
        labels: One-hot labels, shape ``(N, C)``.
        num_classes: Number of classes ``C``.

    Returns:
        A list of length ``num_classes`` with the count for each class.
    """
    counts = torch.bincount(labels.argmax(dim=1), minlength=num_classes)
    return counts.tolist()


class AccuracyTracker:
    """Collect per-class accuracy across seeds and training steps.

    Args:
        num_classes: Number of classes.
        num_steps: Training steps per seed.
        num_seeds: Number of seeds (independent model runs).

    Attributes:
        history: ``history[split][seed][step]`` is a list of per-class
            accuracies, one entry per class.
    """

    SPLITS = ("train", "test")

    def __init__(self, num_classes: int, num_steps: int, num_seeds: int):
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.num_seeds = num_seeds
        self.history: Dict[str, List[List[List[float]]]] = {
            split: [[[] for _ in range(num_steps)] for _ in range(num_seeds)]
            for split in self.SPLITS
        }

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        class_counts: Sequence[int],
        split: str,
        seed: int,
        step: int,
    ) -> float:
        """Record accuracy for one evaluation and return the macro average.

        Args:
            predictions: Raw model outputs, shape ``(N, C)``.
            targets: One-hot labels, shape ``(N, C)``.
            class_counts: Total samples of each class in this split, used as the
                denominator so a class missing from a batch does not inflate the
                score.
            split: ``"train"`` or ``"test"``.
            seed: Index of the current seed.
            step: Index of the current training step.

        Returns:
            The macro-averaged accuracy over classes, in ``[0, 1]``. The
            experiments feed this back as the projection decay term.
        """
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")

        predicted = predictions.argmax(dim=1)
        actual = targets.argmax(dim=1)
        correct = torch.bincount(
            predicted[predicted == actual], minlength=self.num_classes
        ).tolist()

        accuracies = [
            correct[k] / class_counts[k] if class_counts[k] else 0.0
            for k in range(self.num_classes)
        ]
        self.history[split][seed][step] = accuracies
        return sum(accuracies) / self.num_classes

    def mean_per_class(self, split: str) -> Dict[int, List[float]]:
        """Average each class's accuracy curve over seeds.

        Args:
            split: ``"train"`` or ``"test"``.

        Returns:
            ``{class_id: [accuracy at each step]}``.
        """
        runs = self.history[split]
        return {
            k: [
                sum(runs[s][t][k] for s in range(self.num_seeds)) / self.num_seeds
                for t in range(self.num_steps)
            ]
            for k in range(self.num_classes)
        }

    def gap(self, split: str, seed: int, step: int) -> float:
        """Max-minus-min class accuracy at one recorded point.

        Args:
            split: ``"train"`` or ``"test"``.
            seed: Seed index.
            step: Step index.

        Returns:
            The gap, or ``0.0`` if that point has not been recorded yet — which
            lets an experiment read back the gap mid-run.
        """
        accuracies = self.history[split][seed][step]
        return max(accuracies) - min(accuracies) if accuracies else 0.0

    def gap_per_seed(self, split: str) -> List[List[float]]:
        """Max-minus-min class accuracy at each step, for each seed.

        This is the project's headline fairness metric: a gap of zero means every
        class is being learned at the same rate.

        Args:
            split: ``"train"`` or ``"test"``.

        Returns:
            ``gaps[seed][step]``.
        """
        return [
            [self.gap(split, seed, step) for step in range(self.num_steps)]
            for seed in range(self.num_seeds)
        ]

    def mean_gap(self, split: str) -> List[float]:
        """Seed-averaged version of :meth:`gap_per_seed`.

        Args:
            split: ``"train"`` or ``"test"``.

        Returns:
            One averaged gap per training step.
        """
        gaps = self.gap_per_seed(split)
        return [
            sum(gaps[s][t] for s in range(self.num_seeds)) / self.num_seeds
            for t in range(self.num_steps)
        ]
