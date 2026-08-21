"""The linear-classifier fairness experiment.

This is the project's core question, made runnable. The setup:

1. Place ``num_classes`` cluster centres evenly on a circle — a 2D simplex ETF.
2. Deliberately break that symmetry with per-class ``scalars`` (push a class
   further out) and ``rotations`` (twist a class off its vertex).
3. Draw Gaussian samples around the distorted centres and train a linear
   classifier with MSE on one-hot targets.
4. Record per-class accuracy at every step. The headline metric is the
   max-minus-min gap: unequal geometry should show up as classes converging at
   visibly different rates.

Three interventions can then be switched on independently from the config:

``apply_projection``
    Warp the *data* back toward the ETF each step, with strength
    ``1 - mean_accuracy`` so the correction fades as the model learns.
``flags.fairness_loss``
    Penalize the between-class gap in the *objective* instead
    (see :mod:`cir.utils.losses`).
``flags.evo_weights``
    Search for a low-entropy *initialization* rather than sampling one
    (see :mod:`cir.utils.evolution`).

Training runs full-batch for ``num_training_steps`` steps, repeated over
``num_seeds`` seeds, so :meth:`run` replaces the base class's epoch loop.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import torch

from cir.experiments.base import BaseExperiment, seed_everything
from cir.models.linear_classifier import LinearClassifier
from cir.utils import geometry, plotting
from cir.utils.evolution import evolve_weights
from cir.utils.losses import FAIRNESS_LOSSES
from cir.utils.metrics import AccuracyTracker, count_samples

__all__ = ["LinearExperiment"]


class LinearExperiment(BaseExperiment):
    """Multi-seed linear classification on synthetic 2D Gaussian clusters.

    Config keys, beyond those read by
    :class:`~cir.experiments.base.BaseExperiment`:

    ==========================  =================================================
    ``input_dim``               Feature dimension; must be 2 for the plots.
    ``num_classes``             Number of clusters.
    ``samples_per_class``       Points drawn per cluster.
    ``radius``                  Radius of the undistorted ETF.
    ``scalars``                 Per-class radial scale, one per class.
    ``rotations``               Per-class rotation in degrees, one per class.
    ``train_ratio``             Fraction of the shuffled data used for training.
    ``num_training_steps``      Full-batch gradient steps per seed.
    ``num_seeds``               Independent runs to average over.
    ``apply_projection``        Warp data toward the ETF each step.
    ``projection_mode``         ``shift``, ``scale``, or ``norm``.
    ``target``                  Reference norm: ``mean``, ``median``, or ``max``.
    ``flags.fairness_loss``     ``per_class_gap``, ``soft_accuracy_gap``, or null.
    ``flags.evo_weights``       Use the evolutionary initializer.
    ``flags.plot_boundaries``   Save a decision-boundary frame per step.
    ==========================  =================================================
    """

    def __init__(self, cfg: Dict[str, Any], logger=None):
        super().__init__(cfg, logger=logger)

        self.input_dim = int(cfg.get("input_dim", 2))
        self.num_classes = int(cfg["num_classes"])
        self.samples_per_class = int(cfg["samples_per_class"])
        self.radius = float(cfg.get("radius", 10.0))
        self.train_ratio = float(cfg.get("train_ratio", 0.7))
        self.num_training_steps = int(cfg.get("num_training_steps", 50))
        self.num_seeds = int(cfg.get("num_seeds", 1))
        self.total_samples = self.num_classes * self.samples_per_class

        self.scalars = self._per_class(cfg.get("scalars"), default=1.0)
        self.rotations = self._per_class(cfg.get("rotations"), default=0.0)

        self.apply_projection = bool(cfg.get("apply_projection", False))
        self.projection_mode = str(cfg.get("projection_mode", "shift"))
        self.target = str(cfg.get("target", "mean"))

        flags = cfg.get("flags") or {}
        self.plot_boundaries = bool(flags.get("plot_boundaries", False))
        self.use_evo_weights = bool(flags.get("evo_weights", False))
        self.fairness_loss_name = flags.get("fairness_loss")
        if self.fairness_loss_name and self.fairness_loss_name not in FAIRNESS_LOSSES:
            raise ValueError(
                f"flags.fairness_loss must be one of {sorted(FAIRNESS_LOSSES)} or null, "
                f"got {self.fairness_loss_name!r}"
            )

        self.frame_dir = os.path.join(self.output_dir, "frames")
        self.means: torch.Tensor | None = None
        self.tracker: AccuracyTracker | None = None

    def _per_class(self, values, default: float) -> List[float]:
        """Normalize a per-class config list, filling in a default when absent.

        Args:
            values: A list from the config, or ``None``.
            default: Value used for every class when ``values`` is ``None``.

        Returns:
            A list of length ``num_classes``.

        Raises:
            ValueError: If ``values`` has the wrong length.
        """
        if values is None:
            return [default] * self.num_classes
        values = list(values)
        if len(values) != self.num_classes:
            raise ValueError(
                f"expected {self.num_classes} per-class values, got {len(values)}: {values}"
            )
        return [float(v) for v in values]

    # ------------------------------------------------------------------
    # Data and model
    # ------------------------------------------------------------------

    def build_class_means(self) -> torch.Tensor:
        """Build the ETF centres, then distort them per the config.

        Returns:
            Class means of shape ``(num_classes, 2)``.
        """
        means = geometry.make_evenly_spaced_targets(self.num_classes, self.radius)
        means = means * torch.tensor(self.scalars, dtype=means.dtype).unsqueeze(1)
        if any(self.rotations):
            means = geometry.rotate_classes(means, self.rotations)
        return means

    def build_model(self) -> LinearClassifier:
        """Construct a fresh classifier, optionally with evolved weights.

        The evolutionary search needs the class means, so :meth:`run` sets
        :attr:`means` before the first call.

        Returns:
            The new model.
        """
        model = LinearClassifier(self.input_dim, self.num_classes)
        if self.use_evo_weights:
            evo = self.cfg.get("evo") or {}
            model.linear.weight.data = evolve_weights(
                means=self.means,
                num_classes=self.num_classes,
                input_dim=self.input_dim,
                num_iter=int(evo.get("num_iter", 1)),
                pop_size=int(evo.get("pop_size", 1000)),
                tournament_size=int(evo.get("tournament_size", 100)),
                seed=int(evo.get("seed", 0)),
            )
        return model

    def get_dataloaders(self) -> Tuple[Any, Any]:
        """Generate the dataset and split it into train and test halves.

        The data is full-batch — there is no minibatching — so these are plain
        tensor tuples rather than ``DataLoader`` objects.

        Returns:
            ``((X_train, Y_train), (X_test, Y_test))``.
        """
        covs = [torch.eye(2) for _ in range(self.num_classes)]
        X = geometry.generate_samples(self.means, covs, self.num_classes, self.samples_per_class)
        classes = list(torch.eye(self.num_classes))
        Y = geometry.create_labels(self.num_classes, self.samples_per_class, classes)

        plotting.plot_samples(X, self.num_classes, self.samples_per_class, self.output_dir)

        # Shuffle jointly so labels stay attached to their samples.
        perm = torch.randperm(X.size(0))
        X, Y = X[perm].to(self.device), Y[perm].to(self.device)

        split_idx = int(self.train_ratio * self.total_samples)
        return (X[:split_idx], Y[:split_idx]), (X[split_idx:], Y[split_idx:])

    def compute_loss(self, batch: Any, decay: float = 0.0) -> torch.Tensor:
        """Loss for one full-batch step.

        Args:
            batch: An ``(X, Y)`` pair.
            decay: Weight on the fairness penalty when one is configured. The
                caller passes ``1 - mean_accuracy`` so the penalty fades out as
                the classifier converges.

        Returns:
            The scalar loss.
        """
        x, y = batch
        predictions = self.model(x)
        if self.fairness_loss_name:
            total, _, _ = FAIRNESS_LOSSES[self.fairness_loss_name](predictions, y, decay)
            return total
        return self.loss_function(predictions, y)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Train ``num_seeds`` classifiers and record per-class accuracy.

        Overrides the base epoch loop: this experiment takes full-batch steps
        and repeats over seeds rather than iterating minibatch epochs.

        Returns:
            A dict with ``mean_accuracy`` (per split, per class) and ``mean_gap``
            (per split), plus ``final_gap`` for a quick pass/fail read.
        """
        self.means = self.build_class_means()
        print(f"Class means:\n{self.means}")
        print(f"Regular polygon (ETF): {geometry.is_regular_polygon(self.means)}")

        (X_train, Y_train), (X_test, Y_test) = self.get_dataloaders()
        train_counts = count_samples(Y_train, self.num_classes)
        test_counts = count_samples(Y_test, self.num_classes)

        # Precompute the ETF correction once; it depends only on the means.
        transform = geometry.transform_to_even_space(
            self.means, mode=self.projection_mode, ref_mode=self.target
        ).to(self.device)
        if self.apply_projection:
            print(f"Projection ({self.projection_mode}, ref={self.target}):\n{transform}")

        self.tracker = AccuracyTracker(self.num_classes, self.num_training_steps, self.num_seeds)
        clean_train, clean_test = X_train.clone(), X_test.clone()

        for seed in range(self.num_seeds):
            seed_everything(seed)
            self.setup()
            mean_accuracy = 0.0

            for step in range(self.num_training_steps):
                # Projection strength fades as the classifier improves.
                decay = 1.0 - mean_accuracy

                if self.apply_projection:
                    x_train, _ = geometry.apply_projection(
                        clean_train, Y_train, self.means, transform, self.projection_mode, decay
                    )
                    x_test, _ = geometry.apply_projection(
                        clean_test, Y_test, self.means, transform, self.projection_mode, decay
                    )
                else:
                    x_train, x_test = clean_train, clean_test

                predictions = self.model(x_train)
                loss = self.compute_loss((x_train, Y_train), decay)
                self.optimizer.zero_grad()
                loss.backward()

                if self.plot_boundaries:
                    plotting.plot_decision_boundaries(
                        x_train.cpu(),
                        Y_train.cpu(),
                        self.num_classes,
                        self.model.linear.weight.data.cpu(),
                        self.model.linear.bias.data.cpu(),
                        step,
                        seed,
                        self.frame_dir,
                    )

                mean_accuracy = self.tracker.update(
                    predictions.detach(), Y_train, train_counts, "train", seed, step
                )
                self.optimizer.step()

                with torch.no_grad():
                    self.tracker.update(
                        self.model(x_test), Y_test, test_counts, "test", seed, step
                    )

                self.logger.log(
                    seed=seed,
                    step=step,
                    loss=loss.item(),
                    train_accuracy=mean_accuracy,
                    train_gap=self.tracker.gap("train", seed, step),
                    test_gap=self.tracker.gap("test", seed, step),
                )

            print(
                f"Seed {seed + 1}/{self.num_seeds} | loss {loss.item():.4f} | "
                f"train acc {mean_accuracy:.4f} | "
                f"train gap {self.tracker.gap('train', seed, self.num_training_steps - 1):.4f}"
            )

        self.on_run_end()
        return {
            "mean_accuracy": {s: self.tracker.mean_per_class(s) for s in ("train", "test")},
            "mean_gap": {s: self.tracker.mean_gap(s) for s in ("train", "test")},
            "final_gap": {s: self.tracker.mean_gap(s)[-1] for s in ("train", "test")},
        }

    def on_run_end(self) -> None:
        """Write the accuracy, gap, and (optionally) animation figures."""
        plotting.plot_avg_accuracy(
            self.tracker.mean_per_class("train"),
            self.tracker.mean_per_class("test"),
            self.output_dir,
        )
        plotting.plot_accuracy_gap(
            self.tracker.gap_per_seed("train"),
            self.tracker.gap_per_seed("test"),
            self.tracker.mean_gap("train"),
            self.tracker.mean_gap("test"),
            self.output_dir,
        )
        if self.plot_boundaries:
            plotting.make_animation(
                self.num_seeds,
                self.num_training_steps,
                self.frame_dir,
                os.path.join(self.output_dir, "animations"),
            )
        print(f"Figures written to {self.output_dir}")
