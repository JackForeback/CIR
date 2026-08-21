"""Figures for the linear-classifier experiment.

Every function takes an explicit output directory. The pre-refactor version read
a global ``path`` from ``sys.argv[2]`` at import time, which made these functions
unimportable outside the one script that happened to be invoked correctly.

Matplotlib is forced onto the ``Agg`` backend so runs work headless, on a compute
node or in CI, without a display.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
import torch  # noqa: E402
from PIL import Image  # noqa: E402

__all__ = [
    "class_colormap",
    "format_plot",
    "plot_samples",
    "plot_decision_boundaries",
    "plot_avg_accuracy",
    "plot_accuracy_gap",
    "make_animation",
]


def class_colormap(num_classes: int):
    """Return a colormap with one distinguishable colour per class.

    Args:
        num_classes: Number of classes.

    Returns:
        A matplotlib colormap callable as ``cmap(class_id)``.
    """
    return plt.get_cmap("tab10" if num_classes <= 10 else "nipy_spectral", num_classes)


def format_plot(
    save_path: str,
    title: str = "",
    xlabel: str = "X",
    ylabel: str = "Y",
    legend: bool = True,
) -> None:
    """Apply shared styling, save the current figure, and close it.

    Args:
        save_path: Destination file. Parent directories are created as needed.
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        legend: Whether to draw the legend. Turned off for the decision-boundary
            frames, where the per-boundary entries overwhelm the figure.
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_samples(
    data: torch.Tensor, num_classes: int, samples_per_class: int, out_dir: str
) -> str:
    """Scatter the generated 2D clusters, coloured by class.

    Args:
        data: Samples grouped by class, shape ``(num_classes * samples_per_class, 2)``.
        num_classes: Number of clusters.
        samples_per_class: Points per cluster.
        out_dir: Directory to write ``sample_plot.png`` into.

    Returns:
        The path written.
    """
    cmap = class_colormap(num_classes)
    plt.figure(figsize=(10, 6))

    for class_id in range(num_classes):
        start = class_id * samples_per_class
        chunk = data[start : start + samples_per_class]
        plt.scatter(chunk[:, 0], chunk[:, 1], s=4, color=cmap(class_id), label=f"Class {class_id}")

    path = os.path.join(out_dir, "sample_plot.png")
    format_plot(path, "Generated 2D Gaussian Samples")
    return path


def plot_decision_boundaries(
    data: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    weights: torch.Tensor,
    biases: torch.Tensor,
    step: int,
    seed: int,
    out_dir: str,
    max_points: int = 2000,
) -> str:
    """Draw one animation frame: the samples plus every pairwise boundary.

    The boundary between classes *i* and *j* is where their scores are equal,
    i.e. ``(w_i - w_j) · x + (b_i - b_j) = 0``.

    Args:
        data: Samples, shape ``(N, 2)``.
        targets: One-hot labels, shape ``(N, C)``.
        num_classes: Number of classes.
        weights: Classifier weights, shape ``(C, 2)``.
        biases: Classifier biases, shape ``(C,)``.
        step: Training step, used in the filename.
        seed: Seed index, used in the filename.
        out_dir: Directory that receives ``{seed}-{step}.png``.
        max_points: Subsample cap per class, to keep frames fast to render.

    Returns:
        The path written.
    """
    cmap = class_colormap(num_classes)
    labels = targets.argmax(dim=1)
    plt.figure(figsize=(10, 6))

    for class_id in range(num_classes):
        chunk = data[labels == class_id][:max_points]
        if len(chunk):
            plt.scatter(chunk[:, 0], chunk[:, 1], s=4, color=cmap(class_id), label=f"Class {class_id}")

    x_vals = torch.linspace(data[:, 0].min() - 1, data[:, 0].max() + 1, 500)
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            a, b = (weights[i] - weights[j]).tolist()
            c = (biases[i] - biases[j]).item()
            if b != 0:
                plt.plot(x_vals, -(a / b) * x_vals - (c / b), linewidth=1)
            elif a != 0:
                plt.axvline(x=-c / a, linewidth=1)
            # else: w_i == w_j, the boundary is undefined — nothing to draw.

    x_min, x_max = data[:, 0].min().item(), data[:, 0].max().item()
    y_min, y_max = data[:, 1].min().item(), data[:, 1].max().item()
    x_pad, y_pad = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1
    plt.axis([x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad])

    path = os.path.join(out_dir, f"{seed}-{step}.png")
    format_plot(path, f"Decision Boundaries — seed {seed}, step {step}", legend=False)
    return path


def plot_avg_accuracy(
    train: Dict[int, List[float]], test: Dict[int, List[float]], out_dir: str
) -> str:
    """Plot each class's seed-averaged accuracy curve, train vs test.

    Args:
        train: ``{class_id: [accuracy per step]}`` for the training split.
        test: The same for the test split.
        out_dir: Directory to write ``avg_accuracy.png`` into.

    Returns:
        The path written.
    """
    cmap = class_colormap(max(len(train), 1))
    plt.figure(figsize=(10, 6))

    for class_id, curve in train.items():
        plt.plot(curve, linestyle="--", color=cmap(class_id), label=f"Class {class_id} (train)")
    for class_id, curve in test.items():
        plt.plot(curve, linestyle="-", color=cmap(class_id), label=f"Class {class_id} (test)")

    path = os.path.join(out_dir, "avg_accuracy.png")
    format_plot(path, "Per-Class Average Accuracy", "Training Step", "Accuracy")
    return path


def plot_accuracy_gap(
    train_gaps: Sequence[Sequence[float]],
    test_gaps: Sequence[Sequence[float]],
    train_mean: Sequence[float],
    test_mean: Sequence[float],
    out_dir: str,
) -> List[str]:
    """Plot the max-minus-min class accuracy gap, per seed and averaged.

    This is the headline fairness metric: a flat line at zero means all classes
    converge together.

    Args:
        train_gaps: ``gaps[seed][step]`` for the training split.
        test_gaps: The same for the test split.
        train_mean: Seed-averaged training gap per step.
        test_mean: Seed-averaged test gap per step.
        out_dir: Directory receiving ``seed/`` figures and ``avg_gap.png``.

    Returns:
        The paths written, per-seed figures first.
    """
    written = []
    seed_dir = os.path.join(out_dir, "seed")

    for seed, (tr, te) in enumerate(zip(train_gaps, test_gaps)):
        plt.figure(figsize=(10, 6))
        plt.plot(tr, linestyle="-", color="black", label="Train (max - min)")
        plt.plot(te, linestyle="-", color="gray", label="Test (max - min)")
        path = os.path.join(seed_dir, f"gap_seed{seed}.png")
        format_plot(path, f"Max-Min Class Accuracy — seed {seed}", "Training Step", "Accuracy")
        written.append(path)

    plt.figure(figsize=(10, 6))
    plt.plot(train_mean, linestyle="-", color="black", label="Train (max - min)")
    plt.plot(test_mean, linestyle="-", color="gray", label="Test (max - min)")
    path = os.path.join(out_dir, "avg_gap.png")
    format_plot(path, "Average Max-Min Gap at Each Step", "Training Step", "Average Accuracy")
    written.append(path)
    return written


def make_animation(
    num_seeds: int, num_steps: int, frame_dir: str, out_dir: str, duration: int = 200
) -> List[str]:
    """Stitch the per-step decision-boundary frames into one GIF per seed.

    Args:
        num_seeds: Number of seeds that produced frames.
        num_steps: Frames per seed.
        frame_dir: Directory holding ``{seed}-{step}.png`` frames.
        out_dir: Directory to write ``boundaries-seed{N}.gif`` into.
        duration: Milliseconds per frame.

    Returns:
        The GIF paths written. Seeds with no frames on disk are skipped.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []

    for seed in range(num_seeds):
        paths = [os.path.join(frame_dir, f"{seed}-{step}.png") for step in range(num_steps)]
        frames = [Image.open(p) for p in paths if os.path.exists(p)]
        if not frames:
            continue
        gif_path = os.path.join(out_dir, f"boundaries-seed{seed}.gif")
        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:], duration=duration, loop=0
        )
        written.append(gif_path)

    return written
