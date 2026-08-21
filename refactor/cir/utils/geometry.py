"""Geometry of class clusters.

The central idea of this project is that a set of class means is "fair" to a
linear classifier when the means are evenly spaced around the origin — a 2D
simplex equiangular tight frame (ETF), i.e. the vertices of a regular polygon.
This module builds such configurations, measures how far a given configuration
is from one, and produces the per-class scale factors or shift vectors that move
an arbitrary configuration onto one.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch

__all__ = [
    "make_evenly_spaced_targets",
    "rotate_classes",
    "transform_to_even_space",
    "is_regular_polygon",
    "generate_samples",
    "create_labels",
]


def make_evenly_spaced_targets(num_points: int, radius: float = 1.0) -> torch.Tensor:
    """Place ``num_points`` evenly around a circle centred on the origin.

    The points are the vertices of a regular polygon, which is the 2D simplex
    ETF configuration this project projects toward.

    Args:
        num_points: Number of target points (one per class). Must be >= 1.
        radius: Radius of the circle.

    Returns:
        Tensor of shape ``(num_points, 2)`` in **adjacency order** — consecutive
        rows are neighbouring vertices — starting from the topmost point so that
        class *i* keeps a stable position across runs.

        Adjacency order matters: :func:`is_regular_polygon` measures the distance
        between consecutive rows, so a merely deterministic ordering is not
        enough. Sorting the vertices by position (as the pre-refactor code did)
        interleaves opposite sides of the polygon and reports a genuine ETF as
        irregular for five or more classes.
    """
    if num_points < 1:
        raise ValueError(f"num_points must be >= 1, got {num_points}")

    # An odd number of points starts at the top so the layout is left-right
    # symmetric; an even number is offset by half a step for the same reason.
    if num_points % 2:
        start_angle = math.pi / 2
    else:
        start_angle = (math.pi / 2) + (math.pi / num_points)

    angles = torch.linspace(0, 2 * math.pi, steps=num_points + 1)[:-1] + start_angle
    points = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)

    # Rotate the cycle so the topmost vertex (ties broken by smallest x) comes
    # first. Rolling preserves adjacency; sorting would not.
    keys = [(-points[i, 1].item(), points[i, 0].item()) for i in range(num_points)]
    start = min(range(num_points), key=lambda i: keys[i])
    return torch.roll(points, shifts=-start, dims=0)


def rotate_classes(means: torch.Tensor, rotations: Sequence[float]) -> torch.Tensor:
    """Rotate each class mean about the origin by its own angle.

    Used to build deliberately *unfair* configurations: rotating a single class
    off the regular-polygon vertices is one way to break the ETF.

    Args:
        means: Class means, shape ``(num_classes, 2)``.
        rotations: Rotation angle in **degrees** for each class.

    Returns:
        A new tensor of rotated means, shape ``(num_classes, 2)``.
    """
    if len(rotations) != means.shape[0]:
        raise ValueError(
            f"rotations has {len(rotations)} entries but there are {means.shape[0]} classes"
        )

    rotated = torch.empty_like(means)
    for i, degrees in enumerate(rotations):
        theta = math.radians(float(degrees))
        rotation_matrix = torch.tensor(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            dtype=means.dtype,
        )
        rotated[i] = rotation_matrix @ means[i]
    return rotated


def transform_to_even_space(
    means: torch.Tensor, mode: str = "shift", ref_mode: str = "mean"
) -> torch.Tensor:
    """Compute the transformation that moves ``means`` onto an ETF.

    Args:
        means: Class means, shape ``(num_classes, 2)``.
        mode: ``"shift"`` returns displacement vectors onto the target vertices;
            ``"scale"`` returns per-class factors that equalize the norms only.
        ref_mode: Which norm sets the radius of the target polygon —
            ``"mean"``, ``"median"``, or ``"max"``.

    Returns:
        ``(num_classes, 2)`` displacements when ``mode="shift"``, or
        ``(num_classes,)`` scale factors when ``mode="scale"``.

    Raises:
        ValueError: If ``mode`` or ``ref_mode`` is not one of the listed options.
    """
    norms = torch.linalg.norm(means, dim=1)

    if ref_mode == "mean":
        radius = norms.mean().item()
    elif ref_mode == "max":
        radius = norms.max().item()
    elif ref_mode == "median":
        radius = norms.median().item()
    else:
        raise ValueError(f"ref_mode must be 'mean', 'max', or 'median', got {ref_mode!r}")

    if mode == "shift":
        targets = make_evenly_spaced_targets(means.shape[0], radius)
        return targets - means
    if mode in ("scale", "norm"):
        # Equalize norms only; 1e-9 guards a class mean sitting at the origin.
        return radius / (norms + 1e-9)
    raise ValueError(f"mode must be 'shift' or 'scale', got {mode!r}")


def is_regular_polygon(points: torch.Tensor, tol: float = 1e-4) -> bool:
    """Report whether ``points`` form a regular polygon (an ETF in 2D).

    Args:
        points: Shape ``(num_points, 2)``, in adjacency order.
        tol: Absolute tolerance on the comparison of squared side lengths.

    Returns:
        ``True`` when every adjacent pair is equidistant.

    Raises:
        ValueError: If fewer than three points are supplied.
    """
    num_points = points.shape[0]
    if num_points < 3:
        raise ValueError("Need at least 3 points to form an ETF!")

    rolled = torch.roll(points, shifts=-1, dims=0)
    squared_sides = ((points - rolled) ** 2).sum(dim=1)
    return bool(torch.all(torch.isclose(squared_sides, squared_sides[0], atol=tol)))


def generate_samples(
    means: torch.Tensor,
    covs: Sequence[torch.Tensor],
    num_classes: int,
    samples_per_class: int,
) -> torch.Tensor:
    """Draw Gaussian clusters, one per class.

    Args:
        means: Cluster centres, shape ``(num_classes, 2)``.
        covs: One ``(2, 2)`` covariance matrix per class.
        num_classes: Number of clusters to draw.
        samples_per_class: Points drawn from each cluster.

    Returns:
        Tensor of shape ``(num_classes * samples_per_class, 2)``, grouped by
        class: the first ``samples_per_class`` rows are class 0, and so on.
    """
    chunks: List[torch.Tensor] = []
    for class_id in range(num_classes):
        dist = torch.distributions.MultivariateNormal(means[class_id], covs[class_id])
        chunks.append(dist.sample((samples_per_class,)))
    return torch.cat(chunks, dim=0)


def create_labels(
    num_classes: int, samples_per_class: int, classes: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Build one-hot labels matching the layout of :func:`generate_samples`.

    Args:
        num_classes: Number of classes.
        samples_per_class: Points per class.
        classes: One-hot row vector for each class, e.g. ``list(torch.eye(n))``.

    Returns:
        Tensor of shape ``(num_classes * samples_per_class, num_classes)``.
    """
    return torch.cat(
        [classes[i].expand(samples_per_class, -1) for i in range(num_classes)], dim=0
    )


def scale_samples(
    x: torch.Tensor, y: torch.Tensor, scalars: torch.Tensor, decay: float
) -> torch.Tensor:
    """Scale each sample toward its class's ETF norm.

    ``decay`` interpolates between "leave the data alone" and "apply the full
    projection", so the experiments can fade the projection out as accuracy
    rises (``decay = 1 - mean_accuracy``).

    Args:
        x: Samples, shape ``(N, 2)``.
        y: One-hot labels, shape ``(N, C)``.
        scalars: Per-class scale factor, shape ``(C,)``.
        decay: Projection strength in ``[0, 1]``.

    Returns:
        A new tensor of projected samples, shape ``(N, 2)``.
    """
    per_sample = scalars[y.argmax(dim=1)].unsqueeze(1)
    return x * per_sample * decay + x * (1.0 - decay)


def shift_samples(
    x: torch.Tensor, y: torch.Tensor, shifts: torch.Tensor, decay: float
) -> torch.Tensor:
    """Shift each sample toward its class's ETF vertex.

    Args:
        x: Samples, shape ``(N, 2)``.
        y: One-hot labels, shape ``(N, C)``.
        shifts: Per-class displacement, shape ``(C, 2)``.
        decay: Projection strength in ``[0, 1]``.

    Returns:
        A new tensor of projected samples, shape ``(N, 2)``.
    """
    per_sample = shifts[y.argmax(dim=1)]
    return (x + per_sample) * decay + x * (1.0 - decay)


def apply_projection(
    x: torch.Tensor,
    y: torch.Tensor,
    means: torch.Tensor,
    transform: torch.Tensor,
    mode: str,
    decay: float,
) -> tuple:
    """Apply the ETF projection chosen by ``mode`` to both samples and means.

    Args:
        x: Samples, shape ``(N, 2)``.
        y: One-hot labels, shape ``(N, C)``.
        means: Class means, shape ``(C, 2)``.
        transform: Output of :func:`transform_to_even_space` for the same mode.
        mode: ``"shift"``, ``"scale"``, or ``"norm"`` (an alias for ``"scale"``).
        decay: Projection strength in ``[0, 1]``.

    Returns:
        ``(projected_x, projected_means)``.
    """
    if mode in ("scale", "norm"):
        return scale_samples(x, y, transform, decay), means * transform[:, None]
    if mode == "shift":
        return shift_samples(x, y, transform, decay), means + transform
    raise ValueError(f"mode must be 'shift', 'scale', or 'norm', got {mode!r}")


__all__ += ["scale_samples", "shift_samples", "apply_projection"]
