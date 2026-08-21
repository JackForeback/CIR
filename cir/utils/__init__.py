"""Shared utilities.

Submodules are imported eagerly so ``from cir.utils import geometry`` and
``cir.utils.geometry`` both work without a separate import line.
"""

from cir.utils import evolution, geometry, losses, metrics, plotting, solvers

__all__ = ["geometry", "losses", "metrics", "evolution", "plotting", "solvers"]
