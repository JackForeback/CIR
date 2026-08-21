"""Experiment registry.

Maps the ``experiment:`` key in a config to the class that implements it. Add a
new experiment by importing it here and adding one entry — nothing else in the
framework needs to change.
"""

from __future__ import annotations

from typing import Dict, Type

from cir.experiments.alternating import AlternatingVAEExperiment
from cir.experiments.alvae import ALVAEExperiment
from cir.experiments.base import BaseExperiment
from cir.experiments.linear import LinearExperiment
from cir.experiments.vae import VAEExperiment

__all__ = ["EXPERIMENTS", "get_experiment"]

EXPERIMENTS: Dict[str, Type[BaseExperiment]] = {
    "linear": LinearExperiment,
    "vae": VAEExperiment,
    "alvae": ALVAEExperiment,
    "altvae": AlternatingVAEExperiment,
}


def get_experiment(name: str) -> Type[BaseExperiment]:
    """Look up an experiment class by config name.

    Args:
        name: The value of the config's ``experiment`` key.

    Returns:
        The experiment class.

    Raises:
        KeyError: If ``name`` is not registered, listing what is.
    """
    if name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment {name!r}; registered: {sorted(EXPERIMENTS)}")
    return EXPERIMENTS[name]
