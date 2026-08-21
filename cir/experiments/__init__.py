"""Experiment implementations, keyed by name in :mod:`cir.experiments.registry`."""

from cir.experiments.alternating import AlternatingVAEExperiment
from cir.experiments.alvae import ALVAEExperiment
from cir.experiments.base import BaseExperiment
from cir.experiments.linear import LinearExperiment
from cir.experiments.registry import EXPERIMENTS, get_experiment
from cir.experiments.vae import VAEExperiment

__all__ = [
    "BaseExperiment",
    "LinearExperiment",
    "VAEExperiment",
    "ALVAEExperiment",
    "AlternatingVAEExperiment",
    "EXPERIMENTS",
    "get_experiment",
]
