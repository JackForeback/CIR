"""The experiment contract.

Every experiment subclasses :class:`BaseExperiment`, which owns the parts that
never vary — config access, device selection, seeding, optimizer and loss
construction, output directories, logging — and leaves the parts that do vary to
three abstract methods: :meth:`~BaseExperiment.build_model`,
:meth:`~BaseExperiment.get_dataloaders`, and :meth:`~BaseExperiment.compute_loss`.

The default :meth:`~BaseExperiment.run` is an epoch loop, which is what the VAE
experiments need. The linear experiment overrides it, because it runs many seeds
of full-batch gradient steps rather than epochs over minibatches — that is a
legitimate difference in training shape, not a reason for a second base class.
"""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from cir.logging_utils import SimpleLogger

__all__ = ["BaseExperiment", "resolve_device", "seed_everything"]

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
}

LOSS_FUNCTIONS = {
    "mse": nn.MSELoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "nll": nn.NLLLoss,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
}


def resolve_device(spec: str = "auto") -> torch.device:
    """Turn a config device string into a concrete device.

    Args:
        spec: ``"auto"``, ``"cpu"``, ``"cuda"``, or any explicit device string.

    Returns:
        ``cuda`` when ``spec`` is ``"auto"`` and CUDA is available, else the
        requested device.
    """
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs together.

    Args:
        seed: The seed to apply.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BaseExperiment(ABC):
    """Shared scaffolding for every experiment.

    Args:
        cfg: The parsed config mapping.
        logger: Where to write metrics. One is created under ``log_dir`` if not
            supplied.

    Attributes:
        cfg: The config mapping.
        device: Device the model runs on.
        output_dir: Directory for figures and other artifacts.
        history: Accumulated per-epoch metrics.
        model: Built by :meth:`setup`; ``None`` until then.
    """

    def __init__(self, cfg: Dict[str, Any], logger: Optional[SimpleLogger] = None):
        self.cfg = cfg
        self.device = resolve_device(str(cfg.get("device", "auto")))
        self.output_dir = cfg.get("output_dir", os.path.join("runs", str(cfg.get("experiment", "run"))))
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = logger or SimpleLogger(cfg.get("log_dir", self.output_dir), config=cfg)
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}

        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_function: Optional[nn.Module] = None

        seed_everything(int(cfg.get("seed", 42)))

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Construct and return the model for this experiment."""

    @abstractmethod
    def get_dataloaders(self) -> Tuple[Any, Any]:
        """Return ``(train_loader, val_loader)``."""

    @abstractmethod
    def compute_loss(self, batch: Any) -> torch.Tensor:
        """Return the scalar loss for one batch."""

    def on_run_end(self) -> None:
        """Hook for post-training work (plots, animations). No-op by default."""

    # ------------------------------------------------------------------
    # Shared machinery
    # ------------------------------------------------------------------

    def setup(self) -> nn.Module:
        """Build (or rebuild) the model, optimizer, and loss function.

        Safe to call more than once: the linear experiment calls it per seed to
        get a fresh model each run.

        Returns:
            The newly built model.
        """
        self.model = self.build_model().to(self.device)
        self.optimizer = self.build_optimizer(self.model)
        self.loss_function = self.build_loss_function()
        return self.model

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Construct the optimizer named by ``cfg["optimizer"]``.

        Args:
            model: The model whose parameters are optimized.

        Returns:
            The configured optimizer. ``momentum`` is passed through for SGD.

        Raises:
            ValueError: If the optimizer name is unsupported.
        """
        name = str(self.cfg.get("optimizer", "adam")).lower()
        if name not in OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer {name!r}; expected one of {sorted(OPTIMIZERS)}")

        kwargs: Dict[str, Any] = {"lr": float(self.cfg.get("lr", 1e-3))}
        if name == "sgd":
            kwargs["momentum"] = float(self.cfg.get("momentum", 0.0))

        # Only parameters that can actually be updated: models such as
        # cir.models.alternating.FOLVAE deliberately freeze part of themselves.
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("model has no trainable parameters")
        return OPTIMIZERS[name](parameters, **kwargs)

    def build_loss_function(self) -> nn.Module:
        """Construct the loss named by ``cfg["loss_function"]``.

        Returns:
            The configured loss module, defaulting to MSE.

        Raises:
            ValueError: If the loss name is unsupported.
        """
        name = str(self.cfg.get("loss_function", "mse")).lower()
        if name not in LOSS_FUNCTIONS:
            raise ValueError(
                f"Unsupported loss_function {name!r}; expected one of {sorted(LOSS_FUNCTIONS)}"
            )
        return LOSS_FUNCTIONS[name]()

    def train_epoch(self, loader: Any) -> float:
        """Run one training pass over ``loader``.

        Args:
            loader: An iterable of batches accepted by :meth:`compute_loss`.

        Returns:
            The mean training loss over the epoch, or ``0.0`` for an empty loader.
        """
        self.model.train()
        total, batches = 0.0, 0
        log_every = int(self.cfg.get("log_every", 100))

        for batch_idx, batch in enumerate(loader):
            loss = self.compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total += loss.item()
            batches += 1
            if log_every and batch_idx % log_every == 0:
                print(f"  train batch {batch_idx}: loss {loss.item():.4f}")

        return total / batches if batches else 0.0

    @torch.no_grad()
    def validate_epoch(self, loader: Any) -> float:
        """Run one evaluation pass over ``loader``.

        Args:
            loader: An iterable of batches accepted by :meth:`compute_loss`.

        Returns:
            The mean validation loss, or ``0.0`` for an empty loader.
        """
        self.model.eval()
        total, batches = 0.0, 0
        for batch in loader:
            total += self.compute_loss(batch).item()
            batches += 1
        return total / batches if batches else 0.0

    def run(self) -> Dict[str, list]:
        """Train for ``cfg["epochs"]`` epochs, logging each one.

        Returns:
            The accumulated :attr:`history`.
        """
        self.setup()
        train_loader, val_loader = self.get_dataloaders()
        epochs = int(self.cfg.get("epochs", 1))

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.logger.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train loss {train_loss:.4f} | val loss {val_loss:.4f}"
            )

        self.on_run_end()
        return self.history
