"""Run logging.

:class:`SimpleLogger` appends one JSON record per call and rewrites the file each
time, so a run that dies partway through still leaves readable results. It also
snapshots the config that produced the run, which is what makes a run
reproducible after the fact.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

__all__ = ["SimpleLogger"]


class SimpleLogger:
    """Write run metrics to ``<save_dir>/results.json``.

    Args:
        save_dir: Directory for the log files; created if missing.
        config: Optional config snapshot, written to ``config.json``.

    Attributes:
        path: Path of the metrics file.
        records: The records written so far.
    """

    def __init__(self, save_dir: str = "runs", config: Optional[Dict[str, Any]] = None):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.path = os.path.join(save_dir, "results.json")
        self.records: List[Dict[str, Any]] = []

        if config is not None:
            with open(os.path.join(save_dir, "config.json"), "w") as handle:
                json.dump(config, handle, indent=2, default=str)

    def log(self, **metrics: Any) -> Dict[str, Any]:
        """Append one record and flush the file.

        Any keyword arguments are accepted, so an experiment can log whatever it
        measures — ``epoch``/``train_loss``/``val_loss`` for the VAEs, per-class
        accuracy gaps for the linear experiment.

        Args:
            **metrics: The fields of this record.

        Returns:
            The record that was appended.
        """
        self.records.append(dict(metrics))
        with open(self.path, "w") as handle:
            json.dump(self.records, handle, indent=2, default=str)
        return self.records[-1]

    def __repr__(self) -> str:
        return f"SimpleLogger(path={self.path!r}, records={len(self.records)})"
