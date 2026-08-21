"""Entry point: ``python -m cir.train --config configs/<name>.yaml``.

Reads a config, looks the experiment up in the registry, and runs it. All the
behaviour lives in the experiment classes; this file only wires them together.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Sequence

from cir.cli import load_config, parse_args
from cir.experiments.registry import get_experiment
from cir.logging_utils import SimpleLogger

__all__ = ["run_from_config", "main"]


def run_from_config(cfg: Dict[str, Any]) -> Any:
    """Instantiate and run the experiment named by ``cfg["experiment"]``.

    Args:
        cfg: A config mapping.

    Returns:
        Whatever the experiment's ``run()`` returns.

    Raises:
        ValueError: If the config has no ``experiment`` key.
        KeyError: If the named experiment is not registered.
    """
    name = cfg.get("experiment")
    if name is None:
        raise ValueError("Config must include an 'experiment' key, e.g. 'experiment: vae'")

    experiment_class = get_experiment(name)
    logger = SimpleLogger(cfg.get("log_dir", cfg.get("output_dir", "runs")), config=cfg)

    print(f"Running experiment {name!r} ({experiment_class.__name__})")
    return experiment_class(cfg, logger=logger).run()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, load the config, and run.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)
    run_from_config(load_config(args.config, args.override))
    return 0


if __name__ == "__main__":
    sys.exit(main())
