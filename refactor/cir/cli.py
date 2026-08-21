"""Command-line parsing and YAML config loading.

Configs are plain YAML dicts. Any value can be replaced on the command line with
``--override key=value``; dotted keys reach into nested mappings, and values are
parsed as YAML so ``true``, ``3``, ``1e-3``, and ``[1,2,3]`` all arrive with the
type you would expect.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Sequence

import yaml

__all__ = ["parse_args", "load_config", "apply_overrides", "set_nested"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the training entry point's arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        A namespace with ``config`` and ``override``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m cir.train",
        description="Run a CIR experiment from a YAML config.",
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config entries, e.g. --override epochs=5 flags.evo_weights=true",
    )
    return parser.parse_args(argv)


def set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Assign ``value`` at a dotted path inside ``cfg``, creating dicts as needed.

    Args:
        cfg: Config mapping, modified in place.
        dotted_key: Key path, e.g. ``"flags.evo_weights"``.
        value: Value to assign.

    Raises:
        ValueError: If an intermediate key exists but is not a mapping.
    """
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot descend into non-mapping key {key!r} of {dotted_key!r}")
    node[keys[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply ``KEY=VALUE`` overrides to a config.

    Args:
        cfg: Config mapping, modified in place.
        overrides: Strings of the form ``key=value`` or ``a.b=value``. Values are
            parsed as YAML, so quoting a value keeps it a string.

    Returns:
        The same mapping, for chaining.

    Raises:
        ValueError: If an override is missing its ``=``.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must be of the form KEY=VALUE")
        key, raw_value = item.split("=", 1)
        set_nested(cfg, key.strip(), _parse_value(raw_value))
    return cfg


def _parse_value(raw: str):
    """Parse an override value, YAML first with a numeric fallback.

    YAML 1.1 does not recognize exponent forms without a decimal point, so a
    plain ``lr=1e-3`` would otherwise arrive as the string ``"1e-3"``. Anything
    that is neither valid YAML nor a number stays a string.

    Args:
        raw: The text on the right of the ``=``.

    Returns:
        The parsed value.
    """
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw

    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path: str, overrides: Sequence[str] = ()) -> Dict[str, Any]:
    """Load a YAML config and apply command-line overrides.

    Args:
        path: Path to the YAML file.
        overrides: ``KEY=VALUE`` strings, see :func:`apply_overrides`.

    Returns:
        The resulting config mapping.

    Raises:
        ValueError: If the file does not contain a top-level mapping.
    """
    with open(path, "r") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path!r} must contain a top-level mapping, got {type(cfg).__name__}")
    return apply_overrides(cfg, overrides)
