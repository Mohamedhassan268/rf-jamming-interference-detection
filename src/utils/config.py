"""YAML loading, inheritance, and explicit placeholder validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigError(ValueError):
    """Human-readable experiment configuration error."""


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load YAML with optional relative `extends` inheritance."""
    config_path = Path(path).resolve()
    if config_path in _stack:
        raise ConfigError(f"Circular config inheritance: {config_path}")
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError("Top-level YAML configuration must be a mapping.")
    parent = config.pop("extends", None)
    if parent is None:
        return config
    base = load_config(config_path.parent / parent, _stack + (config_path,))
    return _merge(base, config)


def _get(config: dict[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for component in dotted_path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ConfigError(f"Missing required configuration field: {dotted_path}")
        value = value[component]
    return value


def require_resolved(config: dict[str, Any], fields: Iterable[str]) -> None:
    """Reject required fields that are missing, null, or explicitly `TBD`."""
    unresolved = []
    for field in fields:
        value = _get(config, field)
        if value is None or (isinstance(value, str) and value.strip().upper() == "TBD"):
            unresolved.append(field)
    if unresolved:
        raise ConfigError(
            "Cannot continue because required experimental facts are unresolved: "
            + ", ".join(unresolved)
            + ". Replace each TBD only with verified project information."
        )

