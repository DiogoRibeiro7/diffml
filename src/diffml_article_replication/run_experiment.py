"""Unified experiment runner for diffml_article_replication."""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path
from typing import Any

from .config_experiments import load_experiment_config
from .experiments_registry import get_experiment, list_registered_experiments


def run_experiment_from_config(config_path: str) -> None:
    """Load an ExperimentConfig and execute the matching registry entry."""
    print(f"Loading configuration from {config_path}...")
    config = load_experiment_config(config_path)

    _ensure_experiments_registered()
    experiment = get_experiment(config.name)

    print(f"Running experiment '{config.name}' with seed {config.seed}")
    experiment(config)


def _ensure_experiments_registered() -> None:
    """Import experiment modules so registry decorators have executed."""
    modules = [
        "diffml_article_replication.experiments_digital",
        "diffml_article_replication.experiments_barrier",
        "diffml_article_replication.experiments_basket",
        "diffml_article_replication.experiments_smoothing",
        "diffml_article_replication.experiments_asian",
    ]
    for module in modules:
        importlib.import_module(module)


def run_experiment_from_dict(config_dict: dict[str, Any]) -> None:
    """Run experiments using in-memory configurations."""
    toml_module = importlib.import_module("toml")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as fh:
        toml_module.dump(config_dict, fh)
        temp_path = fh.name

    try:
        run_experiment_from_config(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def list_available_experiments() -> None:
    """Pretty-print registered experiments to stdout."""
    _ensure_experiments_registered()
    names = list_registered_experiments()
    print("\nAvailable experiments:")
    for name in names:
        print(f"  - {name}")


def validate_config(config_path: str) -> bool:
    """Validate a configuration without running the experiment."""
    try:
        config = load_experiment_config(config_path)
        _ensure_experiments_registered()
        get_experiment(config.name)
        return True
    except Exception as exc:  # pragma: no cover - CLI feedback path
        print(f"Validation failed: {exc}")
        return False


__all__ = [
    "run_experiment_from_config",
    "run_experiment_from_dict",
    "list_available_experiments",
    "validate_config",
]
