"""Experiment registry for diffml_article_replication."""

from __future__ import annotations

from collections.abc import Callable

from .config_experiments import ExperimentConfig

ExperimentFunc = Callable[[ExperimentConfig], None]
EXPERIMENT_REGISTRY: dict[str, ExperimentFunc] = {}


def register_experiment(name: str) -> Callable[[ExperimentFunc], ExperimentFunc]:
    """Register an experiment in the global registry."""
    def decorator(func: ExperimentFunc) -> ExperimentFunc:
        if name in EXPERIMENT_REGISTRY:
            raise ValueError(f"Experiment '{name}' is already registered")
        EXPERIMENT_REGISTRY[name] = func
        return func

    return decorator


def list_registered_experiments() -> list[str]:
    """Return sorted experiment names."""
    return sorted(EXPERIMENT_REGISTRY)


def get_experiment(name: str) -> ExperimentFunc:
    """Return the registered experiment by name."""
    try:
        return EXPERIMENT_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive branch
        available = ", ".join(list_registered_experiments()) or "<none>"
        raise KeyError(
            f"Experiment '{name}' not found. Available experiments: {available}"
        ) from exc


def clear_registry() -> None:
    """Remove all registered experiments (primarily for testing)."""
    EXPERIMENT_REGISTRY.clear()


__all__ = [
    "ExperimentFunc",
    "EXPERIMENT_REGISTRY",
    "register_experiment",
    "list_registered_experiments",
    "get_experiment",
    "clear_registry",
]
