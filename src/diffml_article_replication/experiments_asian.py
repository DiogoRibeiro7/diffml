"""Asian option experiment wrapper for article replication."""

from __future__ import annotations

from diffml.experiments_digital_v2 import run_asian_experiment as _asian_impl

from .config_experiments import ExperimentConfig
from .experiments_registry import register_experiment


@register_experiment("asian")
def run_asian_experiment(config: ExperimentConfig) -> None:
    """Run the arithmetic Asian option experiment."""
    _asian_impl(config)


__all__ = ["run_asian_experiment"]
