"""Barrier experiment wrapper registered for article replication."""

from __future__ import annotations

from diffml.experiments_digital_v2 import run_barrier_experiment as _barrier_impl

from .config_experiments import ExperimentConfig
from .experiments_registry import register_experiment


@register_experiment("barrier")
def run_barrier_experiment(config: ExperimentConfig) -> None:
    """Run the barrier option experiment using the shared implementation."""
    _barrier_impl(config)


__all__ = ["run_barrier_experiment"]
