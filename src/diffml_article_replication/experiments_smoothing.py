"""Smoothing experiment wrapper for article replication."""

from __future__ import annotations

from diffml.experiments_digital_v2 import run_smoothing_experiment as _smoothing_impl

from .config_experiments import ExperimentConfig
from .experiments_registry import register_experiment


@register_experiment("smoothing")
def run_smoothing_experiment(config: ExperimentConfig) -> None:
    """Run the smoothing experiment with the consolidated implementation."""
    _smoothing_impl(config.to_core_config())


__all__ = ["run_smoothing_experiment"]
