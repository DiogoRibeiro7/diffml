"""Wrappers exposing configured digital experiments for article replication."""

from __future__ import annotations

from diffml.experiments_digital_v2 import run_digital_experiment as _digital_impl

from .config_experiments import ExperimentConfig
from .experiments_registry import register_experiment


@register_experiment("digital")
def run_digital_experiment(config: ExperimentConfig) -> None:
    """Run the digital option experiment using the shared implementation."""
    _digital_impl(config.to_core_config())


__all__ = ["run_digital_experiment"]
