"""Basket experiment wrapper registered for article replication."""

from __future__ import annotations

from diffml.experiments_digital_v2 import run_basket_experiment as _basket_impl

from .config_experiments import ExperimentConfig
from .experiments_registry import register_experiment


@register_experiment("basket")
def run_basket_experiment(config: ExperimentConfig) -> None:
    """Run the basket digital option experiment."""
    _basket_impl(config.to_core_config())


__all__ = ["run_basket_experiment"]
