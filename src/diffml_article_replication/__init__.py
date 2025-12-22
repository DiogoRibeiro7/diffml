"""Public API for diffml_article_replication package."""

from .api import Mode, diffml_price
from .benchmark import format_benchmark_table, run_digital_benchmark, run_lambda_delta_sweep
from .config_experiments import ExperimentConfig, load_experiment_config, save_experiment_config
from .experiments_path_dependent import (
    run_arithmetic_asian_experiment,
    run_lookback_call_experiment,
)
from .experiments_registry import (
    EXPERIMENT_REGISTRY,
    ExperimentFunc,
    clear_registry,
    get_experiment,
    list_registered_experiments,
    register_experiment,
)
from .run_experiment import run_experiment_from_config

__all__ = [
    "ExperimentConfig",
    "load_experiment_config",
    "save_experiment_config",
    "EXPERIMENT_REGISTRY",
    "ExperimentFunc",
    "register_experiment",
    "list_registered_experiments",
    "get_experiment",
    "clear_registry",
    "run_experiment_from_config",
    "run_digital_benchmark",
    "run_lambda_delta_sweep",
    "format_benchmark_table",
    "diffml_price",
    "Mode",
    "run_arithmetic_asian_experiment",
    "run_lookback_call_experiment",
]
