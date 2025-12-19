"""Unified experiment runner using configuration files.

This module provides functions to run experiments based on TOML configuration
files, using the experiment registry system.
"""

from pathlib import Path

from diffml.config_experiments import load_experiment_config
from diffml.experiments_registry import get_experiment, list_registered_experiments


def run_experiment_from_config(config_path: str) -> None:
    """Load an ExperimentConfig and invoke the registered experiment function.

    This function loads a configuration file, looks up the corresponding
    experiment in the registry, and executes it.

    Parameters
    ----------
    config_path : str
        Path to the TOML configuration file.

    Raises
    ------
    FileNotFoundError
        If the configuration file doesn't exist.
    KeyError
        If the experiment name in the config is not registered.
    ValueError
        If the configuration is invalid.

    Examples
    --------
    >>> run_experiment_from_config("configs/digital_default.toml")
    Loading configuration from configs/digital_default.toml...
    Running experiment: digital
    ...
    """
    print(f"Loading configuration from {config_path}...")

    # Load the configuration
    config = load_experiment_config(config_path)

    print("Configuration loaded successfully!")
    print(f"  Experiment: {config.name}")
    print(f"  Seed: {config.seed}")
    print(f"  Training epochs: {config.training.n_epochs}")
    print(f"  Training samples: {config.m_train}")
    print(f"  Test samples: {config.m_test}")

    # Import experiment modules to ensure they're registered
    # This is necessary because the decorators only run when modules are imported
    _ensure_experiments_registered()

    # Get the experiment function from registry
    print(f"\nLooking up experiment '{config.name}' in registry...")
    experiment_func = get_experiment(config.name)

    # Run the experiment
    print("Starting experiment execution...")
    print("=" * 80)

    try:
        experiment_func(config)
        print("\n" + "=" * 80)
        print(f"Experiment '{config.name}' completed successfully!")
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"ERROR: Experiment '{config.name}' failed with error:")
        print(f"  {type(e).__name__}: {e}")
        raise


def _ensure_experiments_registered() -> None:
    """Ensure all experiment modules are imported so they register themselves.

    This function imports all experiment modules that use the @register_experiment
    decorator. The import causes the decorators to run and populate the registry.
    """
    # Import the v2 experiments module which has all registered experiments
    import diffml.experiments_digital_v2  # noqa: F401

    # Alternatively, import individual experiment modules if they're updated
    # import diffml.experiments_digital
    # import diffml.experiments_barrier
    # import diffml.experiments_basket
    # import diffml.experiments_smoothing
    # import diffml.experiments_gamma


def run_experiment_from_dict(config_dict: dict) -> None:
    """Run an experiment from a configuration dictionary.

    This is useful for programmatic experiment execution without files.

    Parameters
    ----------
    config_dict : dict
        Dictionary containing experiment configuration.

    Examples
    --------
    >>> config_dict = {
    ...     "experiment": {"name": "digital", "seed": 42},
    ...     "training": {"n_epochs": 100},
    ...     "dataset": {"m_train": 50, "m_test": 20},
    ...     "payoff": {"K": 100.0}
    ... }
    >>> run_experiment_from_dict(config_dict)
    """
    # Create a temporary config file
    import tempfile

    import toml

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        toml.dump(config_dict, f)
        temp_path = f.name

    try:
        run_experiment_from_config(temp_path)
    finally:
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)


def list_available_experiments() -> None:
    """List all available experiments in the registry.

    This function prints a formatted list of all registered experiments.
    """
    _ensure_experiments_registered()

    experiments = list_registered_experiments()

    print("\nAvailable Experiments:")
    print("=" * 40)

    if not experiments:
        print("No experiments registered.")
    else:
        for i, name in enumerate(experiments, 1):
            print(f"  {i}. {name}")

    print("\nTo run an experiment, use:")
    print("  python scripts/run_experiment.py --config configs/<experiment>_default.toml")


def validate_config(config_path: str) -> bool:
    """Validate a configuration file without running the experiment.

    Parameters
    ----------
    config_path : str
        Path to the TOML configuration file.

    Returns
    -------
    bool
        True if the configuration is valid, False otherwise.
    """
    try:
        config = load_experiment_config(config_path)
        _ensure_experiments_registered()
        experiment_func = get_experiment(config.name)

        print("[OK] Configuration file is valid")
        print(f"  Experiment: {config.name}")
        print(f"  Function: {experiment_func.__name__}")
        return True

    except Exception as e:
        print("[FAIL] Configuration validation failed:")
        print(f"  {type(e).__name__}: {e}")
        return False
