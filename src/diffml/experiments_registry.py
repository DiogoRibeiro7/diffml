"""Experiment registry for managing and running experiments.

This module provides a registry system for experiments, allowing them to be
registered by name and invoked with configuration objects.
"""

from collections.abc import Callable

from diffml.config_experiments import ExperimentConfig

# Type alias for experiment functions
ExperimentFunc = Callable[[ExperimentConfig], None]

# Global experiment registry
EXPERIMENT_REGISTRY: dict[str, ExperimentFunc] = {}


def register_experiment(name: str) -> Callable[[ExperimentFunc], ExperimentFunc]:
    """Decorator to register an experiment function.

    This decorator registers a function in the global experiment registry,
    making it accessible by name for configuration-based execution.

    Parameters
    ----------
    name : str
        The name to register the experiment under. This should match
        the 'name' field in configuration files.

    Returns
    -------
    Callable[[ExperimentFunc], ExperimentFunc]
        A decorator function that registers the experiment and returns it unchanged.

    Raises
    ------
    ValueError
        If an experiment with the given name is already registered.

    Examples
    --------
    >>> @register_experiment("my_experiment")
    ... def run_my_experiment(config: ExperimentConfig) -> None:
    ...     print(f"Running {config.name} with seed {config.seed}")
    ...
    >>> # Now accessible via EXPERIMENT_REGISTRY["my_experiment"]
    """
    def decorator(func: ExperimentFunc) -> ExperimentFunc:
        if name in EXPERIMENT_REGISTRY:
            raise ValueError(
                f"Experiment '{name}' is already registered. "
                f"Existing function: {EXPERIMENT_REGISTRY[name].__name__}"
            )
        EXPERIMENT_REGISTRY[name] = func
        # Add metadata to the function
        func.__experiment_name__ = name  # type: ignore
        return func

    return decorator


def list_registered_experiments() -> list[str]:
    """List all registered experiment names.

    Returns
    -------
    list[str]
        Sorted list of registered experiment names.

    Examples
    --------
    >>> experiments = list_registered_experiments()
    >>> print(f"Available experiments: {', '.join(experiments)}")
    """
    return sorted(EXPERIMENT_REGISTRY.keys())


def get_experiment(name: str) -> ExperimentFunc:
    """Get a registered experiment function by name.

    Parameters
    ----------
    name : str
        The name of the experiment to retrieve.

    Returns
    -------
    ExperimentFunc
        The registered experiment function.

    Raises
    ------
    KeyError
        If no experiment with the given name is registered.

    Examples
    --------
    >>> experiment_func = get_experiment("digital")
    >>> # experiment_func can now be called with an ExperimentConfig
    """
    if name not in EXPERIMENT_REGISTRY:
        available = list_registered_experiments()
        raise KeyError(
            f"Experiment '{name}' not found in registry. "
            f"Available experiments: {', '.join(available) if available else 'none'}"
        )
    return EXPERIMENT_REGISTRY[name]


def clear_registry() -> None:
    """Clear all registered experiments.

    This function is mainly useful for testing purposes.
    """
    EXPERIMENT_REGISTRY.clear()
