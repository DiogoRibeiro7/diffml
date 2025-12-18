"""Configuration structures for experiments.

This module provides data classes and utilities for loading experiment
configurations from TOML files, enabling reproducible and configurable
experiments.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "Python < 3.11 requires 'tomli' package. "
            "Install with: pip install tomli"
        )

from diffml.config import TrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment.

    This dataclass holds all parameters needed to run an experiment,
    including training configuration, dataset parameters, and
    payoff-specific settings.

    Attributes
    ----------
    name : str
        Name of the experiment type (e.g., "digital", "barrier", "basket").
    seed : int
        Random seed for reproducibility.
    training : TrainingConfig
        Training hyperparameters.
    m_train : int
        Number of training samples.
    m_test : int
        Number of test samples.
    n_paths_train : int
        Number of Monte Carlo paths for training data.
    n_paths_test : int
        Number of Monte Carlo paths for test data.
    K : Optional[float]
        Strike price for options (if applicable).
    B : Optional[float]
        Barrier level for barrier options.
    H : Optional[float]
        Upper barrier for up-and-out or double barrier options.
    L : Optional[float]
        Lower barrier for double barrier options.
    d : Optional[int]
        Dimensionality for basket options.
    n_steps : Optional[int]
        Number of time steps for path-dependent options.
    eps_multipliers : Optional[List[float]]
        Epsilon multipliers for smoothing experiments.
    x_min : float
        Minimum spot price as fraction of K.
    x_max : float
        Maximum spot price as fraction of K.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity.
    extra_params : Dict[str, Any]
        Additional parameters not covered by standard fields.
    """

    # Required fields
    name: str
    seed: int
    training: TrainingConfig
    m_train: int
    m_test: int
    n_paths_train: int
    n_paths_test: int

    # Payoff-specific parameters (optional)
    K: Optional[float] = None
    B: Optional[float] = None
    H: Optional[float] = None
    L: Optional[float] = None
    d: Optional[int] = None
    n_steps: Optional[int] = None
    eps_multipliers: Optional[List[float]] = None

    # Grid parameters
    x_min: float = 0.5
    x_max: float = 1.5

    # Black-Scholes parameters
    r: float = 0.05
    sigma: float = 0.2
    T: float = 0.25

    # Catch-all for additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.m_train <= 0:
            raise ValueError(f"m_train must be positive, got {self.m_train}")
        if self.m_test <= 0:
            raise ValueError(f"m_test must be positive, got {self.m_test}")
        if self.n_paths_train <= 0:
            raise ValueError(f"n_paths_train must be positive, got {self.n_paths_train}")
        if self.n_paths_test <= 0:
            raise ValueError(f"n_paths_test must be positive, got {self.n_paths_test}")
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min must be less than x_max, got {self.x_min} >= {self.x_max}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load experiment configuration from a TOML file.

    Parameters
    ----------
    path : str
        Path to the TOML configuration file.

    Returns
    -------
    ExperimentConfig
        Parsed experiment configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration file doesn't exist.
    ValueError
        If the configuration is invalid or missing required fields.

    Examples
    --------
    >>> config = load_experiment_config("configs/digital_default.toml")
    >>> print(config.name)
    'digital'
    >>> print(config.training.n_epochs)
    1000
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(config_path, "rb") as f:
        toml_data = tomllib.load(f)

    # Extract main sections
    experiment_data = toml_data.get("experiment", {})
    training_data = toml_data.get("training", {})
    dataset_data = toml_data.get("dataset", {})
    payoff_data = toml_data.get("payoff", {})
    bs_params_data = toml_data.get("black_scholes", {})

    # Validate required fields
    required_fields = ["name", "seed"]
    for field in required_fields:
        if field not in experiment_data:
            raise ValueError(f"Missing required field in [experiment]: {field}")

    # Create TrainingConfig
    training_config = TrainingConfig(
        n_epochs=training_data.get("n_epochs", 2000),
        batch_size=training_data.get("batch_size", 256),
        lr_initial=training_data.get("lr_initial", 1e-3),
        lr_min=training_data.get("lr_min", 1e-6),
        lambda_delta=training_data.get("lambda_delta", 1.0),
        lambda_gamma=training_data.get("lambda_gamma", 0.0),
    )

    # Build ExperimentConfig
    config = ExperimentConfig(
        name=experiment_data["name"],
        seed=experiment_data["seed"],
        training=training_config,
        # Dataset parameters
        m_train=dataset_data.get("m_train", 1024),
        m_test=dataset_data.get("m_test", 256),
        n_paths_train=dataset_data.get("n_paths_train", 10000),
        n_paths_test=dataset_data.get("n_paths_test", 50000),
        x_min=dataset_data.get("x_min", 0.5),
        x_max=dataset_data.get("x_max", 1.5),
        # Payoff parameters
        K=payoff_data.get("K"),
        B=payoff_data.get("B"),
        H=payoff_data.get("H"),
        L=payoff_data.get("L"),
        d=payoff_data.get("d"),
        n_steps=payoff_data.get("n_steps"),
        eps_multipliers=payoff_data.get("eps_multipliers"),
        # Black-Scholes parameters
        r=bs_params_data.get("r", 0.05),
        sigma=bs_params_data.get("sigma", 0.2),
        T=bs_params_data.get("T", 0.25),
        # Store any extra parameters
        extra_params={k: v for k, v in toml_data.items()
                     if k not in ["experiment", "training", "dataset", "payoff", "black_scholes"]}
    )

    return config


def save_experiment_config(config: ExperimentConfig, path: str) -> None:
    """Save experiment configuration to a TOML file.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration to save.
    path : str
        Path where to save the TOML file.

    Notes
    -----
    This function requires the `toml` package for writing TOML files.
    Install with: pip install toml
    """
    try:
        import toml
    except ImportError:
        raise ImportError(
            "Saving TOML files requires 'toml' package. "
            "Install with: pip install toml"
        )

    config_dict = {
        "experiment": {
            "name": config.name,
            "seed": config.seed,
        },
        "training": {
            "n_epochs": config.training.n_epochs,
            "batch_size": config.training.batch_size,
            "lr_initial": config.training.lr_initial,
            "lr_final": config.training.lr_final,
            "lambda_delta": config.training.lambda_delta,
            "lambda_gamma": config.training.lambda_gamma,
        },
        "dataset": {
            "m_train": config.m_train,
            "m_test": config.m_test,
            "n_paths_train": config.n_paths_train,
            "n_paths_test": config.n_paths_test,
            "x_min": config.x_min,
            "x_max": config.x_max,
        },
        "black_scholes": {
            "r": config.r,
            "sigma": config.sigma,
            "T": config.T,
        },
    }

    # Add payoff parameters if they exist
    payoff_dict = {}
    if config.K is not None:
        payoff_dict["K"] = config.K
    if config.B is not None:
        payoff_dict["B"] = config.B
    if config.H is not None:
        payoff_dict["H"] = config.H
    if config.L is not None:
        payoff_dict["L"] = config.L
    if config.d is not None:
        payoff_dict["d"] = config.d
    if config.n_steps is not None:
        payoff_dict["n_steps"] = config.n_steps
    if config.eps_multipliers is not None:
        payoff_dict["eps_multipliers"] = config.eps_multipliers

    if payoff_dict:
        config_dict["payoff"] = payoff_dict

    # Add extra parameters
    config_dict.update(config.extra_params)

    # Save to file
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        toml.dump(config_dict, f)