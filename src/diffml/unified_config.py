"""Unified configuration module for DiffML.

This module consolidates all configuration logic to eliminate duplication
between the core diffml and article replication packages.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Dict, List, Union
from abc import ABC, abstractmethod

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError as exc:
        raise ImportError(
            "Python < 3.11 requires 'tomli' package. Install with: pip install tomli"
        ) from exc

import torch

# Re-export for backward compatibility
from .config import (
    BSParams,
    NetworkConfig,
    TrainingConfig,
    SimulationConfig,
    DEFAULT_DTYPE,
    get_device,
)

# Import validation if available
try:
    from .config_validation import (
        ExperimentConfigSchema,
        validate_toml_config,
        load_and_validate_config as validate_config_file,
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False


@dataclass
class UnifiedExperimentConfig:
    """Unified experiment configuration for both core and article replication.

    This class consolidates the duplicate ExperimentConfig classes and provides
    a single source of truth for experiment configurations.

    Attributes:
        name: Name of the experiment type (e.g., "digital", "barrier", "basket")
        seed: Random seed for reproducibility
        training: Training hyperparameters
        m_train: Number of training samples
        m_test: Number of test samples
        n_paths_train: Number of Monte Carlo paths for training data
        n_paths_test: Number of Monte Carlo paths for test data
        K: Strike price for options (if applicable)
        B: Barrier level for barrier options
        H: Upper barrier for up-and-out or double barrier options
        L: Lower barrier for double barrier options
        d: Dimensionality for basket options
        n_steps: Number of time steps for path-dependent options
        eps_multipliers: Epsilon multipliers for smoothing experiments
        x_min: Minimum spot price as fraction of K
        x_max: Maximum spot price as fraction of K
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        extra_params: Additional parameters not covered by standard fields
        validate_on_init: Whether to validate parameters on initialization
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

    # Market parameters
    r: float = 0.05
    sigma: float = 0.2
    T: float = 0.25

    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # Validation flag
    validate_on_init: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.validate_on_init:
            self.validate()

    def validate(self) -> None:
        """Validate key numeric parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        # Basic validation
        if self.m_train <= 0:
            raise ValueError(f"m_train must be positive, got {self.m_train}")
        if self.m_test <= 0:
            raise ValueError(f"m_test must be positive, got {self.m_test}")
        if self.n_paths_train <= 0:
            raise ValueError(f"n_paths_train must be positive, got {self.n_paths_train}")
        if self.n_paths_test <= 0:
            raise ValueError(f"n_paths_test must be positive, got {self.n_paths_test}")

        # Grid validation
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be < x_max ({self.x_max})")

        # Market parameters validation
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.r < 0:
            raise ValueError(f"r must be non-negative, got {self.r}")

        # Option-specific validation
        if self.K is not None and self.K <= 0:
            raise ValueError(f"Strike K must be positive, got {self.K}")
        if self.d is not None and self.d <= 0:
            raise ValueError(f"Dimension d must be positive, got {self.d}")
        if self.n_steps is not None and self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")

        # Barrier validation
        if self.B is not None and self.B <= 0:
            raise ValueError(f"Barrier B must be positive, got {self.B}")
        if self.H is not None and self.H <= 0:
            raise ValueError(f"Upper barrier H must be positive, got {self.H}")
        if self.L is not None and self.L <= 0:
            raise ValueError(f"Lower barrier L must be positive, got {self.L}")

        # Double barrier consistency
        if self.L is not None and self.H is not None and self.L >= self.H:
            raise ValueError(f"Lower barrier L ({self.L}) must be < upper barrier H ({self.H})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        config_dict = asdict(self)
        # Convert TrainingConfig to dict if present
        if isinstance(config_dict.get('training'), TrainingConfig):
            config_dict['training'] = asdict(config_dict['training'])
        return config_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedExperimentConfig':
        """Create configuration from dictionary.

        Parameters:
            data: Dictionary containing configuration parameters

        Returns:
            UnifiedExperimentConfig instance
        """
        # Handle TrainingConfig
        if 'training' in data and isinstance(data['training'], dict):
            data['training'] = TrainingConfig(**data['training'])
        return cls(**data)

    def clone(self, **updates) -> 'UnifiedExperimentConfig':
        """Create a copy with optional updates.

        Parameters:
            **updates: Fields to update in the copy

        Returns:
            New UnifiedExperimentConfig instance with updates applied
        """
        config_dict = self.to_dict()
        config_dict.update(updates)
        return self.from_dict(config_dict)

    def get_bs_params(self) -> BSParams:
        """Extract Black-Scholes parameters.

        Returns:
            BSParams instance with current market parameters
        """
        return BSParams(r=self.r, sigma=self.sigma, T=self.T)

    def get_network_config(self, input_dim: int = 1, output_dim: int = 1) -> NetworkConfig:
        """Extract network configuration from training config.

        Parameters:
            input_dim: Input dimension for the network
            output_dim: Output dimension for the network

        Returns:
            NetworkConfig instance
        """
        # Extract from training config if available
        hidden_dims = self.extra_params.get('hidden_dims', [50, 50, 50])
        activation = self.extra_params.get('activation', 'relu')
        dropout_rate = self.extra_params.get('dropout_rate', 0.0)

        return NetworkConfig(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout_rate=dropout_rate
        )


# Alias for backward compatibility
ExperimentConfig = UnifiedExperimentConfig


class ConfigLoader(ABC):
    """Abstract base class for configuration loaders."""

    @abstractmethod
    def load(self, source: Any) -> UnifiedExperimentConfig:
        """Load configuration from source.

        Parameters:
            source: Configuration source (file path, dict, etc.)

        Returns:
            Loaded configuration
        """
        pass


class TOMLConfigLoader(ConfigLoader):
    """Loader for TOML configuration files."""

    def __init__(self, validate: bool = True):
        """Initialize TOML loader.

        Parameters:
            validate: Whether to validate loaded configurations
        """
        self.validate = validate

    def load(self, source: Union[str, Path]) -> UnifiedExperimentConfig:
        """Load configuration from TOML file.

        Parameters:
            source: Path to TOML file

        Returns:
            Loaded configuration

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If configuration is invalid
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open("rb") as f:
            data = tomllib.load(f)

        # Extract experiment section
        experiment_data = data.get("experiment", data)

        # Handle training section
        if "training" in experiment_data:
            training_data = experiment_data["training"]
            experiment_data["training"] = TrainingConfig(**training_data)

        # Create config
        config = UnifiedExperimentConfig.from_dict(experiment_data)

        # Optional validation with Pydantic if available
        if self.validate and VALIDATION_AVAILABLE:
            try:
                validate_toml_config(data)
            except Exception as e:
                print(f"Warning: Pydantic validation failed: {e}")

        return config


class DictConfigLoader(ConfigLoader):
    """Loader for dictionary configurations."""

    def load(self, source: Dict[str, Any]) -> UnifiedExperimentConfig:
        """Load configuration from dictionary.

        Parameters:
            source: Configuration dictionary

        Returns:
            Loaded configuration
        """
        return UnifiedExperimentConfig.from_dict(source)


class ConfigFactory:
    """Factory for creating configuration loaders."""

    _loaders = {
        'toml': TOMLConfigLoader,
        'dict': DictConfigLoader,
    }

    @classmethod
    def get_loader(cls, loader_type: str, **kwargs) -> ConfigLoader:
        """Get configuration loader by type.

        Parameters:
            loader_type: Type of loader ('toml', 'dict')
            **kwargs: Additional arguments for loader initialization

        Returns:
            ConfigLoader instance

        Raises:
            ValueError: If loader type is not supported
        """
        if loader_type not in cls._loaders:
            raise ValueError(f"Unknown loader type: {loader_type}")
        return cls._loaders[loader_type](**kwargs)

    @classmethod
    def register_loader(cls, name: str, loader_class: type[ConfigLoader]) -> None:
        """Register a new configuration loader.

        Parameters:
            name: Name for the loader type
            loader_class: Loader class to register
        """
        cls._loaders[name] = loader_class


def load_experiment_config(
    source: Union[str, Path, Dict[str, Any]],
    loader_type: Optional[str] = None,
    **kwargs
) -> UnifiedExperimentConfig:
    """Load experiment configuration from various sources.

    This is the main entry point for loading configurations, automatically
    detecting the source type if not specified.

    Parameters:
        source: Configuration source (file path or dictionary)
        loader_type: Type of loader to use (auto-detected if None)
        **kwargs: Additional arguments for loader

    Returns:
        Loaded configuration

    Examples:
        >>> # Load from TOML file
        >>> config = load_experiment_config("experiment.toml")
        >>>
        >>> # Load from dictionary
        >>> config_dict = {"name": "test", "seed": 42, ...}
        >>> config = load_experiment_config(config_dict)
    """
    # Auto-detect loader type if not specified
    if loader_type is None:
        if isinstance(source, dict):
            loader_type = 'dict'
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() in ['.toml', '.tml']:
                loader_type = 'toml'
            else:
                # Default to TOML for string paths
                loader_type = 'toml'
        else:
            raise ValueError(f"Cannot auto-detect loader type for {type(source)}")

    # Get loader and load configuration
    loader = ConfigFactory.get_loader(loader_type, **kwargs)
    return loader.load(source)


def create_default_config(experiment_type: str) -> UnifiedExperimentConfig:
    """Create a default configuration for a given experiment type.

    Parameters:
        experiment_type: Type of experiment ('digital', 'barrier', 'basket', etc.)

    Returns:
        Default configuration for the experiment type
    """
    defaults = {
        'digital': {
            'name': 'digital',
            'K': 100.0,
            'n_paths_train': 10000,
            'n_paths_test': 100000,
        },
        'barrier': {
            'name': 'barrier',
            'K': 100.0,
            'B': 90.0,
            'n_paths_train': 10000,
            'n_paths_test': 100000,
            'n_steps': 100,
        },
        'basket': {
            'name': 'basket',
            'K': 100.0,
            'd': 20,
            'n_paths_train': 10000,
            'n_paths_test': 100000,
        },
        'american': {
            'name': 'american',
            'K': 100.0,
            'n_paths_train': 5000,
            'n_paths_test': 50000,
            'n_steps': 50,
        },
        'multibarrier': {
            'name': 'multibarrier',
            'K': 100.0,
            'L': 85.0,
            'H': 115.0,
            'n_paths_train': 10000,
            'n_paths_test': 100000,
            'n_steps': 100,
        },
    }

    if experiment_type not in defaults:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    # Base configuration
    config_data = {
        'seed': 42,
        'training': TrainingConfig(),
        'm_train': 10000,
        'm_test': 2000,
        'n_paths_train': 1000,
        'n_paths_test': 10000,
    }

    # Update with experiment-specific defaults
    config_data.update(defaults[experiment_type])

    return UnifiedExperimentConfig.from_dict(config_data)


# Backward compatibility exports
__all__ = [
    'UnifiedExperimentConfig',
    'ExperimentConfig',  # Alias for backward compatibility
    'BSParams',
    'NetworkConfig',
    'TrainingConfig',
    'SimulationConfig',
    'ConfigLoader',
    'TOMLConfigLoader',
    'DictConfigLoader',
    'ConfigFactory',
    'load_experiment_config',
    'create_default_config',
    'DEFAULT_DTYPE',
    'get_device',
]