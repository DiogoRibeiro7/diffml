"""Configuration module for DiffML experiments.

This module provides configuration settings, hyperparameters, and utility
functions for managing experiment configurations and device selection.
"""

from dataclasses import dataclass, field

import torch

# Module-level constant for default precision
DEFAULT_DTYPE = torch.float64


@dataclass
class BSParams:
    """Black-Scholes model parameters.

    Parameters
    ----------
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility (annualized).
    T : float
        Time to maturity in years.
    """

    r: float = 0.0  # Risk-free rate
    sigma: float = 0.20  # Volatility (20% annualized)
    T: float = 1.0 / 3.0  # 4 months maturity


@dataclass
class NetworkConfig:
    """Neural network architecture configuration.

    Parameters
    ----------
    input_dim : int
        Input dimension of the network.
    hidden_dims : list[int]
        List of hidden layer dimensions.
    output_dim : int
        Output dimension of the network.
    activation : str
        Activation function name ('relu', 'tanh', 'sigmoid').
    dropout_rate : float
        Dropout rate for regularization.
    """

    input_dim: int = 1
    hidden_dims: list[int] = field(default_factory=lambda: [50, 50, 50])
    output_dim: int = 1
    activation: str = "relu"
    dropout_rate: float = 0.0


@dataclass
class TrainingConfig:
    """Training configuration for differential ML experiments.

    Parameters
    ----------
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    lr_initial : float
        Initial learning rate.
    lr_min : float
        Minimum learning rate for scheduler.
    lambda_delta : float
        Weight for delta (first derivative) loss term.
    lambda_gamma : float
        Weight for gamma (second derivative) loss term.
    use_mixed_precision : bool
        Enable CUDA automatic mixed precision during training.
    """

    n_epochs: int = 2000
    batch_size: int = 256
    lr_initial: float = 1e-3
    lr_min: float = 1e-6
    lambda_delta: float = 0.0  # Weight for delta loss
    lambda_gamma: float = 0.0  # Weight for gamma loss
    use_mixed_precision: bool = False  # Enable optional mixed precision on CUDA

    @property
    def learning_rate(self) -> float:
        """Backward-compatible alias for lr_initial."""
        return self.lr_initial

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.lr_initial = value


@dataclass
class SimulationConfig:
    """Monte Carlo simulation configuration.

    Parameters
    ----------
    n_paths : int
        Number of Monte Carlo paths.
    n_timesteps : int
        Number of time steps in discretization.
    seed : int | None
        Random seed for reproducibility.
    antithetic : bool
        Use antithetic variates for variance reduction.
    """

    n_paths: int = 10000
    n_timesteps: int = 100
    seed: int | None = 42
    antithetic: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration.

    Parameters
    ----------
    name : str
        Experiment name.
    network : NetworkConfig
        Network architecture configuration.
    training : TrainingConfig
        Training hyperparameters.
    simulation : SimulationConfig
        Monte Carlo simulation settings.
    output_dir : str
        Directory for saving outputs.
    device : str | None
        Device for computation ('cuda', 'cpu', or None for auto).
    """

    name: str = "default_experiment"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    output_dir: str = "output"
    device: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.network is None:
            self.network = NetworkConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.simulation is None:
            self.simulation = SimulationConfig()


def get_device(preferred: str | None = None) -> torch.device:
    """Get the appropriate torch device for computation.

    Returns CUDA device if available, otherwise CPU device.

    Returns
    -------
    torch.device
        The selected torch device (cuda if available, else cpu).

    Examples
    --------
    >>> device = get_device()
    >>> print(device)
    cuda  # or cpu if CUDA not available
    """
    if preferred is None or preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalized = preferred.lower()
    if normalized == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if normalized == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown device preference: {preferred}")


def set_default_dtype() -> None:
    """Set the default tensor dtype to DEFAULT_DTYPE (float64).

    This ensures high precision for financial computations where
    numerical accuracy is critical.
    """
    torch.set_default_dtype(DEFAULT_DTYPE)


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
