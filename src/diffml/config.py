"""Configuration module for DiffML experiments.

This module provides configuration settings, hyperparameters, and utility
functions for managing experiment configurations and device selection.
"""

from dataclasses import dataclass
from typing import Optional

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
    hidden_dims: list[int] = None
    output_dim: int = 1
    activation: str = "relu"
    dropout_rate: float = 0.0

    def __post_init__(self) -> None:
        """Initialize default hidden dimensions if not provided."""
        if self.hidden_dims is None:
            self.hidden_dims = [50, 50, 50]


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
    """

    n_epochs: int = 2000
    batch_size: int = 256
    lr_initial: float = 1e-3
    lr_min: float = 1e-6
    lambda_delta: float = 0.0  # Weight for delta loss
    lambda_gamma: float = 0.0  # Weight for gamma loss


@dataclass
class SimulationConfig:
    """Monte Carlo simulation configuration.

    Parameters
    ----------
    n_paths : int
        Number of Monte Carlo paths.
    n_timesteps : int
        Number of time steps in discretization.
    seed : Optional[int]
        Random seed for reproducibility.
    antithetic : bool
        Use antithetic variates for variance reduction.
    """

    n_paths: int = 10000
    n_timesteps: int = 100
    seed: Optional[int] = 42
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
    device : Optional[str]
        Device for computation ('cuda', 'cpu', or None for auto).
    """

    name: str = "default_experiment"
    network: NetworkConfig = None
    training: TrainingConfig = None
    simulation: SimulationConfig = None
    output_dir: str = "output"
    device: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize default configurations if not provided."""
        if self.network is None:
            self.network = NetworkConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.simulation is None:
            self.simulation = SimulationConfig()


def get_device() -> torch.device:
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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
