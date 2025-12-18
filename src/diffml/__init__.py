"""DiffML: Differential Machine Learning Implementation.

A professional PyTorch implementation of experiments from the paper
"Differential ML with a Difference" by Paul Glasserman and Siddharth
Hemant Karmarkar (2025). This package demonstrates the application of
differential machine learning techniques to pricing and hedging financial
derivatives, with a focus on improving convergence speed and accuracy of
sensitivities (Greeks).

Key Components
--------------
- config: Configuration classes and utilities
- networks: Neural network architectures for option pricing
- bs_analytics: Black-Scholes analytical formulas
- simulation: Monte Carlo simulation engine
- losses: DML loss functions
- training: Training loops and utilities
- datasets_*: Dataset generators for different option types
- experiments_*: Experiment implementations from the paper

Example Usage
-------------
```python
from diffml.config import BSParams, TrainingConfig
from diffml.networks import PricingNet
from diffml.datasets_digital import make_digital_dataset
from diffml.training import train_model

# Configure Black-Scholes parameters
params = BSParams(r=0.05, sigma=0.2, T=0.25)

# Generate dataset
x_train, price_train, delta_pw, delta_lrm = make_digital_dataset(
    m=1000, K=100.0, params=params
)

# Create and train model
model = PricingNet(input_dim=1, hidden_dim=20, n_hidden=4)
config = TrainingConfig(n_epochs=1000, lambda_delta=1.0)
model = train_model(model, dataset, config, mode="delta_lrm")
```
"""

__version__ = "0.1.0"
__author__ = "Diogo Ribeiro"
__email__ = "dfr@esmad.ipp.pt"

# Core configuration and utilities
from diffml.config import (
    BSParams,
    TrainingConfig,
    get_device,
    set_default_dtype,
)

# Neural network architectures
from diffml.networks import PricingNet

# Black-Scholes analytics
from diffml.bs_analytics import (
    bs_call_gamma,
    bs_call_price,
    bs_digital_delta,
    bs_digital_price,
)

# Monte Carlo simulation
from diffml.simulation import (
    simulate_bs_terminal,
    simulate_bs_two_step,
)

# Loss functions
from diffml.losses import (
    AdaptiveDifferentialLoss,
    DifferentialLoss,
    HuberDifferentialLoss,
    dml_loss,
)

# Training utilities
from diffml.training import (
    nn_value_delta_gamma,
    rmse,
    train_model,
)

# Dataset generators
from diffml.datasets_barrier import make_barrier_dataset
from diffml.datasets_basket import make_basket_digital_dataset
from diffml.datasets_digital import make_digital_dataset
from diffml.datasets_gamma_portfolio import make_portfolio_gamma_dataset
from diffml.datasets_smoothing import make_smoothed_digital_dataset

# Experiment runners
from diffml.experiments_barrier import run_barrier_experiment
from diffml.experiments_basket import run_basket_digital_experiment
from diffml.experiments_digital import run_digital_experiment
from diffml.experiments_gamma import run_gamma_experiment
from diffml.experiments_smoothing import run_smoothing_experiment

__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",
    # Configuration
    "BSParams",
    "TrainingConfig",
    "get_device",
    "set_default_dtype",
    # Networks
    "PricingNet",
    # Analytics
    "bs_call_gamma",
    "bs_call_price",
    "bs_digital_delta",
    "bs_digital_price",
    # Simulation
    "simulate_bs_terminal",
    "simulate_bs_two_step",
    # Losses
    "AdaptiveDifferentialLoss",
    "DifferentialLoss",
    "HuberDifferentialLoss",
    "dml_loss",
    # Training
    "nn_value_delta_gamma",
    "rmse",
    "train_model",
    # Dataset generators
    "make_barrier_dataset",
    "make_basket_digital_dataset",
    "make_digital_dataset",
    "make_portfolio_gamma_dataset",
    "make_smoothed_digital_dataset",
    # Experiment runners
    "run_barrier_experiment",
    "run_basket_digital_experiment",
    "run_digital_experiment",
    "run_gamma_experiment",
    "run_smoothing_experiment",
]