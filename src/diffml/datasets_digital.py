"""Dataset generation for digital option experiments.

This module provides functions to generate training and validation datasets
for digital option pricing using differential machine learning.
"""


from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from diffml.config import DEFAULT_DTYPE, BSParams, get_device
from diffml.simulation import simulate_bs_terminal, simulate_bs_terminal_shared

DigitalSample = tuple[Tensor, Tensor, Tensor, Tensor]


def make_digital_dataset(
    m: int,
    K: float,
    params: BSParams,
    x_min: float = 40.0,
    x_max: float = 160.0,
    n_paths_per_x: int = 10,
    seed: int | None = 1234,
    use_shared_paths: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate dataset for 1D digital call option under Black-Scholes.

    Creates a dataset with spot price inputs and corresponding digital call
    option prices and sensitivities (deltas). Uses Monte Carlo simulation
    for pricing and likelihood ratio method for delta estimation.

    For a digital call with payoff 1_{ST > K}:
    - Price: E[e^(-rT) * 1_{ST > K}]
    - Pathwise delta: 0 (discontinuous payoff)
    - LRM delta: E[e^(-rT) * 1_{ST > K} * xi / (S0 * sigma * sqrt(T))]

    Parameters
    ----------
    m : int
        Number of spot price points in the dataset.
    K : float
        Strike price of the digital option.
    params : BSParams
        Black-Scholes parameters (r, sigma, T).
    x_min : float, optional
        Minimum spot price. Default is 40.0.
    x_max : float, optional
        Maximum spot price. Default is 160.0.
    n_paths_per_x : int, optional
        Number of Monte Carlo paths per spot price. Default is 10.
    seed : int | None, optional
        Random seed for reproducibility. Default is 1234.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor]
        A tuple containing:
        - x: Spot prices of shape (m, 1)
        - price_label: Monte Carlo prices of shape (m, 1)
        - delta_pathwise: Pathwise deltas (zeros) of shape (m, 1)
        - delta_lrm: Likelihood ratio method deltas of shape (m, 1)

    Raises
    ------
    ValueError
        If m <= 0 or n_paths_per_x <= 0.
        If x_min >= x_max or K <= 0.

    Examples
    --------
    >>> params = BSParams(r=0.05, sigma=0.2, T=0.25)
    >>> x, prices, delta_pw, delta_lrm = make_digital_dataset(
    ...     m=100, K=100.0, params=params, n_paths_per_x=10000
    ... )
    >>> x.shape, prices.shape
    (torch.Size([100, 1]), torch.Size([100, 1]))
    """
    # Input validation
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if n_paths_per_x <= 0:
        raise ValueError(f"n_paths_per_x must be positive, got {n_paths_per_x}")
    if x_min >= x_max:
        raise ValueError(f"x_min must be less than x_max, got x_min={x_min}, x_max={x_max}")
    if K <= 0:
        raise ValueError(f"Strike K must be positive, got {K}")

    # Get device and set precision
    device = get_device()

    # Build grid of spot prices
    # Shape: (m, 1)
    x = torch.linspace(x_min, x_max, m, device=device, dtype=DEFAULT_DTYPE).reshape(m, 1)

    if use_shared_paths:
        ST, xi_shared = simulate_bs_terminal_shared(x, params, n_paths_per_x, seed=seed)
        xi = xi_shared.unsqueeze(0).expand_as(ST)
    else:
        ST, xi = simulate_bs_terminal(x, params, n_paths_per_x, seed=seed)

    # Compute discount factor
    discount = torch.exp(
        torch.tensor(-params.r * params.T, dtype=DEFAULT_DTYPE, device=device)
    )

    # Digital payoff: 1_{ST > K}
    # Shape: (m, n_paths_per_x)
    payoff = (ST > K).to(dtype=DEFAULT_DTYPE)

    # Discounted payoff
    disc_payoff = discount * payoff

    # Price label: Monte Carlo mean over paths
    # Shape: (m, 1)
    price_label = disc_payoff.mean(dim=1, keepdim=True)

    # Pathwise delta: exactly zero for discontinuous payoff
    # Shape: (m, 1)
    delta_pathwise = torch.zeros_like(price_label)

    # Likelihood Ratio Method (LRM) delta
    # Score function: xi / (S0 * sigma * sqrt(T))
    # LRM estimator: E[payoff * score]
    sqrt_T = torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE, device=device))
    score = xi / (x * params.sigma * sqrt_T)  # Broadcasting: x is (m, 1), xi is (m, n_paths)

    # LRM delta: mean of discounted payoff times score
    # Shape: (m, 1)
    delta_lrm = (disc_payoff * score).mean(dim=1, keepdim=True)

    return x, price_label, delta_pathwise, delta_lrm


class DigitalOptionDataset(Dataset[DigitalSample]):
    """PyTorch Dataset for digital option pricing.

    Parameters
    ----------
    x : Tensor
        Spot prices of shape (m, 1).
    price_label : Tensor
        Option prices of shape (m, 1).
    delta_pathwise : Tensor
        Pathwise deltas of shape (m, 1).
    delta_lrm : Tensor
        LRM deltas of shape (m, 1).
    """

    def __init__(
        self,
        x: Tensor,
        price_label: Tensor,
        delta_pathwise: Tensor,
        delta_lrm: Tensor
    ) -> None:
        """Initialize the digital option dataset."""
        self.x = x
        self.price_label = price_label
        self.delta_pathwise = delta_pathwise
        self.delta_lrm = delta_lrm
        self.n_samples = x.shape[0]

    def __len__(self) -> int:
        """Get dataset size.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self.n_samples

    def __getitem__(self, idx: int) -> DigitalSample:
        """Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor]
            (spot, price, delta_pw, delta_lrm) for the sample.
        """
        return (
            self.x[idx],
            self.price_label[idx],
            self.delta_pathwise[idx],
            self.delta_lrm[idx]
        )


def create_digital_dataloaders(
    n_train: int = 50000,
    n_val: int = 10000,
    batch_size: int = 256,
    K: float = 100.0,
    params: BSParams | None = None,
    n_paths_per_x: int = 10000,
    **kwargs: Any,
) -> tuple[DataLoader[DigitalSample], DataLoader[DigitalSample]]:
    """Create training and validation dataloaders for digital options.

    Parameters
    ----------
    n_train : int
        Number of training samples.
    n_val : int
        Number of validation samples.
    batch_size : int
        Batch size for dataloaders.
    K : float
        Strike price.
    params : BSParams | None
        Black-Scholes parameters. If None, uses defaults.
    n_paths_per_x : int
        Number of MC paths per spot price.
    **kwargs
        Additional arguments for make_digital_dataset.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training and validation dataloaders.
    """
    if params is None:
        params = BSParams()

    # Generate training data
    x_train, price_train, delta_pw_train, delta_lrm_train = make_digital_dataset(
        m=n_train,
        K=K,
        params=params,
        n_paths_per_x=n_paths_per_x,
        seed=42,
        **kwargs
    )

    # Generate validation data
    x_val, price_val, delta_pw_val, delta_lrm_val = make_digital_dataset(
        m=n_val,
        K=K,
        params=params,
        n_paths_per_x=n_paths_per_x,
        seed=123,
        **kwargs
    )

    # Create datasets
    train_dataset = DigitalOptionDataset(
        x_train, price_train, delta_pw_train, delta_lrm_train
    )
    val_dataset = DigitalOptionDataset(
        x_val, price_val, delta_pw_val, delta_lrm_val
    )

    # Create dataloaders
    train_loader: DataLoader[DigitalSample] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader: DataLoader[DigitalSample] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader
