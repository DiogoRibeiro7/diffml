"""Dataset generation for barrier option experiments.

This module provides functions to generate training and validation datasets
for barrier option pricing using differential machine learning.
"""

from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from diffml.config import DEFAULT_DTYPE, BSParams, get_device
from diffml.simulation import simulate_bs_two_step


def make_barrier_dataset(
    m: int,
    K: float,
    B: float,
    params: BSParams,
    T1: float,
    T2: float,
    x_min: float = 0.4,
    x_max: float = 1.6,
    n_paths_per_x: int = 10,
    seed: Optional[int] = 1234
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate dataset for down-and-out call barrier option under Black-Scholes.

    Creates a dataset for a down-and-out call option where the option knocks out
    if the price falls below barrier B at time T1. The option expires at T2.

    Payoff: e^(-rT2) * 1_{S1 > B} * max(S2 - K, 0)

    Parameters
    ----------
    m : int
        Number of spot price points in the dataset.
    K : float
        Strike price of the option.
    B : float
        Barrier level (down-and-out).
    params : BSParams
        Black-Scholes parameters (r, sigma).
    T1 : float
        Time point for barrier observation.
    T2 : float
        Maturity of the option.
    x_min : float, optional
        Minimum spot price. Default is 0.4.
    x_max : float, optional
        Maximum spot price. Default is 1.6.
    n_paths_per_x : int, optional
        Number of Monte Carlo paths per spot price. Default is 10.
    seed : Optional[int], optional
        Random seed for reproducibility. Default is 1234.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor]
        A tuple containing:
        - x: Spot prices of shape (m, 1)
        - price_label: Monte Carlo prices of shape (m, 1)
        - delta_pathwise: Pathwise deltas of shape (m, 1)
        - delta_lrm: Likelihood ratio method deltas of shape (m, 1)

    Raises
    ------
    ValueError
        If m <= 0 or n_paths_per_x <= 0.
        If x_min >= x_max or K <= 0 or B <= 0.
        If T2 < T1 or T1 < 0.

    Examples
    --------
    >>> params = BSParams(r=0.05, sigma=0.2)
    >>> x, prices, delta_pw, delta_lrm = make_barrier_dataset(
    ...     m=100, K=1.0, B=0.8, params=params, T1=0.25, T2=0.5, n_paths_per_x=10000
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
    if B <= 0:
        raise ValueError(f"Barrier B must be positive, got {B}")
    if T1 < 0:
        raise ValueError(f"T1 must be non-negative, got {T1}")
    if T2 < T1:
        raise ValueError(f"T2 must be >= T1, got T2={T2}, T1={T1}")

    # Get device and set precision
    device = get_device()

    # Build grid of spot prices
    # Shape: (m, 1)
    x = torch.linspace(x_min, x_max, m, device=device, dtype=DEFAULT_DTYPE).reshape(m, 1)

    # Simulate two-step paths
    # S1, S2 shape: (m, n_paths_per_x)
    # xi1, xi2 shape: (m, n_paths_per_x)
    S1, S2, xi1, xi2 = simulate_bs_two_step(x, params, T1, T2, n_paths_per_x, seed=seed)

    # Compute discount factor for maturity T2
    discount = torch.exp(-params.r * T2)

    # Barrier indicator: option survives if S1 > B
    # Shape: (m, n_paths_per_x)
    survived = (S1 > B).to(dtype=DEFAULT_DTYPE)

    # Call payoff at T2: max(S2 - K, 0)
    # Shape: (m, n_paths_per_x)
    call_payoff = torch.maximum(S2 - K, torch.tensor(0.0, dtype=DEFAULT_DTYPE))

    # Down-and-out call payoff
    # Shape: (m, n_paths_per_x)
    payoff = survived * call_payoff
    disc_payoff = discount * payoff

    # Price label: Monte Carlo mean over paths
    # Shape: (m, 1)
    price_label = disc_payoff.mean(dim=1, keepdim=True)

    # Pathwise delta per path
    # For a down-and-out call: delta = disc * 1_{S1 > B} * 1_{S2 > K} * (S2 / x)
    # The S2/x term comes from the lognormal chain rule: d(S2)/d(S0) = S2/S0
    # Shape: (m, n_paths_per_x)
    in_the_money = (S2 > K).to(dtype=DEFAULT_DTYPE)
    delta_paths = discount * survived * in_the_money * (S2 / x)

    # Pathwise delta label: mean over paths
    # Shape: (m, 1)
    delta_pathwise = delta_paths.mean(dim=1, keepdim=True)

    # Likelihood Ratio Method (LRM) delta
    # Use the first-step shock xi1 with weight xi1 / (x * sigma * sqrt(T1))
    # LRM estimator: E[payoff * score]
    if T1 > 0:
        sqrt_T1 = torch.sqrt(torch.tensor(T1, dtype=DEFAULT_DTYPE))
        score = xi1 / (x * params.sigma * sqrt_T1)  # Broadcasting: x is (m, 1), xi1 is (m, n_paths)
    else:
        # If T1 = 0, no first step, use zero score
        score = torch.zeros_like(xi1)

    # LRM delta: mean of discounted payoff times score
    # Shape: (m, 1)
    delta_lrm = (disc_payoff * score).mean(dim=1, keepdim=True)

    return x, price_label, delta_pathwise, delta_lrm


class BarrierOptionDataset(Dataset):
    """PyTorch Dataset for barrier option pricing.

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
        """Initialize the barrier option dataset."""
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

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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


def create_barrier_dataloaders(
    n_train: int = 50000,
    n_val: int = 10000,
    batch_size: int = 256,
    K: float = 1.0,
    B: float = 0.8,
    T1: float = 0.25,
    T2: float = 0.5,
    params: Optional[BSParams] = None,
    n_paths_per_x: int = 10000,
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders for barrier options.

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
    B : float
        Barrier level.
    T1 : float
        Barrier observation time.
    T2 : float
        Maturity.
    params : Optional[BSParams]
        Black-Scholes parameters. If None, uses defaults.
    n_paths_per_x : int
        Number of MC paths per spot price.
    **kwargs
        Additional arguments for make_barrier_dataset.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training and validation dataloaders.
    """
    if params is None:
        params = BSParams()

    # Generate training data
    x_train, price_train, delta_pw_train, delta_lrm_train = make_barrier_dataset(
        m=n_train,
        K=K,
        B=B,
        params=params,
        T1=T1,
        T2=T2,
        n_paths_per_x=n_paths_per_x,
        seed=42,
        **kwargs
    )

    # Generate validation data
    x_val, price_val, delta_pw_val, delta_lrm_val = make_barrier_dataset(
        m=n_val,
        K=K,
        B=B,
        params=params,
        T1=T1,
        T2=T2,
        n_paths_per_x=n_paths_per_x,
        seed=123,
        **kwargs
    )

    # Create datasets
    train_dataset = BarrierOptionDataset(
        x_train, price_train, delta_pw_train, delta_lrm_train
    )
    val_dataset = BarrierOptionDataset(
        x_val, price_val, delta_pw_val, delta_lrm_val
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader
