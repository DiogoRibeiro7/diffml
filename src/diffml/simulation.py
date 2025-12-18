"""Monte Carlo simulation engine for option pricing.

This module provides Monte Carlo simulation functionality for generating
price paths under the Black-Scholes model and computing option payoffs.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from diffml.config import BSParams, DEFAULT_DTYPE, get_device


def simulate_bs_terminal(
    spots: Tensor,
    params: BSParams,
    n_paths: int,
    seed: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    """Simulate terminal prices under Black-Scholes using exact one-step solution.

    Uses the exact Black-Scholes solution for terminal price:
    ST = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*xi)

    where xi ~ N(0,1).

    Parameters
    ----------
    spots : Tensor
        Initial spot prices of shape (m, 1).
    params : BSParams
        Black-Scholes parameters containing r, sigma, and T.
    n_paths : int
        Number of Monte Carlo paths to simulate per spot.
    seed : Optional[int]
        Random seed for reproducibility. If None, no seed is set.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple containing:
        - ST: Terminal prices of shape (m, n_paths)
        - xi: Standard normal draws of shape (m, n_paths)

    Raises
    ------
    ValueError
        If spots has incorrect shape or invalid values.
        If n_paths is not positive.

    Examples
    --------
    >>> spots = torch.tensor([[100.0], [110.0]])
    >>> params = BSParams(r=0.05, sigma=0.2, T=0.25)
    >>> ST, xi = simulate_bs_terminal(spots, params, n_paths=10000)
    >>> ST.shape
    torch.Size([2, 10000])
    """
    # Input validation
    if spots.dim() != 2 or spots.shape[1] != 1:
        raise ValueError(f"spots must have shape (m, 1), got {spots.shape}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    if (spots <= 0).any():
        raise ValueError("All spot prices must be positive")

    # Get device and ensure double precision
    device = get_device()
    spots = spots.to(device=device, dtype=DEFAULT_DTYPE)

    m = spots.shape[0]

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Generate standard normal random variables
    # Shape: (m, n_paths)
    xi = torch.randn(m, n_paths, device=device, dtype=DEFAULT_DTYPE)

    # Compute drift and diffusion terms
    # drift = (r - 0.5 * sigma^2) * T
    drift = (params.r - 0.5 * params.sigma ** 2) * params.T
    # diffusion = sigma * sqrt(T)
    diffusion = params.sigma * torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE))

    # Compute terminal prices using exact BS solution
    # ST = S0 * exp(drift + diffusion * xi)
    log_ST = torch.log(spots) + drift + diffusion * xi
    ST = torch.exp(log_ST)

    return ST, xi


def simulate_bs_two_step(
    spots: Tensor,
    params: BSParams,
    T1: float,
    T2: float,
    n_paths: int,
    seed: Optional[int] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Simulate Black-Scholes prices at two time points for barrier options.

    Performs two-step simulation:
    1. From 0 to T1: S1 = S0 * exp((r - 0.5*sigma^2)*T1 + sigma*sqrt(T1)*xi1)
    2. From T1 to T2: S2 = S1 * exp((r - 0.5*sigma^2)*(T2-T1) + sigma*sqrt(T2-T1)*xi2)

    Parameters
    ----------
    spots : Tensor
        Initial spot prices of shape (m, 1).
    params : BSParams
        Black-Scholes parameters containing r and sigma.
    T1 : float
        First time point (for barrier observation).
    T2 : float
        Second time point (maturity). Must be >= T1.
    n_paths : int
        Number of Monte Carlo paths to simulate per spot.
    seed : Optional[int]
        Random seed for reproducibility. If None, no seed is set.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor]
        A tuple containing:
        - S1: Prices at T1 of shape (m, n_paths)
        - S2: Prices at T2 of shape (m, n_paths)
        - xi1: Standard normal shocks for [0, T1] of shape (m, n_paths)
        - xi2: Standard normal shocks for [T1, T2] of shape (m, n_paths)

    Raises
    ------
    ValueError
        If spots has incorrect shape or invalid values.
        If n_paths is not positive.
        If T2 < T1 or T1 < 0.

    Examples
    --------
    >>> spots = torch.tensor([[100.0]])
    >>> params = BSParams(r=0.05, sigma=0.2)
    >>> S1, S2, xi1, xi2 = simulate_bs_two_step(spots, params, T1=0.25, T2=0.5, n_paths=10000)
    >>> S1.shape, S2.shape
    (torch.Size([1, 10000]), torch.Size([1, 10000]))
    """
    # Input validation
    if spots.dim() != 2 or spots.shape[1] != 1:
        raise ValueError(f"spots must have shape (m, 1), got {spots.shape}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    if (spots <= 0).any():
        raise ValueError("All spot prices must be positive")
    if T1 < 0:
        raise ValueError(f"T1 must be non-negative, got {T1}")
    if T2 < T1:
        raise ValueError(f"T2 must be >= T1, got T2={T2}, T1={T1}")

    # Get device and ensure double precision
    device = get_device()
    spots = spots.to(device=device, dtype=DEFAULT_DTYPE)

    m = spots.shape[0]

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Generate standard normal random variables for both steps
    xi1 = torch.randn(m, n_paths, device=device, dtype=DEFAULT_DTYPE)
    xi2 = torch.randn(m, n_paths, device=device, dtype=DEFAULT_DTYPE)

    # Step 1: Simulate from 0 to T1
    if T1 > 0:
        drift1 = (params.r - 0.5 * params.sigma ** 2) * T1
        diffusion1 = params.sigma * torch.sqrt(torch.tensor(T1, dtype=DEFAULT_DTYPE))
        log_S1 = torch.log(spots) + drift1 + diffusion1 * xi1
        S1 = torch.exp(log_S1)
    else:
        # If T1 = 0, S1 = S0
        S1 = spots.expand(m, n_paths)

    # Step 2: Simulate from T1 to T2
    dt = T2 - T1
    if dt > 0:
        drift2 = (params.r - 0.5 * params.sigma ** 2) * dt
        diffusion2 = params.sigma * torch.sqrt(torch.tensor(dt, dtype=DEFAULT_DTYPE))
        log_S2 = torch.log(S1) + drift2 + diffusion2 * xi2
        S2 = torch.exp(log_S2)
    else:
        # If T2 = T1, S2 = S1
        S2 = S1

    return S1, S2, xi1, xi2