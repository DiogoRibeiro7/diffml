"""Monte Carlo simulation engine for option pricing.

This module provides Monte Carlo simulation functionality for generating
price paths under the Black-Scholes model and computing option payoffs.
"""


import torch
from torch import Tensor

from diffml.config import DEFAULT_DTYPE, BSParams, get_device


def simulate_bs_terminal(
    spots: Tensor,
    params: BSParams,
    n_paths: int,
    seed: int | None = None
) -> tuple[Tensor, Tensor]:
    """Simulate terminal prices under Black-Scholes using exact one-step solution.

    Mathematical Derivation:
    ------------------------
    Under the risk-neutral measure Q, the stock price follows the SDE:
        dS_t = r S_t dt + σ S_t dW_t

    where:
        r = risk-free rate
        σ = volatility
        W_t = Brownian motion under Q

    Applying Itô's lemma to log(S_t):
        d(log S_t) = (r - σ²/2) dt + σ dW_t

    Integrating from 0 to T:
        log(S_T) - log(S_0) = (r - σ²/2)T + σ(W_T - W_0)
        log(S_T) = log(S_0) + (r - σ²/2)T + σ W_T

    Since W_T ~ N(0, T), we can write W_T = √T ξ where ξ ~ N(0,1):
        log(S_T) = log(S_0) + (r - σ²/2)T + σ√T ξ

    Therefore:
        S_T = S_0 exp((r - σ²/2)T + σ√T ξ)

    This is the exact solution - no discretization error!

    Parameters
    ----------
    spots : Tensor
        Initial spot prices of shape (m, 1).
    params : BSParams
        Black-Scholes parameters containing r, sigma, and T.
    n_paths : int
        Number of Monte Carlo paths to simulate per spot.
    seed : int | None
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


def simulate_bs_terminal_shared(
    spots: Tensor,
    params: BSParams,
    n_paths: int,
    seed: int | None = None
) -> tuple[Tensor, Tensor]:
    """Simulate terminal prices using shocks shared across the ``spots`` batch."""
    if spots.dim() != 2 or spots.shape[1] != 1:
        raise ValueError(f"spots must have shape (m, 1), got {spots.shape}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    if (spots <= 0).any():
        raise ValueError("All spot prices must be positive")

    device = get_device()
    spots = spots.to(device=device, dtype=DEFAULT_DTYPE)
    if seed is not None:
        torch.manual_seed(seed)

    xi = torch.randn(n_paths, device=device, dtype=DEFAULT_DTYPE)
    drift = (params.r - 0.5 * params.sigma ** 2) * params.T
    diffusion = params.sigma * torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE))
    xi_expanded = xi.unsqueeze(0)
    log_ST = torch.log(spots) + drift + diffusion * xi_expanded
    ST = torch.exp(log_ST)
    return ST, xi


def simulate_bs_two_step(
    spots: Tensor,
    params: BSParams,
    T1: float,
    T2: float,
    n_paths: int,
    seed: int | None = None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
    seed : int | None
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


def build_time_grid(T: float, n_steps: int) -> Tensor:
    """Build a uniform time grid from 0 to T.

    Creates a tensor of evenly spaced time points for discretized simulation.

    Parameters
    ----------
    T : float
        Final time (maturity).
    n_steps : int
        Number of time steps (not including t=0).

    Returns
    -------
    Tensor
        A 1D tensor of shape (n_steps + 1,) containing time points from 0 to T.

    Raises
    ------
    ValueError
        If T is negative or n_steps is not positive.

    Examples
    --------
    >>> grid = build_time_grid(T=1.0, n_steps=4)
    >>> grid
    tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
    """
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    device = get_device()
    return torch.linspace(0, T, n_steps + 1, device=device, dtype=DEFAULT_DTYPE)


def simulate_bs_paths(
    spots: Tensor,
    params: BSParams,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Simulate full Black-Scholes paths with n_steps between 0 and T.

    Uses exact Black-Scholes increments per step to generate the full price paths.
    For each step dt, the price evolves as:
    S(t+dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*xi)

    Parameters
    ----------
    spots : Tensor
        Initial spot prices of shape (m, 1).
    params : BSParams
        Black-Scholes parameters containing r, sigma, and T.
    n_steps : int
        Number of time steps to simulate (not including t=0).
    n_paths : int
        Number of Monte Carlo paths to simulate per spot.
    seed : int | None
        Random seed for reproducibility. If None, no seed is set.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple containing:
        - paths: Price paths of shape (m, n_paths, n_steps + 1), including time 0
        - xi: Standard normal increments of shape (m, n_paths, n_steps)

    Raises
    ------
    ValueError
        If spots has incorrect shape or invalid values.
        If n_paths or n_steps is not positive.

    Examples
    --------
    >>> spots = torch.tensor([[100.0], [110.0]])
    >>> params = BSParams(r=0.05, sigma=0.2, T=1.0)
    >>> paths, xi = simulate_bs_paths(spots, params, n_steps=252, n_paths=1000)
    >>> paths.shape
    torch.Size([2, 1000, 253])
    """
    # Input validation
    if spots.dim() != 2 or spots.shape[1] != 1:
        raise ValueError(f"spots must have shape (m, 1), got {spots.shape}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if (spots <= 0).any():
        raise ValueError("All spot prices must be positive")

    # Get device and ensure double precision
    device = get_device()
    spots = spots.to(device=device, dtype=DEFAULT_DTYPE)

    m = spots.shape[0]

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Build time grid
    dt = params.T / n_steps

    # Generate all standard normal increments at once
    # Shape: (m, n_paths, n_steps)
    xi = torch.randn(m, n_paths, n_steps, device=device, dtype=DEFAULT_DTYPE)

    # Pre-compute drift and diffusion per step
    drift_per_step = (params.r - 0.5 * params.sigma ** 2) * dt
    diffusion_per_step = params.sigma * torch.sqrt(torch.tensor(dt, dtype=DEFAULT_DTYPE))

    # Initialize paths tensor
    paths = torch.zeros((m, n_paths, n_steps + 1), device=device, dtype=DEFAULT_DTYPE)
    paths[:, :, 0] = spots.expand(m, n_paths)

    # Simulate paths step by step using cumulative sum of log increments
    # This is more numerically stable than multiplying prices directly
    log_spots = torch.log(spots)
    log_increments = drift_per_step + diffusion_per_step * xi
    log_cumsum = torch.cumsum(log_increments, dim=2)

    # Compute all paths at once
    paths[:, :, 1:] = torch.exp(log_spots.unsqueeze(2) + log_cumsum)

    return paths, xi
