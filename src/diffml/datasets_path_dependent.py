"""Dataset generators for path-dependent options.

This module provides dataset generation functions for path-dependent options
including arithmetic Asian options and lookback options, using Monte Carlo
simulation with both pathwise and likelihood ratio method (LRM) for
sensitivity estimation.
"""

from typing import Optional

import torch
from torch import Tensor

from diffml.config import DEFAULT_DTYPE, BSParams, get_device
from diffml.simulation import simulate_bs_paths


def make_arithmetic_asian_call_dataset(
    m: int,
    K: float,
    params: BSParams,
    n_steps: int = 16,
    x_min: float = 0.5,
    x_max: float = 1.5,
    n_paths_per_x: int = 10,
    seed: Optional[int] = 1234,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate dataset for arithmetic Asian call option pricing.

    An arithmetic Asian call has payoff max(A - K, 0) where A is the arithmetic
    average of the stock price over the life of the option. We compute the average
    including time 0 (so we have n_steps + 1 points in the average).

    Parameters
    ----------
    m : int
        Number of initial spot prices to generate.
    K : float
        Strike price.
    params : BSParams
        Black-Scholes parameters (r, sigma, T).
    n_steps : int, optional
        Number of time steps for path discretization. Default is 16.
    x_min : float, optional
        Minimum initial spot price as fraction of K. Default is 0.5.
    x_max : float, optional
        Maximum initial spot price as fraction of K. Default is 1.5.
    n_paths_per_x : int, optional
        Number of Monte Carlo paths per initial spot. Default is 10.
    seed : Optional[int], optional
        Random seed for reproducibility. Default is 1234.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor]
        A tuple containing:
        - x: Initial spot prices of shape (m, 1)
        - price_label: MC price estimates of shape (m, 1)
        - delta_pathwise: Pathwise delta estimates of shape (m, 1)
        - delta_lrm: LRM delta estimates of shape (m, 1)

    Notes
    -----
    The pathwise delta is computed using the chain rule:
    dV/dS0 = E[1_{A > K} * dA/dS0]
    where dA/dS0 = (1/(n_steps+1)) * sum_{i=0}^{n_steps} dS_i/dS0
    and dS_i/dS0 = S_i/S0 (from the multiplicative nature of BS dynamics).

    The LRM delta uses the likelihood ratio:
    dV/dS0 = E[payoff * score]
    where score = sum_{i=1}^{n_steps} xi_i / (S0 * sigma * sqrt(T))
    This is derived from the log-likelihood of the full path.

    References
    ----------
    Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering, Chapter 7.
    """
    # Input validation
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if n_paths_per_x <= 0:
        raise ValueError(f"n_paths_per_x must be positive, got {n_paths_per_x}")

    device = get_device()

    # Generate initial spot prices grid
    x = torch.linspace(x_min * K, x_max * K, m, device=device, dtype=DEFAULT_DTYPE)
    x = x.reshape(-1, 1)

    # Simulate paths
    paths, xi = simulate_bs_paths(x, params, n_steps, n_paths_per_x, seed)
    # paths shape: (m, n_paths_per_x, n_steps + 1)

    # Compute arithmetic average including time 0
    # Average over the time dimension (last dimension)
    avg_prices = paths.mean(dim=2)  # Shape: (m, n_paths_per_x)

    # Compute discounted payoff
    discount_factor = torch.exp(torch.tensor(-params.r * params.T, dtype=DEFAULT_DTYPE))
    payoffs = discount_factor * torch.maximum(
        avg_prices - K,
        torch.zeros_like(avg_prices)
    )

    # Price label: MC average
    price_label = payoffs.mean(dim=1, keepdim=True)  # Shape: (m, 1)

    # Pathwise delta
    # For arithmetic Asian: dA/dS0 = (1/(n_steps+1)) * sum_{i=0}^{n_steps} S_i/S0
    # Since S_i/S0 follows a lognormal distribution, we have dS_i/dS0 = S_i/S0
    # Therefore: dA/dS0 = avg(paths) / S0
    indicator = (avg_prices > K).to(dtype=DEFAULT_DTYPE)  # Shape: (m, n_paths_per_x)

    # Compute dA/dS0 for each path
    # Average price derivative w.r.t. S0
    dA_dS0 = avg_prices / x  # Shape: (m, n_paths_per_x)

    # Pathwise delta = E[1_{A > K} * dA/dS0]
    delta_pathwise = discount_factor * (indicator * dA_dS0).mean(dim=1, keepdim=True)

    # LRM delta
    # Score function for the full path: sum of all xi_i normalized
    # score = sum_{i=1}^{n_steps} xi_i / (S0 * sigma * sqrt(dt))
    # where dt = T / n_steps
    dt = params.T / n_steps
    normalizer = x * params.sigma * torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE))

    # Sum all increments: xi has shape (m, n_paths_per_x, n_steps)
    xi_sum = xi.sum(dim=2)  # Shape: (m, n_paths_per_x)
    score = xi_sum / normalizer  # Shape: (m, n_paths_per_x)

    # LRM delta = E[payoff * score]
    delta_lrm = (payoffs * score).mean(dim=1, keepdim=True)

    return x, price_label, delta_pathwise, delta_lrm


def make_lookback_call_dataset(
    m: int,
    K: float,
    params: BSParams,
    n_steps: int = 32,
    x_min: float = 0.5,
    x_max: float = 1.5,
    n_paths_per_x: int = 10,
    seed: Optional[int] = 1234,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate dataset for fixed-strike lookback call option pricing.

    A fixed-strike lookback call has payoff max(M - K, 0) where M = max_{t} S_t
    is the maximum stock price over the life of the option.

    Parameters
    ----------
    m : int
        Number of initial spot prices to generate.
    K : float
        Strike price.
    params : BSParams
        Black-Scholes parameters (r, sigma, T).
    n_steps : int, optional
        Number of time steps for path discretization. Default is 32.
    x_min : float, optional
        Minimum initial spot price as fraction of K. Default is 0.5.
    x_max : float, optional
        Maximum initial spot price as fraction of K. Default is 1.5.
    n_paths_per_x : int, optional
        Number of Monte Carlo paths per initial spot. Default is 10.
    seed : Optional[int], optional
        Random seed for reproducibility. Default is 1234.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor]
        A tuple containing:
        - x: Initial spot prices of shape (m, 1)
        - price_label: MC price estimates of shape (m, 1)
        - delta_pathwise: Pathwise delta estimates of shape (m, 1)
        - delta_lrm: LRM delta estimates of shape (m, 1)

    Notes
    -----
    The pathwise delta is computed as:
    dV/dS0 = E[1_{M > K} * dM/dS0]
    where dM/dS0 depends on which S_i achieves the maximum.
    For the path where S_tau = M (tau is the time of maximum),
    we have dM/dS0 = S_tau/S0 = M/S0.

    The LRM delta uses:
    dV/dS0 = E[payoff * score]
    where score = sum_{i=1}^{n_steps} xi_i / (S0 * sigma * sqrt(T))

    This is an approximation as the exact pathwise derivative for lookback
    options involves more complex calculations with indicator functions.

    References
    ----------
    Broadie, M., Glasserman, P., & Kou, S. G. (1997). A continuity correction
    for discrete barrier options. Mathematical Finance, 7(4), 325-349.
    """
    # Input validation
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if n_paths_per_x <= 0:
        raise ValueError(f"n_paths_per_x must be positive, got {n_paths_per_x}")

    device = get_device()

    # Generate initial spot prices grid
    x = torch.linspace(x_min * K, x_max * K, m, device=device, dtype=DEFAULT_DTYPE)
    x = x.reshape(-1, 1)

    # Simulate paths
    paths, xi = simulate_bs_paths(x, params, n_steps, n_paths_per_x, seed)
    # paths shape: (m, n_paths_per_x, n_steps + 1)

    # Compute maximum price along each path
    max_prices = paths.max(dim=2)[0]  # Shape: (m, n_paths_per_x)

    # Compute discounted payoff
    discount_factor = torch.exp(torch.tensor(-params.r * params.T, dtype=DEFAULT_DTYPE))
    payoffs = discount_factor * torch.maximum(
        max_prices - K,
        torch.zeros_like(max_prices)
    )

    # Price label: MC average
    price_label = payoffs.mean(dim=1, keepdim=True)  # Shape: (m, 1)

    # Pathwise delta
    # For lookback: dM/dS0 = M/S0 (approximate)
    # This is exact when the maximum occurs at a fixed time, but an approximation
    # in general due to the discrete monitoring
    indicator = (max_prices > K).to(dtype=DEFAULT_DTYPE)  # Shape: (m, n_paths_per_x)

    # Compute dM/dS0 for each path
    dM_dS0 = max_prices / x  # Shape: (m, n_paths_per_x)

    # Pathwise delta = E[1_{M > K} * dM/dS0]
    delta_pathwise = discount_factor * (indicator * dM_dS0).mean(dim=1, keepdim=True)

    # LRM delta
    # Score function: same as for Asian option
    normalizer = x * params.sigma * torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE))

    # Sum all increments
    xi_sum = xi.sum(dim=2)  # Shape: (m, n_paths_per_x)
    score = xi_sum / normalizer  # Shape: (m, n_paths_per_x)

    # LRM delta = E[payoff * score]
    delta_lrm = (payoffs * score).mean(dim=1, keepdim=True)

    return x, price_label, delta_pathwise, delta_lrm
