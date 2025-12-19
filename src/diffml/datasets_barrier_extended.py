"""Dataset generators for extended barrier options.

This module provides dataset generation functions for additional barrier option
types including up-and-out calls and double barrier calls, using Monte Carlo
simulation with both pathwise and likelihood ratio method (LRM) for sensitivity
estimation.
"""

from typing import Optional

import torch
from torch import Tensor

from diffml.config import DEFAULT_DTYPE, BSParams, get_device
from diffml.simulation import simulate_bs_paths, simulate_bs_two_step


def make_up_and_out_call_dataset(
    m: int,
    K: float,
    H: float,
    params: BSParams,
    n_steps: int = 16,
    x_min: float = 0.5,
    x_max: float = 1.2,
    n_paths_per_x: int = 100,
    seed: Optional[int] = 1234,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate dataset for up-and-out call option pricing.

    An up-and-out call pays max(S_T - K, 0) if the stock price never exceeds
    the upper barrier H during the option's life. If S_t >= H at any time,
    the option knocks out and becomes worthless.

    Parameters
    ----------
    m : int
        Number of initial spot prices to generate.
    K : float
        Strike price.
    H : float
        Upper barrier level. Must be greater than K for meaningful results.
    params : BSParams
        Black-Scholes parameters (r, sigma, T).
    n_steps : int, optional
        Number of time steps for barrier monitoring. Default is 16.
    x_min : float, optional
        Minimum initial spot price as fraction of K. Default is 0.5.
    x_max : float, optional
        Maximum initial spot price as fraction of K. Default is 1.2.
        Should be less than H/K to avoid immediate knock-out.
    n_paths_per_x : int, optional
        Number of Monte Carlo paths per initial spot. Default is 100.
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
    dV/dS0 = E[1_{max S_t < H} * d(payoff)/dS0]
    where the payoff derivative includes both the knock-out condition and
    the terminal payoff.

    The LRM delta uses the standard likelihood ratio approach with the
    score function based on all path increments.

    References
    ----------
    Hull, J. C. (2018). Options, Futures, and Other Derivatives (10th ed.).
    Chapter 26: Exotic Options.
    """
    # Input validation
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if H <= K:
        raise ValueError(f"Barrier H={H} should be greater than strike K={K}")
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

    # Check knock-out condition: did price ever exceed H?
    max_prices = paths.max(dim=2)[0]  # Shape: (m, n_paths_per_x)
    not_knocked_out = (max_prices < H).to(dtype=DEFAULT_DTYPE)

    # Terminal prices
    terminal_prices = paths[:, :, -1]  # Shape: (m, n_paths_per_x)

    # Compute payoff (only if not knocked out)
    vanilla_payoff = torch.maximum(
        terminal_prices - K,
        torch.zeros_like(terminal_prices)
    )
    payoff = vanilla_payoff * not_knocked_out

    # Discounted payoff
    discount_factor = torch.exp(torch.tensor(-params.r * params.T, dtype=DEFAULT_DTYPE))
    discounted_payoff = discount_factor * payoff

    # Price label: MC average
    price_label = discounted_payoff.mean(dim=1, keepdim=True)  # Shape: (m, 1)

    # Pathwise delta
    # For up-and-out call: derivative includes both knock-out and payoff effects
    # Approximate: dV/dS0 ≈ E[1_{not knocked} * 1_{S_T > K} * S_T/S0]
    in_the_money = (terminal_prices > K).to(dtype=DEFAULT_DTYPE)
    dST_dS0 = terminal_prices / x  # Shape: (m, n_paths_per_x)

    delta_pathwise_values = discount_factor * not_knocked_out * in_the_money * dST_dS0
    delta_pathwise = delta_pathwise_values.mean(dim=1, keepdim=True)

    # LRM delta
    # Score function based on all increments
    normalizer = x * params.sigma * torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE))
    xi_sum = xi.sum(dim=2)  # Shape: (m, n_paths_per_x)
    score = xi_sum / normalizer

    # LRM delta = E[discounted_payoff * score]
    delta_lrm = (discounted_payoff * score).mean(dim=1, keepdim=True)

    return x, price_label, delta_pathwise, delta_lrm


def make_double_barrier_call_dataset(
    m: int,
    K: float,
    L: float,
    H: float,
    params: BSParams,
    n_steps: int = 16,
    x_min: float = 0.6,
    x_max: float = 0.9,
    n_paths_per_x: int = 100,
    seed: Optional[int] = 1234,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate dataset for double barrier call option pricing.

    A double barrier call pays max(S_T - K, 0) if the stock price stays within
    [L, H] during the option's life. If S_t <= L or S_t >= H at any time,
    the option knocks out and becomes worthless.

    Parameters
    ----------
    m : int
        Number of initial spot prices to generate.
    K : float
        Strike price.
    L : float
        Lower barrier level. Must be less than K.
    H : float
        Upper barrier level. Must be greater than K.
    params : BSParams
        Black-Scholes parameters (r, sigma, T).
    n_steps : int, optional
        Number of time steps for barrier monitoring. Default is 16.
    x_min : float, optional
        Minimum initial spot price as fraction of K. Default is 0.6.
        Should be greater than L/K to avoid immediate knock-out.
    x_max : float, optional
        Maximum initial spot price as fraction of K. Default is 0.9.
        Should be less than H/K to avoid immediate knock-out.
    n_paths_per_x : int, optional
        Number of Monte Carlo paths per initial spot. Default is 100.
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
    The pathwise delta considers both barrier constraints:
    dV/dS0 = E[1_{L < min S_t and max S_t < H} * d(payoff)/dS0]

    The discrete monitoring may introduce some bias compared to continuous
    monitoring, which decreases as n_steps increases.

    References
    ----------
    Rubinstein, M., & Reiner, E. (1991). Breaking down the barriers.
    Risk Magazine, 4(8), 28-35.
    """
    # Input validation
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if L >= K:
        raise ValueError(f"Lower barrier L={L} should be less than strike K={K}")
    if H <= K:
        raise ValueError(f"Upper barrier H={H} should be greater than strike K={K}")
    if L >= H:
        raise ValueError(f"Lower barrier L={L} must be less than upper barrier H={H}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if n_paths_per_x <= 0:
        raise ValueError(f"n_paths_per_x must be positive, got {n_paths_per_x}")

    device = get_device()

    # Generate initial spot prices grid
    # Ensure initial prices are within barriers
    x_min_safe = max(x_min, L / K * 1.01)  # Slightly above lower barrier
    x_max_safe = min(x_max, H / K * 0.99)  # Slightly below upper barrier
    x = torch.linspace(x_min_safe * K, x_max_safe * K, m, device=device, dtype=DEFAULT_DTYPE)
    x = x.reshape(-1, 1)

    # Simulate paths
    paths, xi = simulate_bs_paths(x, params, n_steps, n_paths_per_x, seed)
    # paths shape: (m, n_paths_per_x, n_steps + 1)

    # Check knock-out conditions
    min_prices = paths.min(dim=2)[0]  # Shape: (m, n_paths_per_x)
    max_prices = paths.max(dim=2)[0]  # Shape: (m, n_paths_per_x)
    not_knocked_out = ((min_prices > L) & (max_prices < H)).to(dtype=DEFAULT_DTYPE)

    # Terminal prices
    terminal_prices = paths[:, :, -1]  # Shape: (m, n_paths_per_x)

    # Compute payoff (only if not knocked out)
    vanilla_payoff = torch.maximum(
        terminal_prices - K,
        torch.zeros_like(terminal_prices)
    )
    payoff = vanilla_payoff * not_knocked_out

    # Discounted payoff
    discount_factor = torch.exp(torch.tensor(-params.r * params.T, dtype=DEFAULT_DTYPE))
    discounted_payoff = discount_factor * payoff

    # Price label: MC average
    price_label = discounted_payoff.mean(dim=1, keepdim=True)  # Shape: (m, 1)

    # Pathwise delta
    # For double barrier: similar to single barrier but with both constraints
    in_the_money = (terminal_prices > K).to(dtype=DEFAULT_DTYPE)
    dST_dS0 = terminal_prices / x  # Shape: (m, n_paths_per_x)

    delta_pathwise_values = discount_factor * not_knocked_out * in_the_money * dST_dS0
    delta_pathwise = delta_pathwise_values.mean(dim=1, keepdim=True)

    # LRM delta
    normalizer = x * params.sigma * torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE))
    xi_sum = xi.sum(dim=2)  # Shape: (m, n_paths_per_x)
    score = xi_sum / normalizer

    # LRM delta = E[discounted_payoff * score]
    delta_lrm = (discounted_payoff * score).mean(dim=1, keepdim=True)

    return x, price_label, delta_pathwise, delta_lrm


def make_down_and_in_call_dataset(
    m: int,
    K: float,
    B: float,
    params: BSParams,
    x_min: float = 0.85,
    x_max: float = 1.5,
    n_paths_per_x: int = 100,
    seed: Optional[int] = 1234,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate dataset for down-and-in call option pricing.

    A down-and-in call pays max(S_T - K, 0) only if the stock price touches
    or goes below the barrier B at some point during the option's life.
    This uses a two-step simulation for simplicity.

    Parameters
    ----------
    m : int
        Number of initial spot prices to generate.
    K : float
        Strike price.
    B : float
        Lower barrier level (knock-in). Must be less than K.
    params : BSParams
        Black-Scholes parameters (r, sigma, T).
    x_min : float, optional
        Minimum initial spot price as fraction of K. Default is 0.85.
    x_max : float, optional
        Maximum initial spot price as fraction of K. Default is 1.5.
    n_paths_per_x : int, optional
        Number of Monte Carlo paths per initial spot. Default is 100.
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
    Uses two-step simulation with monitoring at T/2 and T for simplicity.
    The option knocks in if S_{T/2} <= B.

    The pathwise delta is:
    dV/dS0 = E[1_{knocked in} * d(payoff)/dS0]

    References
    ----------
    Broadie, M., & Glasserman, P. (1997). Pricing American-style securities
    using simulation. Journal of Economic Dynamics and Control, 21(8-9), 1323-1352.
    """
    # Input validation
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if B >= K:
        raise ValueError(f"Barrier B={B} should be less than strike K={K}")
    if n_paths_per_x <= 0:
        raise ValueError(f"n_paths_per_x must be positive, got {n_paths_per_x}")

    device = get_device()

    # Generate initial spot prices grid
    x = torch.linspace(x_min * K, x_max * K, m, device=device, dtype=DEFAULT_DTYPE)
    x = x.reshape(-1, 1)

    # Two-step simulation: monitor at T/2 and T
    T_half = params.T / 2
    S1, S2, xi1, xi2 = simulate_bs_two_step(x, params, T_half, params.T, n_paths_per_x, seed)
    # S1: prices at T/2, S2: prices at T, shape: (m, n_paths_per_x)

    # Check knock-in condition: did price go below B at T/2?
    knocked_in = (S1 <= B).to(dtype=DEFAULT_DTYPE)

    # Compute payoff (only if knocked in)
    vanilla_payoff = torch.maximum(S2 - K, torch.zeros_like(S2))
    payoff = vanilla_payoff * knocked_in

    # Discounted payoff
    discount_factor = torch.exp(torch.tensor(-params.r * params.T, dtype=DEFAULT_DTYPE))
    discounted_payoff = discount_factor * payoff

    # Price label: MC average
    price_label = discounted_payoff.mean(dim=1, keepdim=True)

    # Pathwise delta
    in_the_money = (S2 > K).to(dtype=DEFAULT_DTYPE)
    dS2_dS0 = S2 / x  # Terminal price sensitivity

    delta_pathwise_values = discount_factor * knocked_in * in_the_money * dS2_dS0
    delta_pathwise = delta_pathwise_values.mean(dim=1, keepdim=True)

    # LRM delta using both increments
    normalizer = x * params.sigma * torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE))

    # Combine increments from both steps
    # Weight xi1 by sqrt(T/2) and xi2 by sqrt(T/2) to account for different time intervals
    weight1 = torch.sqrt(torch.tensor(T_half / params.T, dtype=DEFAULT_DTYPE))
    weight2 = torch.sqrt(torch.tensor((params.T - T_half) / params.T, dtype=DEFAULT_DTYPE))
    xi_weighted = xi1 * weight1 + xi2 * weight2
    score = xi_weighted / normalizer

    # LRM delta
    delta_lrm = (discounted_payoff * score).mean(dim=1, keepdim=True)

    return x, price_label, delta_pathwise, delta_lrm
