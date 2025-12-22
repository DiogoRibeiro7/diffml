"""Dataset generation for smoothing technique comparison experiments.

This module provides functions to generate datasets for comparing different
smoothing techniques in differential machine learning.
"""


import torch
from torch import Tensor

from diffml.config import DEFAULT_DTYPE, BSParams, get_device
from diffml.simulation import simulate_bs_terminal


def make_smoothed_digital_dataset(
    m: int,
    K: float,
    params: BSParams,
    eps: float | None = None,
    eps_multiplier: float | None = 1.0,
    x_min: float = 40.0,
    x_max: float = 160.0,
    n_paths_per_x: int = 10,
    seed: int | None = 1234
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate dataset for smoothed digital call option with ramp function.

    Uses a ramp smoothing function:
    - g(ST) = 0 for ST <= K - eps/2
    - g(ST) = 1 for ST >= K + eps/2
    - Linear ramp in between

    Parameters
    ----------
    m : int
        Number of spot price points in the dataset.
    K : float
        Strike price of the digital option.
    params : BSParams
        Black-Scholes parameters (r, sigma, T).
    eps : float | None
        Absolute width of the smoothing ramp. If None, computed from eps_multiplier.
    eps_multiplier : float | None
        Multiplier for sigma * sqrt(T) * K when eps is not provided.
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
        - delta_pathwise: Pathwise deltas of shape (m, 1)
        - delta_lrm: LRM deltas of shape (m, 1)

    Raises
    ------
    ValueError
        If m <= 0 or n_paths_per_x <= 0.
        If x_min >= x_max or K <= 0 or eps <= 0.

    Examples
    --------
    >>> params = BSParams(r=0.05, sigma=0.2, T=0.25)
    >>> x, prices, delta_pw, delta_lrm = make_smoothed_digital_dataset(
    ...     m=100, K=100.0, eps=5.0, params=params, n_paths_per_x=10000
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

    if eps is None:
        if eps_multiplier is None:
            raise ValueError("Provide either eps or eps_multiplier.")
        if eps_multiplier <= 0:
            raise ValueError(f"eps_multiplier must be positive, got {eps_multiplier}")
        sqrt_T = torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE))
        eps_value = float(eps_multiplier * params.sigma * float(sqrt_T) * K)
    else:
        eps_value = float(eps)

    if eps_value <= 0:
        raise ValueError(f"Smoothing parameter eps must be positive, got {eps_value}")

    # Get device and set precision
    device = get_device()

    # Build grid of spot prices
    # Shape: (m, 1)
    x = torch.linspace(x_min, x_max, m, device=device, dtype=DEFAULT_DTYPE).reshape(m, 1)

    # Simulate terminal prices
    # ST shape: (m, n_paths_per_x), xi shape: (m, n_paths_per_x)
    ST, xi = simulate_bs_terminal(x, params, n_paths_per_x, seed=seed)

    # Compute discount factor
    discount = torch.exp(
        torch.tensor(-params.r * params.T, dtype=DEFAULT_DTYPE, device=device)
    )

    # Smoothed payoff using ramp function
    # g(ST) = 0 for ST <= K - eps/2
    # g(ST) = 1 for ST >= K + eps/2
    # g(ST) = (ST - (K - eps/2)) / eps for K - eps/2 < ST < K + eps/2

    eps_tensor = torch.tensor(eps_value, dtype=DEFAULT_DTYPE, device=device)
    strike = torch.tensor(K, dtype=DEFAULT_DTYPE, device=device)
    lower_bound = strike - eps_tensor / 2
    upper_bound = strike + eps_tensor / 2

    # Initialize payoff tensor
    # Shape: (m, n_paths_per_x)
    payoff = torch.zeros_like(ST)

    # Apply ramp function
    # Below lower bound: payoff = 0 (already initialized)
    # Above upper bound: payoff = 1
    above_upper = upper_bound <= ST
    payoff[above_upper] = 1.0

    # In ramp region: linear interpolation
    in_ramp = (lower_bound < ST) & (upper_bound > ST)
    payoff[in_ramp] = (ST[in_ramp] - lower_bound) / eps_tensor

    # Discounted payoff
    disc_payoff = discount * payoff

    # Price label: Monte Carlo mean over paths
    # Shape: (m, 1)
    price_label = disc_payoff.mean(dim=1, keepdim=True)

    # Pathwise delta using chain rule
    # dg/dST = 1/eps in ramp region, 0 elsewhere
    # dST/dS0 = ST/S0 (lognormal property)
    # delta = disc * (dg/dST) * (dST/dS0)

    # Derivative of smoothed payoff
    # Shape: (m, n_paths_per_x)
    dg_dST = torch.zeros_like(ST)
    dg_dST[in_ramp] = 1.0 / eps_tensor

    # Chain rule: delta = disc * (dg/dST) * (ST/x)
    # Broadcasting: x is (m, 1), ST is (m, n_paths_per_x)
    delta_paths = discount * dg_dST * (ST / x)

    # Pathwise delta label: mean over paths
    # Shape: (m, 1)
    delta_pathwise = delta_paths.mean(dim=1, keepdim=True)

    # LRM delta using xi from simulation
    sqrt_T = torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE, device=device))
    score = xi / (params.sigma * sqrt_T)
    score = score / x  # Scale by initial spot
    delta_lrm_paths = disc_payoff * score
    delta_lrm = delta_lrm_paths.mean(dim=1, keepdim=True)

    return x, price_label, delta_pathwise, delta_lrm
