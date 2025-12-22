"""Dataset generation for gamma portfolio hedging experiments.

This module provides functions to generate datasets for gamma portfolio
hedging using differential machine learning.
"""


import torch
from torch import Tensor

from diffml.bs_analytics import bs_call_gamma, bs_call_price
from diffml.config import DEFAULT_DTYPE, BSParams, get_device
from diffml.simulation import simulate_bs_terminal


def make_portfolio_gamma_dataset(
    m: int,
    params: BSParams,
    strikes: Tensor,
    weights: Tensor,
    x_min: float = 0.5,
    x_max: float = 1.5,
    n_paths_per_x: int = 20,
    seed: int | None = 1234
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Generate dataset for portfolio gamma hedging.

    Portfolio payoff:
    sum_i weights[i] * max(S_T - strikes[i], 0)

    Default example portfolio:
    (S_T - 0.85)_+ - 1.5 * (S_T - 0.9)_+ + 0.75 * (S_T - 1.15)_+

    Parameters
    ----------
    m : int
        Number of spot price points in the dataset.
    params : BSParams
        Black-Scholes parameters (r, sigma, T).
    strikes : Tensor
        Strike prices for portfolio options, shape (n_options,).
    weights : Tensor
        Weights for portfolio options, shape (n_options,).
    x_min : float, optional
        Minimum spot price. Default is 0.5.
    x_max : float, optional
        Maximum spot price. Default is 1.5.
    n_paths_per_x : int, optional
        Number of Monte Carlo paths per spot price. Default is 20.
    seed : int | None, optional
        Random seed for reproducibility. Default is 1234.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        A tuple containing:
        - x: Spot prices of shape (m, 1)
        - price_true: Analytic portfolio prices of shape (m, 1)
        - delta_true: Analytic portfolio deltas of shape (m, 1)
        - gamma_true: Analytic portfolio gammas of shape (m, 1)
        - price_mc: Monte Carlo prices of shape (m, 1)
        - delta_pw: Pathwise deltas of shape (m, 1)
        - gamma_pwlr: PW-LR gammas of shape (m, 1)

    Raises
    ------
    ValueError
        If m <= 0 or n_paths_per_x <= 0.
        If x_min >= x_max.
        If strikes and weights have different shapes.

    Examples
    --------
    >>> params = BSParams(r=0.05, sigma=0.2, T=0.25)
    >>> strikes = torch.tensor([0.85, 0.9, 1.15])
    >>> weights = torch.tensor([1.0, -1.5, 0.75])
    >>> results = make_portfolio_gamma_dataset(
    ...     m=100, params=params, strikes=strikes, weights=weights, n_paths_per_x=10000
    ... )
    >>> x, price_true, delta_true, gamma_true = results[:4]
    >>> x.shape, price_true.shape
    (torch.Size([100, 1]), torch.Size([100, 1]))
    """
    # Input validation
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if n_paths_per_x <= 0:
        raise ValueError(f"n_paths_per_x must be positive, got {n_paths_per_x}")
    if x_min >= x_max:
        raise ValueError(f"x_min must be less than x_max, got x_min={x_min}, x_max={x_max}")
    if strikes.shape != weights.shape:
        raise ValueError(f"strikes and weights must have same shape, got {strikes.shape} and {weights.shape}")

    # Get device and set precision
    device = get_device()
    strikes = strikes.to(device=device, dtype=DEFAULT_DTYPE)
    weights = weights.to(device=device, dtype=DEFAULT_DTYPE)

    # Build grid of spot prices
    # Shape: (m, 1)
    x = torch.linspace(x_min, x_max, m, device=device, dtype=DEFAULT_DTYPE).reshape(m, 1)

    # Compute analytic portfolio values using Black-Scholes formulas
    price_true = torch.zeros_like(x)
    gamma_true = torch.zeros_like(x)

    # For delta, we need to use the Black-Scholes N(d1) formula
    # Delta of call = N(d1) where d1 = (log(S/K) + (r + sigma^2/2)*T) / (sigma * sqrt(T))
    from diffml.bs_analytics import _normal_cdf

    delta_true = torch.zeros_like(x)

    for K_i, w_i in zip(strikes, weights, strict=False):
        # Compute call price and gamma for this strike
        call_price_i = bs_call_price(x, K_i, params)
        call_gamma_i = bs_call_gamma(x, K_i, params)

        # Compute delta using N(d1)
        sqrt_T = torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE))
        d1 = (torch.log(x / K_i) + (params.r + 0.5 * params.sigma ** 2) * params.T) / (params.sigma * sqrt_T)
        call_delta_i = _normal_cdf(d1)

        # Add weighted contribution to portfolio
        price_true += w_i * call_price_i
        delta_true += w_i * call_delta_i
        gamma_true += w_i * call_gamma_i

    # Monte Carlo simulation
    # ST shape: (m, n_paths_per_x), xi shape: (m, n_paths_per_x)
    ST, xi = simulate_bs_terminal(x, params, n_paths_per_x, seed=seed)

    # Compute portfolio payoff for each path
    # Shape: (m, n_paths_per_x)
    portfolio_payoff = torch.zeros_like(ST)

    for K_i, w_i in zip(strikes, weights, strict=False):
        call_payoff_i = torch.maximum(
            ST - K_i, torch.tensor(0.0, dtype=DEFAULT_DTYPE, device=device)
        )
        portfolio_payoff += w_i * call_payoff_i

    # Discount factor
    discount = torch.exp(
        torch.tensor(-params.r * params.T, dtype=DEFAULT_DTYPE, device=device)
    )
    disc_payoff = discount * portfolio_payoff

    # Monte Carlo price
    # Shape: (m, 1)
    price_mc = disc_payoff.mean(dim=1, keepdim=True)

    # Pathwise delta
    # For each call: delta_paths = disc * 1_{ST > K} * (ST / x)
    # Portfolio delta: sum of weighted call deltas
    # Shape: (m, n_paths_per_x)
    delta_paths = torch.zeros_like(ST)

    for K_i, w_i in zip(strikes, weights, strict=False):
        in_the_money_i = (K_i < ST).to(dtype=DEFAULT_DTYPE)
        delta_paths_i = discount * in_the_money_i * (ST / x)
        delta_paths += w_i * delta_paths_i

    # Pathwise delta label
    # Shape: (m, 1)
    delta_pw = delta_paths.mean(dim=1, keepdim=True)

    # PW-LR gamma (combined pathwise-likelihood ratio)
    # For a single call: gamma = disc * 1_{ST > K} * (ST / S0^2) * (xi / (sigma * sqrt(T)) - 1)
    # Portfolio gamma: sum of weighted call gammas
    # Shape: (m, n_paths_per_x)
    sqrt_T = torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE, device=device))
    gamma_paths = torch.zeros_like(ST)

    for K_i, w_i in zip(strikes, weights, strict=False):
        in_the_money_i = (K_i < ST).to(dtype=DEFAULT_DTYPE)
        # PW-LR gamma formula
        gamma_paths_i = discount * in_the_money_i * (ST / (x ** 2)) * (xi / (params.sigma * sqrt_T) - 1)
        gamma_paths += w_i * gamma_paths_i

    # PW-LR gamma label
    # Shape: (m, 1)
    gamma_pwlr = gamma_paths.mean(dim=1, keepdim=True)

    return x, price_true, delta_true, gamma_true, price_mc, delta_pw, gamma_pwlr
