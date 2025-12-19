"""Black-Scholes analytical formulas for option pricing and Greeks.

This module provides analytical solutions for various option types under
the Black-Scholes model, used for benchmarking and validation.
All computations are done in double precision using PyTorch.
"""

from typing import Union

import torch
from torch import Tensor

from diffml.config import DEFAULT_DTYPE, BSParams


def _normal_cdf(z: Tensor) -> Tensor:
    """Compute the cumulative distribution function of standard normal.

    Parameters
    ----------
    z : Tensor
        Input values.

    Returns
    -------
    Tensor
        CDF values N(z).
    """
    # Using error function: N(z) = 0.5 * (1 + erf(z/sqrt(2)))
    return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, dtype=z.dtype))))


def _normal_pdf(z: Tensor) -> Tensor:
    """Compute the probability density function of standard normal.

    Parameters
    ----------
    z : Tensor
        Input values.

    Returns
    -------
    Tensor
        PDF values n(z) = exp(-z^2/2) / sqrt(2*pi).
    """
    return torch.exp(-0.5 * z * z) / torch.sqrt(
        2.0 * torch.tensor(torch.pi, dtype=z.dtype)
    )


def bs_digital_price(x: Tensor, K: float, params: BSParams) -> Tensor:
    """Calculate Black-Scholes price for a digital (binary) call option.

    A digital call pays 1 if S_T > K, and 0 otherwise.

    Price = e^(-rT) * N(d2)

    where d2 = (log(S/K) + (r - sigma^2/2)*T) / (sigma * sqrt(T))

    Parameters
    ----------
    x : Tensor
        Current spot price(s). Shape: (batch_size, 1) or (batch_size,).
    K : float
        Strike price.
    params : BSParams
        Black-Scholes parameters (r, sigma, T).

    Returns
    -------
    Tensor
        Digital call option price(s). Same shape as x.
    """
    # Ensure double precision
    x = x.to(dtype=DEFAULT_DTYPE)

    # Handle edge case when T = 0
    if params.T <= 0:
        return (x > K).to(dtype=DEFAULT_DTYPE)

    # Calculate d2
    # d2 = (log(S/K) + (r - sigma^2/2)*T) / (sigma * sqrt(T))
    sqrt_T = torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE))
    d2 = (
        torch.log(x / K) + (params.r - 0.5 * params.sigma ** 2) * params.T
    ) / (params.sigma * sqrt_T)

    # Digital call price = e^(-rT) * N(d2)
    discount_factor = torch.exp(torch.tensor(-params.r * params.T, dtype=DEFAULT_DTYPE))
    price = discount_factor * _normal_cdf(d2)

    return price


def bs_digital_delta(x: Tensor, K: float, params: BSParams) -> Tensor:
    """Calculate Black-Scholes delta for a digital call option.

    Delta = dPrice/dS = e^(-rT) * n(d2) / (S * sigma * sqrt(T))

    where n(d2) is the standard normal PDF at d2.

    Parameters
    ----------
    x : Tensor
        Current spot price(s). Shape: (batch_size, 1) or (batch_size,).
    K : float
        Strike price.
    params : BSParams
        Black-Scholes parameters (r, sigma, T).

    Returns
    -------
    Tensor
        Digital call option delta(s). Same shape as x.
    """
    # Ensure double precision
    x = x.to(dtype=DEFAULT_DTYPE)

    # Handle edge case when T = 0
    if params.T <= 0:
        # Delta is technically undefined at S = K when T = 0
        # Return 0 for practical purposes
        return torch.zeros_like(x)

    # Calculate d2
    sqrt_T = torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE))
    d2 = (
        torch.log(x / K) + (params.r - 0.5 * params.sigma ** 2) * params.T
    ) / (params.sigma * sqrt_T)

    # Digital delta = e^(-rT) * n(d2) / (S * sigma * sqrt(T))
    discount_factor = torch.exp(torch.tensor(-params.r * params.T, dtype=DEFAULT_DTYPE))
    delta = discount_factor * _normal_pdf(d2) / (x * params.sigma * sqrt_T)

    return delta


def bs_call_price(x: Tensor, K: Union[float, Tensor], params: BSParams) -> Tensor:
    """Calculate Black-Scholes price for a European call option.

    Price = S * N(d1) - K * e^(-rT) * N(d2)

    where:
        d1 = (log(S/K) + (r + sigma^2/2)*T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    Parameters
    ----------
    x : Tensor
        Current spot price(s). Shape: (batch_size, 1) or (batch_size,).
    K : Union[float, Tensor]
        Strike price(s). Can be scalar or same shape as x.
    params : BSParams
        Black-Scholes parameters (r, sigma, T).

    Returns
    -------
    Tensor
        European call option price(s). Same shape as x.
    """
    # Ensure double precision
    x = x.to(dtype=DEFAULT_DTYPE)
    if isinstance(K, Tensor):
        K = K.to(dtype=DEFAULT_DTYPE)
    else:
        K = torch.tensor(K, dtype=DEFAULT_DTYPE)

    # Handle edge case when T = 0
    if params.T <= 0:
        return torch.maximum(x - K, torch.tensor(0.0, dtype=DEFAULT_DTYPE))

    # Calculate d1 and d2
    sqrt_T = torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE))
    d1 = (
        torch.log(x / K) + (params.r + 0.5 * params.sigma ** 2) * params.T
    ) / (params.sigma * sqrt_T)
    d2 = d1 - params.sigma * sqrt_T

    # Call price = S * N(d1) - K * e^(-rT) * N(d2)
    discount_factor = torch.exp(torch.tensor(-params.r * params.T, dtype=DEFAULT_DTYPE))
    price = x * _normal_cdf(d1) - K * discount_factor * _normal_cdf(d2)

    return price


def bs_call_gamma(x: Tensor, K: Union[float, Tensor], params: BSParams) -> Tensor:
    """Calculate Black-Scholes gamma for a European call option.

    Gamma = d^2Price/dS^2 = n(d1) / (S * sigma * sqrt(T))

    where n(d1) is the standard normal PDF at d1.

    Parameters
    ----------
    x : Tensor
        Current spot price(s). Shape: (batch_size, 1) or (batch_size,).
    K : Union[float, Tensor]
        Strike price(s). Can be scalar or same shape as x.
    params : BSParams
        Black-Scholes parameters (r, sigma, T).

    Returns
    -------
    Tensor
        European call option gamma(s). Same shape as x.
    """
    # Ensure double precision
    x = x.to(dtype=DEFAULT_DTYPE)
    if isinstance(K, Tensor):
        K = K.to(dtype=DEFAULT_DTYPE)
    else:
        K = torch.tensor(K, dtype=DEFAULT_DTYPE)

    # Handle edge case when T = 0
    if params.T <= 0:
        # Gamma is technically undefined at S = K when T = 0
        # Return 0 for practical purposes
        return torch.zeros_like(x)

    # Calculate d1
    sqrt_T = torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE))
    d1 = (
        torch.log(x / K) + (params.r + 0.5 * params.sigma ** 2) * params.T
    ) / (params.sigma * sqrt_T)

    # Gamma = n(d1) / (S * sigma * sqrt(T))
    gamma = _normal_pdf(d1) / (x * params.sigma * sqrt_T)

    return gamma
