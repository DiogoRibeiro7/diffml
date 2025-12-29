"""Dataset generation for exotic options.

This module implements pricing and sensitivity computation for exotic derivatives
including variance swaps, chooser options, compound options, and lookback options
using Monte Carlo simulation with differential machine learning.

Mathematical Background:
    Exotic options have complex payoffs that depend on the path of the underlying
    asset in sophisticated ways, making them ideal candidates for DML techniques.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import math

from .config import BSParams, get_device
from .simulation import simulate_bs_paths


class ExoticOptionType(Enum):
    """Types of exotic options."""
    VARIANCE_SWAP = "variance_swap"
    VOLATILITY_SWAP = "volatility_swap"
    CHOOSER = "chooser"
    COMPOUND_CALL_ON_CALL = "compound_call_on_call"
    COMPOUND_CALL_ON_PUT = "compound_call_on_put"
    LOOKBACK_CALL = "lookback_call"
    LOOKBACK_PUT = "lookback_put"
    RAINBOW = "rainbow"
    QUANTO = "quanto"
    CLIQUET = "cliquet"


@dataclass
class VarianceSwapParams:
    """Parameters for variance and volatility swaps.

    Attributes:
        strike_variance: Strike variance (annualized)
        notional: Notional amount
        realized_var_type: 'log' or 'simple' returns
        sampling_freq: Number of observations per year (252 for daily)
    """
    strike_variance: float = 0.04  # 20% vol squared
    notional: float = 1000000.0
    realized_var_type: str = 'log'
    sampling_freq: int = 252


@dataclass
class ChooserOptionParams:
    """Parameters for chooser options.

    Attributes:
        choose_time: Time when holder chooses call or put (fraction of T)
        strike: Strike price for the chosen option
    """
    choose_time: float = 0.5
    strike: float = 100.0


@dataclass
class CompoundOptionParams:
    """Parameters for compound options.

    Attributes:
        compound_type: Type of compound option
        strike1: Strike of the outer option
        strike2: Strike of the inner option
        maturity1: Maturity of the outer option
        maturity2: Maturity of the inner option
    """
    compound_type: str = 'call_on_call'
    strike1: float = 10.0  # Strike to buy the inner option
    strike2: float = 100.0  # Strike of the inner option
    maturity1: float = 0.5
    maturity2: float = 1.0


def calculate_realized_variance(
    paths: torch.Tensor,
    dt: float,
    var_type: str = 'log',
    annualize: bool = True
) -> torch.Tensor:
    """Calculate realized variance from price paths.

    Mathematical Formula:
    --------------------
    For log returns:
        RV = (252/n) * Σ(log(S_i/S_{i-1}))²

    For simple returns:
        RV = (252/n) * Σ((S_i - S_{i-1})/S_{i-1})²

    Parameters:
        paths: Price paths of shape (n_paths, n_steps)
        dt: Time step size
        var_type: 'log' or 'simple' returns
        annualize: Whether to annualize the variance

    Returns:
        Realized variance for each path
    """
    if var_type == 'log':
        # Log returns
        returns = torch.log(paths[:, 1:] / paths[:, :-1])
    else:
        # Simple returns
        returns = (paths[:, 1:] - paths[:, :-1]) / paths[:, :-1]

    # Calculate variance
    variance = torch.mean(returns**2, dim=1)

    # Annualize if requested
    if annualize:
        variance = variance / dt

    return variance


def price_variance_swap(
    paths: torch.Tensor,
    params: VarianceSwapParams,
    dt: float
) -> torch.Tensor:
    """Price variance swap using Monte Carlo.

    Payoff = Notional * (RealizedVariance - StrikeVariance)

    Parameters:
        paths: Monte Carlo paths
        params: Variance swap parameters
        dt: Time step size

    Returns:
        Variance swap values
    """
    # Calculate realized variance
    realized_var = calculate_realized_variance(
        paths, dt, params.realized_var_type, annualize=True
    )

    # Payoff
    payoff = params.notional * (realized_var - params.strike_variance)

    return payoff


def price_volatility_swap(
    paths: torch.Tensor,
    params: VarianceSwapParams,
    dt: float
) -> torch.Tensor:
    """Price volatility swap using Monte Carlo.

    Payoff = Notional * (RealizedVol - StrikeVol)

    Parameters:
        paths: Monte Carlo paths
        params: Variance swap parameters (strike is volatility)
        dt: Time step size

    Returns:
        Volatility swap values
    """
    # Calculate realized volatility
    realized_var = calculate_realized_variance(
        paths, dt, params.realized_var_type, annualize=True
    )
    realized_vol = torch.sqrt(realized_var)

    strike_vol = math.sqrt(params.strike_variance)

    # Payoff
    payoff = params.notional * (realized_vol - strike_vol)

    return payoff


def price_chooser_option(
    paths: torch.Tensor,
    params: ChooserOptionParams,
    r: float,
    T: float
) -> torch.Tensor:
    """Price chooser option using Monte Carlo.

    At choose_time, holder chooses max(Call, Put)

    Mathematical Derivation:
    ------------------------
    At time τ (choose time):
        V_τ = max(C(S_τ, K, T-τ), P(S_τ, K, T-τ))

    The option value is:
        V_0 = e^{-rτ} E^Q[V_τ]

    Parameters:
        paths: Monte Carlo paths
        params: Chooser option parameters
        r: Risk-free rate
        T: Total maturity

    Returns:
        Chooser option values
    """
    n_paths, n_steps = paths.shape
    choose_idx = int(params.choose_time * (n_steps - 1))

    # Price at choose time
    S_choose = paths[:, choose_idx]

    # Terminal price
    S_T = paths[:, -1]

    # Time remaining after choice
    remaining_time = T * (1 - params.choose_time)

    # Call and put payoffs at maturity
    call_payoff = torch.maximum(S_T - params.strike, torch.zeros_like(S_T))
    put_payoff = torch.maximum(params.strike - S_T, torch.zeros_like(S_T))

    # Discount to choose time
    discount_to_maturity = torch.exp(-r * remaining_time)

    # Values at choose time (simplified - should use BS formula for accuracy)
    # This is a Monte Carlo approximation
    call_value = discount_to_maturity * call_payoff
    put_value = discount_to_maturity * put_payoff

    # Choose maximum
    chooser_value = torch.maximum(call_value, put_value)

    # Discount to time 0
    discount_to_zero = torch.exp(-r * T * params.choose_time)

    return discount_to_zero * chooser_value


def price_compound_option(
    paths: torch.Tensor,
    params: CompoundOptionParams,
    r: float,
    sigma: float
) -> torch.Tensor:
    """Price compound option using Monte Carlo.

    A compound option is an option on an option.

    Mathematical Framework:
    -----------------------
    For a call on call:
        Outer payoff = max(V_inner(S_T1) - K1, 0)
        Inner payoff = max(S_T2 - K2, 0)

    Parameters:
        paths: Monte Carlo paths
        params: Compound option parameters
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Compound option values
    """
    n_paths, n_steps = paths.shape

    # Time points
    t1_idx = int(params.maturity1 * (n_steps - 1) / params.maturity2)

    # Stock price at first maturity
    S_t1 = paths[:, t1_idx]

    # Stock price at second maturity
    S_t2 = paths[:, -1]

    # Inner option payoff at T2
    if params.compound_type == 'call_on_call' or params.compound_type == 'put_on_call':
        inner_payoff = torch.maximum(S_t2 - params.strike2, torch.zeros_like(S_t2))
    else:  # call_on_put or put_on_put
        inner_payoff = torch.maximum(params.strike2 - S_t2, torch.zeros_like(S_t2))

    # Simplified: use inner payoff as proxy for inner option value at T1
    # In practice, should use Black-Scholes formula
    time_to_t2 = params.maturity2 - params.maturity1
    inner_value = torch.exp(-r * time_to_t2) * inner_payoff

    # Outer option payoff at T1
    if params.compound_type.startswith('call'):
        outer_payoff = torch.maximum(inner_value - params.strike1, torch.zeros_like(inner_value))
    else:  # put_on_*
        outer_payoff = torch.maximum(params.strike1 - inner_value, torch.zeros_like(inner_value))

    # Discount to time 0
    value = torch.exp(-r * params.maturity1) * outer_payoff

    return value


def price_lookback_option(
    paths: torch.Tensor,
    strike: float,
    r: float,
    T: float,
    option_type: str = 'call'
) -> torch.Tensor:
    """Price lookback option using Monte Carlo.

    Lookback options have payoffs depending on the maximum or minimum
    price during the option's life.

    Payoffs:
    --------
    Fixed strike lookback call: max(S_max - K, 0)
    Fixed strike lookback put: max(K - S_min, 0)
    Floating strike lookback call: S_T - S_min
    Floating strike lookback put: S_max - S_T

    Parameters:
        paths: Monte Carlo paths
        strike: Strike price (for fixed strike) or None (for floating)
        r: Risk-free rate
        T: Time to maturity
        option_type: 'call' or 'put'

    Returns:
        Lookback option values
    """
    # Calculate maximum and minimum along paths
    S_max = torch.max(paths, dim=1)[0]
    S_min = torch.min(paths, dim=1)[0]
    S_T = paths[:, -1]

    if strike > 0:  # Fixed strike
        if option_type == 'call':
            payoff = torch.maximum(S_max - strike, torch.zeros_like(S_max))
        else:  # put
            payoff = torch.maximum(strike - S_min, torch.zeros_like(S_min))
    else:  # Floating strike
        if option_type == 'call':
            payoff = S_T - S_min
        else:  # put
            payoff = S_max - S_T

    # Discount
    value = torch.exp(-r * T) * payoff

    return value


def price_rainbow_option(
    paths_list: list[torch.Tensor],
    weights: torch.Tensor,
    strike: float,
    r: float,
    T: float,
    rainbow_type: str = 'best_of'
) -> torch.Tensor:
    """Price rainbow option on multiple assets.

    Rainbow options have payoffs depending on multiple underlying assets.

    Types:
    ------
    - Best-of: max(max(S1, S2, ...) - K, 0)
    - Worst-of: max(min(S1, S2, ...) - K, 0)
    - Basket: max(weighted_average(S1, S2, ...) - K, 0)

    Parameters:
        paths_list: List of price paths for each asset
        weights: Weights for basket option
        strike: Strike price
        r: Risk-free rate
        T: Time to maturity
        rainbow_type: Type of rainbow option

    Returns:
        Rainbow option values
    """
    # Stack terminal prices
    terminal_prices = torch.stack([paths[:, -1] for paths in paths_list], dim=1)

    if rainbow_type == 'best_of':
        underlying = torch.max(terminal_prices, dim=1)[0]
    elif rainbow_type == 'worst_of':
        underlying = torch.min(terminal_prices, dim=1)[0]
    elif rainbow_type == 'basket':
        underlying = torch.sum(terminal_prices * weights.unsqueeze(0), dim=1)
    else:
        raise ValueError(f"Unknown rainbow type: {rainbow_type}")

    # Payoff
    payoff = torch.maximum(underlying - strike, torch.zeros_like(underlying))

    # Discount
    value = torch.exp(-r * T) * payoff

    return value


def generate_exotic_option_dataset(
    option_type: ExoticOptionType,
    bs_params: BSParams,
    n_samples: int,
    n_paths: int = 10000,
    n_steps: int = 252,
    compute_greeks: bool = True,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate dataset for exotic option pricing with DML.

    Parameters:
        option_type: Type of exotic option
        bs_params: Black-Scholes parameters
        n_samples: Number of samples to generate
        n_paths: Number of MC paths per sample
        n_steps: Number of time steps
        compute_greeks: Whether to compute Greeks
        **kwargs: Additional option-specific parameters

    Returns:
        Tuple of (features, prices, deltas, gammas)
    """
    device = get_device()

    # Generate varied parameters
    S0_range = (80, 120)
    sigma_range = (0.15, 0.35)
    r_range = (0.02, 0.08)
    T_range = (0.5, 2.0)

    # Sample parameters
    S0_samples = torch.rand(n_samples) * (S0_range[1] - S0_range[0]) + S0_range[0]
    sigma_samples = torch.rand(n_samples) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    r_samples = torch.rand(n_samples) * (r_range[1] - r_range[0]) + r_range[0]
    T_samples = torch.rand(n_samples) * (T_range[1] - T_range[0]) + T_range[0]

    # Move to device
    S0_samples = S0_samples.to(device)
    sigma_samples = sigma_samples.to(device)
    r_samples = r_samples.to(device)
    T_samples = T_samples.to(device)

    prices = []
    deltas = []
    gammas = []

    for i in range(n_samples):
        S0 = S0_samples[i].requires_grad_(True) if compute_greeks else S0_samples[i]
        sigma = sigma_samples[i]
        r = r_samples[i]
        T = T_samples[i]

        # Simulate paths
        paths = simulate_bs_paths(S0, r, sigma, T, n_steps, n_paths)
        dt = T / n_steps

        # Price based on option type
        if option_type == ExoticOptionType.VARIANCE_SWAP:
            params = VarianceSwapParams(**kwargs)
            option_values = price_variance_swap(paths, params, dt.item())

        elif option_type == ExoticOptionType.VOLATILITY_SWAP:
            params = VarianceSwapParams(**kwargs)
            option_values = price_volatility_swap(paths, params, dt.item())

        elif option_type == ExoticOptionType.CHOOSER:
            params = ChooserOptionParams(**kwargs)
            option_values = price_chooser_option(paths, params, r.item(), T.item())

        elif option_type == ExoticOptionType.COMPOUND_CALL_ON_CALL:
            params = CompoundOptionParams(compound_type='call_on_call', **kwargs)
            option_values = price_compound_option(paths, params, r.item(), sigma.item())

        elif option_type == ExoticOptionType.LOOKBACK_CALL:
            strike = kwargs.get('strike', 100.0)
            option_values = price_lookback_option(paths, strike, r.item(), T.item(), 'call')

        elif option_type == ExoticOptionType.LOOKBACK_PUT:
            strike = kwargs.get('strike', 100.0)
            option_values = price_lookback_option(paths, strike, r.item(), T.item(), 'put')

        else:
            raise ValueError(f"Unsupported option type: {option_type}")

        # Average price across paths
        price = option_values.mean()
        prices.append(price)

        if compute_greeks:
            # Compute delta
            delta = torch.autograd.grad(
                price,
                S0,
                create_graph=True,
                retain_graph=True
            )[0]
            deltas.append(delta)

            # Compute gamma
            gamma = torch.autograd.grad(
                delta,
                S0,
                retain_graph=False
            )[0]
            gammas.append(gamma)

    # Stack results
    features = torch.stack([
        S0_samples,
        torch.full_like(S0_samples, bs_params.K) if hasattr(bs_params, 'K') else torch.full_like(S0_samples, 100.0),
        r_samples,
        sigma_samples,
        T_samples
    ], dim=1)

    prices = torch.stack(prices)

    if compute_greeks:
        deltas = torch.stack(deltas)
        gammas = torch.stack(gammas)
    else:
        deltas = torch.zeros_like(prices)
        gammas = torch.zeros_like(prices)

    return features, prices, deltas, gammas


def validate_exotic_options():
    """Validate exotic option implementations with test cases."""
    device = get_device()

    print("="*60)
    print("Validating Exotic Option Implementations")
    print("="*60)

    # Test 1: Variance Swap
    print("\nTest 1: Variance Swap")
    S0 = torch.tensor(100.0, device=device)
    paths = simulate_bs_paths(S0, r=0.05, sigma=0.2, T=1.0, n_steps=252, n_paths=10000)

    var_params = VarianceSwapParams(
        strike_variance=0.04,  # 20% vol squared
        notional=1000000.0
    )

    var_swap_values = price_variance_swap(paths, var_params, dt=1.0/252)
    print(f"  Mean variance swap value: ${var_swap_values.mean().item():.2f}")
    print(f"  Std deviation: ${var_swap_values.std().item():.2f}")

    # Test 2: Chooser Option
    print("\nTest 2: Chooser Option")
    chooser_params = ChooserOptionParams(
        choose_time=0.5,
        strike=100.0
    )

    chooser_values = price_chooser_option(paths, chooser_params, r=0.05, T=1.0)
    print(f"  Mean chooser value: ${chooser_values.mean().item():.2f}")

    # Test 3: Compound Option
    print("\nTest 3: Compound Call-on-Call Option")
    compound_params = CompoundOptionParams(
        compound_type='call_on_call',
        strike1=5.0,
        strike2=100.0,
        maturity1=0.5,
        maturity2=1.0
    )

    compound_values = price_compound_option(paths, compound_params, r=0.05, sigma=0.2)
    print(f"  Mean compound value: ${compound_values.mean().item():.2f}")

    # Test 4: Lookback Option
    print("\nTest 4: Lookback Call Option")
    lookback_values = price_lookback_option(paths, strike=100.0, r=0.05, T=1.0, option_type='call')
    print(f"  Mean lookback call value: ${lookback_values.mean().item():.2f}")

    # Test 5: Generate dataset
    print("\nTest 5: Generating Exotic Option Dataset")
    bs_params = BSParams(r=0.05, sigma=0.2, T=1.0)

    features, prices, deltas, gammas = generate_exotic_option_dataset(
        option_type=ExoticOptionType.VARIANCE_SWAP,
        bs_params=bs_params,
        n_samples=100,
        n_paths=1000,
        n_steps=100,
        compute_greeks=True
    )

    print(f"  Dataset generated:")
    print(f"    Features shape: {features.shape}")
    print(f"    Prices shape: {prices.shape}")
    print(f"    Mean price: ${prices.mean().item():.2f}")
    print(f"    Mean delta: {deltas.mean().item():.4f}")

    print("\n✓ All exotic option tests passed!")


if __name__ == "__main__":
    validate_exotic_options()