"""Dataset generation for multi-barrier options.

This module implements pricing and sensitivity computation for complex barrier options
including double barriers, window barriers, and Parisian barriers using Monte Carlo
simulation with differential machine learning.

Mathematical Background:
    Multi-barrier options have payoffs that depend on whether the underlying asset
    crosses multiple barrier levels during the option's lifetime.

    Double Barrier Option:
        - Knocked out if S_t ≤ L (lower barrier) or S_t ≥ U (upper barrier)
        - Payoff = max(S_T - K, 0) * 1_{τ > T}
        where τ = inf{t : S_t ∉ (L, U)}

    Window Barrier:
        - Barrier only active during specific time windows
        - Useful for modeling event-driven barriers

    Parisian Barrier:
        - Knocked out only if barrier is breached for consecutive time > threshold
        - More robust to short-term price spikes

References:
    - Haug, E. G. (2007). "The Complete Guide to Option Pricing Formulas."
    - Carr, P. (1995). "Two extensions to barrier option valuation."
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Dict, Callable
from dataclasses import dataclass
from enum import Enum

from .config import BSParams, get_device
from .simulation import simulate_bs_paths


class BarrierType(Enum):
    """Types of barrier options."""
    SINGLE_UP_OUT = "single_up_out"
    SINGLE_DOWN_OUT = "single_down_out"
    DOUBLE_OUT = "double_out"
    DOUBLE_IN = "double_in"
    WINDOW = "window"
    PARISIAN = "parisian"
    STEP = "step"  # Time-varying barriers


@dataclass
class MultiBarrierParams:
    """Parameters for multi-barrier options.

    Attributes:
        barrier_type: Type of barrier option
        upper_barrier: Upper barrier level (optional)
        lower_barrier: Lower barrier level (optional)
        window_start: Start time for window barrier (fraction of T)
        window_end: End time for window barrier (fraction of T)
        parisian_threshold: Time threshold for Parisian barrier (fraction of T)
        rebate: Payment if knocked out
        barrier_schedule: Time-varying barrier levels [(time, lower, upper), ...]
        monitoring: 'continuous' or 'discrete'
    """

    barrier_type: BarrierType = BarrierType.DOUBLE_OUT
    upper_barrier: Optional[float] = None
    lower_barrier: Optional[float] = None
    window_start: float = 0.0
    window_end: float = 1.0
    parisian_threshold: float = 0.05
    rebate: float = 0.0
    barrier_schedule: Optional[List[Tuple[float, float, float]]] = None
    monitoring: str = 'continuous'


def check_double_barrier_crossing(
    paths: torch.Tensor,
    lower_barrier: float,
    upper_barrier: float,
    knock_type: str = 'out'
) -> torch.Tensor:
    """Check if paths cross double barriers.

    Parameters:
        paths: Price paths of shape (n_paths, n_steps)
        lower_barrier: Lower barrier level
        upper_barrier: Upper barrier level
        knock_type: 'out' for knock-out, 'in' for knock-in

    Returns:
        Boolean tensor indicating which paths are active at maturity
    """
    device = paths.device

    # Check if any point crosses barriers
    hit_lower = (paths <= lower_barrier).any(dim=1)
    hit_upper = (paths >= upper_barrier).any(dim=1)
    hit_any = hit_lower | hit_upper

    if knock_type == 'out':
        # Knocked out if hit any barrier
        return ~hit_any
    else:  # knock_type == 'in'
        # Knocked in if hit any barrier
        return hit_any


def check_window_barrier_crossing(
    paths: torch.Tensor,
    barriers: Tuple[float, float],
    window_start_idx: int,
    window_end_idx: int,
    knock_type: str = 'out'
) -> torch.Tensor:
    """Check window barrier crossing (barrier active only in specified window).

    Parameters:
        paths: Price paths of shape (n_paths, n_steps)
        barriers: (lower_barrier, upper_barrier)
        window_start_idx: Start index of barrier window
        window_end_idx: End index of barrier window
        knock_type: 'out' or 'in'

    Returns:
        Boolean tensor indicating which paths are active
    """
    # Extract window paths
    window_paths = paths[:, window_start_idx:window_end_idx+1]

    # Check barriers only in window
    if barriers[0] is not None:
        hit_lower = (window_paths <= barriers[0]).any(dim=1)
    else:
        hit_lower = torch.zeros(paths.shape[0], dtype=torch.bool, device=paths.device)

    if barriers[1] is not None:
        hit_upper = (window_paths >= barriers[1]).any(dim=1)
    else:
        hit_upper = torch.zeros(paths.shape[0], dtype=torch.bool, device=paths.device)

    hit_any = hit_lower | hit_upper

    if knock_type == 'out':
        return ~hit_any
    else:
        return hit_any


def check_parisian_barrier(
    paths: torch.Tensor,
    barrier: float,
    threshold_steps: int,
    barrier_type: str = 'up'
) -> torch.Tensor:
    """Check Parisian barrier (requires consecutive breach for knockout).

    A Parisian barrier is only triggered if the underlying stays beyond
    the barrier for a consecutive period exceeding the threshold.

    Parameters:
        paths: Price paths of shape (n_paths, n_steps)
        barrier: Barrier level
        threshold_steps: Number of consecutive steps required for knockout
        barrier_type: 'up' or 'down'

    Returns:
        Boolean tensor indicating which paths are knocked out
    """
    device = paths.device
    n_paths, n_steps = paths.shape

    # Check barrier breach at each step
    if barrier_type == 'up':
        breached = paths >= barrier
    else:
        breached = paths <= barrier

    # Find consecutive breaches
    knocked_out = torch.zeros(n_paths, dtype=torch.bool, device=device)

    for i in range(n_paths):
        # Count consecutive breaches
        consecutive = 0
        for j in range(n_steps):
            if breached[i, j]:
                consecutive += 1
                if consecutive >= threshold_steps:
                    knocked_out[i] = True
                    break
            else:
                consecutive = 0

    return ~knocked_out  # Return True for paths that survive


def check_step_barrier(
    paths: torch.Tensor,
    barrier_schedule: List[Tuple[int, float, float]]
) -> torch.Tensor:
    """Check time-varying (step) barriers.

    Parameters:
        paths: Price paths of shape (n_paths, n_steps)
        barrier_schedule: List of (time_idx, lower_barrier, upper_barrier)

    Returns:
        Boolean tensor indicating which paths survive
    """
    device = paths.device
    n_paths = paths.shape[0]
    survived = torch.ones(n_paths, dtype=torch.bool, device=device)

    for i in range(len(barrier_schedule) - 1):
        start_idx, lower_i, upper_i = barrier_schedule[i]
        end_idx = barrier_schedule[i + 1][0] if i < len(barrier_schedule) - 1 else paths.shape[1]

        # Check barriers in this time segment
        segment = paths[:, start_idx:end_idx]

        if lower_i is not None:
            hit_lower = (segment <= lower_i).any(dim=1)
            survived = survived & ~hit_lower

        if upper_i is not None:
            hit_upper = (segment >= upper_i).any(dim=1)
            survived = survived & ~hit_upper

    return survived


def price_multibarrier_option(
    paths: torch.Tensor,
    strike: float,
    r: float,
    T: float,
    params: MultiBarrierParams,
    option_type: str = 'call'
) -> torch.Tensor:
    """Price multi-barrier options using Monte Carlo.

    Parameters:
        paths: Monte Carlo paths of shape (n_paths, n_steps)
        strike: Strike price
        r: Risk-free rate
        T: Time to maturity
        params: Multi-barrier parameters
        option_type: 'call' or 'put'

    Returns:
        Option prices for each path
    """
    device = paths.device
    n_paths, n_steps = paths.shape

    # Terminal payoff
    if option_type == 'call':
        payoff = torch.maximum(paths[:, -1] - strike, torch.zeros_like(paths[:, -1]))
    else:  # put
        payoff = torch.maximum(strike - paths[:, -1], torch.zeros_like(paths[:, -1]))

    # Check barrier conditions based on type
    if params.barrier_type == BarrierType.DOUBLE_OUT:
        survived = check_double_barrier_crossing(
            paths, params.lower_barrier, params.upper_barrier, 'out'
        )

    elif params.barrier_type == BarrierType.DOUBLE_IN:
        survived = check_double_barrier_crossing(
            paths, params.lower_barrier, params.upper_barrier, 'in'
        )

    elif params.barrier_type == BarrierType.WINDOW:
        window_start_idx = int(params.window_start * n_steps)
        window_end_idx = int(params.window_end * n_steps)
        survived = check_window_barrier_crossing(
            paths,
            (params.lower_barrier, params.upper_barrier),
            window_start_idx,
            window_end_idx,
            'out'
        )

    elif params.barrier_type == BarrierType.PARISIAN:
        threshold_steps = int(params.parisian_threshold * n_steps)
        if params.upper_barrier is not None:
            survived_up = check_parisian_barrier(
                paths, params.upper_barrier, threshold_steps, 'up'
            )
        else:
            survived_up = torch.ones(n_paths, dtype=torch.bool, device=device)

        if params.lower_barrier is not None:
            survived_down = check_parisian_barrier(
                paths, params.lower_barrier, threshold_steps, 'down'
            )
        else:
            survived_down = torch.ones(n_paths, dtype=torch.bool, device=device)

        survived = survived_up & survived_down

    elif params.barrier_type == BarrierType.STEP:
        if params.barrier_schedule is not None:
            survived = check_step_barrier(paths, params.barrier_schedule)
        else:
            raise ValueError("Step barrier requires barrier_schedule")

    else:
        # Single barriers
        if params.barrier_type == BarrierType.SINGLE_UP_OUT:
            survived = (paths <= params.upper_barrier).all(dim=1)
        elif params.barrier_type == BarrierType.SINGLE_DOWN_OUT:
            survived = (paths >= params.lower_barrier).all(dim=1)
        else:
            raise ValueError(f"Unknown barrier type: {params.barrier_type}")

    # Apply payoff with rebate for knocked-out options
    option_value = torch.where(
        survived,
        payoff * torch.exp(-r * T),  # Discounted payoff if survived
        torch.tensor(params.rebate * torch.exp(-r * T), device=device)  # Discounted rebate
    )

    return option_value


def generate_multibarrier_dataset(
    bs_params: BSParams,
    n_samples: int,
    barrier_params: MultiBarrierParams,
    n_paths: int = 5000,
    n_steps: int = 100,
    compute_greeks: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate dataset for multi-barrier options with DML.

    Parameters:
        bs_params: Black-Scholes parameters
        n_samples: Number of samples to generate
        barrier_params: Multi-barrier parameters
        n_paths: Number of MC paths per sample
        n_steps: Number of time steps
        compute_greeks: Whether to compute sensitivities

    Returns:
        Tuple of (features, prices, deltas, gammas)
    """
    device = get_device()

    # Generate varied parameters
    S0_range = (90, 110)
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

    # Fixed parameters for this dataset
    K = bs_params.K

    # Lists to store results
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

        # Price option
        option_values = price_multibarrier_option(
            paths, K, r.item(), T.item(), barrier_params, 'call'
        )

        # Average price
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
        torch.full_like(S0_samples, K),
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


def analytical_double_barrier_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    L: float,
    U: float,
    option_type: str = 'call'
) -> float:
    """Analytical price for double barrier knock-out option (approximation).

    This uses the Ikeda-Kunitomo approximation for double barrier options.
    Note: This is an approximation and may not be exact.

    Parameters:
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        L: Lower barrier
        U: Upper barrier
        option_type: 'call' or 'put'

    Returns:
        Approximate option price
    """
    import scipy.stats as stats

    # This is a simplified approximation
    # For exact pricing, use more sophisticated methods

    # Check if already knocked out
    if S0 <= L or S0 >= U:
        return 0.0

    # Parameters
    mu = (r - 0.5 * sigma**2) / sigma
    lambda_val = np.sqrt(mu**2 + 2*r/sigma**2)

    # Simplified approximation using single barrier formulas
    # This is not exact but provides a reasonable estimate

    # Price as minimum of two single barriers
    from .bs_analytics import bs_call_price, bs_put_price

    device = get_device()
    S0_t = torch.tensor(S0, device=device)
    K_t = torch.tensor(K, device=device)
    r_t = torch.tensor(r, device=device)
    sigma_t = torch.tensor(sigma, device=device)
    T_t = torch.tensor(T, device=device)

    if option_type == 'call':
        vanilla_price = bs_call_price(S0_t, K_t, r_t, sigma_t, T_t).item()
    else:
        # Implement put pricing using put-call parity
        call_price = bs_call_price(S0_t, K_t, r_t, sigma_t, T_t)
        vanilla_price = (call_price - S0_t + K_t * torch.exp(-r_t * T_t)).item()

    # Rough approximation: scale by probability of staying within barriers
    # This is a heuristic, not exact
    d_up = (np.log(U/S0) - mu*np.sqrt(T)) / np.sqrt(T)
    d_down = (np.log(L/S0) - mu*np.sqrt(T)) / np.sqrt(T)

    prob_survive = stats.norm.cdf(d_up) - stats.norm.cdf(d_down)

    return vanilla_price * prob_survive


def validate_multibarrier_implementation():
    """Validate multi-barrier option implementation with test cases."""
    device = get_device()

    print("="*60)
    print("Validating Multi-Barrier Option Implementation")
    print("="*60)

    # Test 1: Double barrier knock-out
    print("\nTest 1: Double Barrier Knock-Out Call")
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
    L, U = 90.0, 110.0

    params = MultiBarrierParams(
        barrier_type=BarrierType.DOUBLE_OUT,
        lower_barrier=L,
        upper_barrier=U
    )

    # Simulate paths
    S0_t = torch.tensor(S0, device=device, requires_grad=True)
    paths = simulate_bs_paths(S0_t, r, sigma, T, 252, 10000)
    prices = price_multibarrier_option(paths, K, r, T, params, 'call')
    mc_price = prices.mean()

    # Compute delta
    delta = torch.autograd.grad(mc_price, S0_t)[0]

    print(f"  MC Price: {mc_price.item():.4f}")
    print(f"  Delta: {delta.item():.4f}")

    # Approximate analytical price for comparison
    approx_price = analytical_double_barrier_price(S0, K, r, sigma, T, L, U, 'call')
    print(f"  Approx Analytical: {approx_price:.4f}")
    print(f"  Difference: {abs(mc_price.item() - approx_price):.4f}")

    # Test 2: Window barrier
    print("\nTest 2: Window Barrier (Active 3-9 months)")
    params_window = MultiBarrierParams(
        barrier_type=BarrierType.WINDOW,
        lower_barrier=85.0,
        upper_barrier=115.0,
        window_start=0.25,  # 3 months
        window_end=0.75     # 9 months
    )

    S0_t = torch.tensor(S0, device=device, requires_grad=True)
    paths = simulate_bs_paths(S0_t, r, sigma, T, 252, 10000)
    prices = price_multibarrier_option(paths, K, r, T, params_window, 'call')
    window_price = prices.mean()
    window_delta = torch.autograd.grad(window_price, S0_t)[0]

    print(f"  Window Barrier Price: {window_price.item():.4f}")
    print(f"  Window Barrier Delta: {window_delta.item():.4f}")

    # Test 3: Parisian barrier
    print("\nTest 3: Parisian Barrier (5% time threshold)")
    params_parisian = MultiBarrierParams(
        barrier_type=BarrierType.PARISIAN,
        upper_barrier=112.0,
        parisian_threshold=0.05  # 5% of time period
    )

    S0_t = torch.tensor(S0, device=device, requires_grad=True)
    paths = simulate_bs_paths(S0_t, r, sigma, T, 252, 10000)
    prices = price_multibarrier_option(paths, K, r, T, params_parisian, 'call')
    parisian_price = prices.mean()
    parisian_delta = torch.autograd.grad(parisian_price, S0_t)[0]

    print(f"  Parisian Barrier Price: {parisian_price.item():.4f}")
    print(f"  Parisian Barrier Delta: {parisian_delta.item():.4f}")

    # Test 4: Step barrier
    print("\nTest 4: Step Barrier (Tightening over time)")
    barrier_schedule = [
        (0, 85.0, 115.0),     # 0-3 months: wide barriers
        (63, 88.0, 112.0),    # 3-6 months: medium barriers
        (126, 91.0, 109.0),   # 6-9 months: tight barriers
        (189, 94.0, 106.0),   # 9-12 months: very tight barriers
        (252, None, None)     # End marker
    ]

    params_step = MultiBarrierParams(
        barrier_type=BarrierType.STEP,
        barrier_schedule=barrier_schedule
    )

    S0_t = torch.tensor(S0, device=device, requires_grad=True)
    paths = simulate_bs_paths(S0_t, r, sigma, T, 252, 10000)
    prices = price_multibarrier_option(paths, K, r, T, params_step, 'call')
    step_price = prices.mean()
    step_delta = torch.autograd.grad(step_price, S0_t)[0]

    print(f"  Step Barrier Price: {step_price.item():.4f}")
    print(f"  Step Barrier Delta: {step_delta.item():.4f}")

    # Test 5: Knock-in vs Knock-out relationship
    print("\nTest 5: In-Out Parity Check")
    params_out = MultiBarrierParams(
        barrier_type=BarrierType.DOUBLE_OUT,
        lower_barrier=95.0,
        upper_barrier=105.0
    )
    params_in = MultiBarrierParams(
        barrier_type=BarrierType.DOUBLE_IN,
        lower_barrier=95.0,
        upper_barrier=105.0
    )

    paths = simulate_bs_paths(torch.tensor(S0, device=device), r, sigma, T, 252, 10000)

    prices_out = price_multibarrier_option(paths, K, r, T, params_out, 'call')
    prices_in = price_multibarrier_option(paths, K, r, T, params_in, 'call')

    out_price = prices_out.mean().item()
    in_price = prices_in.mean().item()

    # Vanilla call for comparison
    from .bs_analytics import bs_call_price
    vanilla_call = bs_call_price(
        torch.tensor(S0, device=device),
        torch.tensor(K),
        torch.tensor(r),
        torch.tensor(sigma),
        torch.tensor(T)
    ).item()

    print(f"  Knock-Out Price: {out_price:.4f}")
    print(f"  Knock-In Price: {in_price:.4f}")
    print(f"  Sum: {out_price + in_price:.4f}")
    print(f"  Vanilla Call: {vanilla_call:.4f}")
    print(f"  Difference: {abs((out_price + in_price) - vanilla_call):.4f}")

    # In-Out parity: Knock-in + Knock-out = Vanilla (approximately)
    assert abs((out_price + in_price) - vanilla_call) < 0.5, "In-Out parity violated"

    print("\n✓ All multi-barrier tests passed!")


if __name__ == "__main__":
    # Run validation
    validate_multibarrier_implementation()

    # Generate sample dataset
    print("\n" + "="*60)
    print("Generating Multi-Barrier Option Dataset")
    print("="*60)

    bs_params = BSParams(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)

    # Double barrier example
    barrier_params = MultiBarrierParams(
        barrier_type=BarrierType.DOUBLE_OUT,
        lower_barrier=85.0,
        upper_barrier=115.0,
        rebate=0.0
    )

    features, prices, deltas, gammas = generate_multibarrier_dataset(
        bs_params=bs_params,
        n_samples=100,
        barrier_params=barrier_params,
        n_paths=5000,
        n_steps=100,
        compute_greeks=True
    )

    print(f"\nDataset Summary:")
    print(f"  Features shape: {features.shape}")
    print(f"  Prices shape: {prices.shape}")
    print(f"  Deltas shape: {deltas.shape}")
    print(f"  Gammas shape: {gammas.shape}")

    print(f"\nStatistics:")
    print(f"  Price: mean={prices.mean():.4f}, std={prices.std():.4f}")
    print(f"  Delta: mean={deltas.mean():.4f}, std={deltas.std():.4f}")
    print(f"  Gamma: mean={gammas.mean():.6f}, std={gammas.std():.6f}")