"""Dataset generation for American options using Longstaff-Schwartz method.

This module implements American option pricing using the Least Squares Monte Carlo
(LSM) method, also known as the Longstaff-Schwartz algorithm, combined with
differential machine learning for sensitivity computation.

Mathematical Background:
    The American option pricing problem involves finding the optimal exercise boundary.
    At each time step t, the holder must decide whether to exercise immediately
    (receiving intrinsic value) or continue holding (receiving continuation value).

    The optimal exercise decision:
        V(S_t) = max(h(S_t), C(S_t))

    where:
        - h(S_t) is the intrinsic value (payoff if exercised now)
        - C(S_t) is the continuation value (expected discounted future payoff)

    The LSM method approximates C(S_t) using regression on basis functions.

References:
    - Longstaff, F. A., & Schwartz, E. S. (2001). "Valuing American options by simulation:
      A simple least-squares approach." Review of Financial Studies, 14(1), 113-147.
    - Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering." Springer.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

from .config import BSParams, get_device
from .simulation import simulate_bs_paths


@dataclass
class AmericanOptionParams:
    """Parameters for American option pricing.

    Attributes:
        exercise_type: 'call' or 'put'
        n_basis: Number of basis functions for regression (default: 4)
        basis_type: Type of basis functions ('laguerre', 'power', 'hermite')
    """

    exercise_type: str = 'put'
    n_basis: int = 4
    basis_type: str = 'laguerre'


def laguerre_polynomials(x: torch.Tensor, n: int) -> torch.Tensor:
    """Generate weighted Laguerre polynomials up to degree n.

    Laguerre polynomials are particularly well-suited for American option pricing
    as they naturally handle the exponential decay in stock prices.

    L_0(x) = e^(-x/2)
    L_1(x) = e^(-x/2) * (1 - x)
    L_2(x) = e^(-x/2) * (1 - 2x + x^2/2)
    L_3(x) = e^(-x/2) * (1 - 3x + 3x^2/2 - x^3/6)

    Parameters:
        x: Input tensor of stock prices
        n: Maximum degree of polynomial

    Returns:
        Tensor of shape (len(x), n+1) containing basis functions
    """
    device = x.device
    basis = torch.zeros(x.shape[0], n + 1, device=device)

    # Weighted by e^(-x/2) for numerical stability
    weight = torch.exp(-x / 2)

    # Generate polynomials recursively
    basis[:, 0] = weight
    if n >= 1:
        basis[:, 1] = weight * (1 - x)
    if n >= 2:
        basis[:, 2] = weight * (1 - 2*x + x**2/2)
    if n >= 3:
        basis[:, 3] = weight * (1 - 3*x + 3*x**2/2 - x**3/6)

    # Higher order terms if needed
    for i in range(4, n + 1):
        basis[:, i] = ((2*i - 1 - x) * basis[:, i-1] - (i-1) * basis[:, i-2]) / i

    return basis


def power_basis(x: torch.Tensor, n: int) -> torch.Tensor:
    """Generate power basis functions.

    Simple polynomial basis: 1, x, x^2, x^3, ...

    Parameters:
        x: Input tensor
        n: Maximum degree

    Returns:
        Tensor of shape (len(x), n+1) containing power basis
    """
    device = x.device
    basis = torch.zeros(x.shape[0], n + 1, device=device)

    for i in range(n + 1):
        basis[:, i] = x ** i

    return basis


def hermite_polynomials(x: torch.Tensor, n: int) -> torch.Tensor:
    """Generate Hermite polynomials (physicists' version).

    Hermite polynomials are orthogonal with respect to the Gaussian weight function.

    H_0(x) = 1
    H_1(x) = 2x
    H_2(x) = 4x^2 - 2
    H_3(x) = 8x^3 - 12x

    Parameters:
        x: Input tensor (typically normalized)
        n: Maximum degree

    Returns:
        Tensor of shape (len(x), n+1) containing Hermite basis
    """
    device = x.device
    basis = torch.zeros(x.shape[0], n + 1, device=device)

    basis[:, 0] = 1
    if n >= 1:
        basis[:, 1] = 2 * x
    if n >= 2:
        basis[:, 2] = 4 * x**2 - 2
    if n >= 3:
        basis[:, 3] = 8 * x**3 - 12 * x

    # Recurrence relation: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    for i in range(4, n + 1):
        basis[:, i] = 2 * x * basis[:, i-1] - 2 * (i-1) * basis[:, i-2]

    return basis


def longstaff_schwartz_american(
    paths: torch.Tensor,
    strike: float,
    r: float,
    dt: float,
    option_type: str = 'put',
    n_basis: int = 4,
    basis_type: str = 'laguerre'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Longstaff-Schwartz algorithm for American option pricing.

    This implementation uses regression on basis functions to estimate
    the continuation value at each time step, working backwards from maturity.

    Parameters:
        paths: Monte Carlo paths of shape (n_paths, n_steps + 1)
        strike: Strike price
        r: Risk-free rate
        dt: Time step size
        option_type: 'call' or 'put'
        n_basis: Number of basis functions
        basis_type: Type of basis ('laguerre', 'power', 'hermite')

    Returns:
        Tuple of:
            - prices: American option prices for each path
            - exercise_times: Optimal exercise time for each path
            - continuation_values: Estimated continuation values
    """
    device = paths.device
    n_paths, n_steps = paths.shape[0], paths.shape[1] - 1

    # Payoff function
    if option_type == 'put':
        payoff = lambda s: torch.maximum(strike - s, torch.zeros_like(s))
    else:  # call
        payoff = lambda s: torch.maximum(s - strike, torch.zeros_like(s))

    # Initialize cash flows at maturity
    cash_flows = payoff(paths[:, -1])
    exercise_times = torch.full((n_paths,), n_steps, dtype=torch.float32, device=device)

    # Basis function selector
    if basis_type == 'laguerre':
        basis_func = lambda x: laguerre_polynomials(x / strike, n_basis)  # Normalize by strike
    elif basis_type == 'power':
        basis_func = lambda x: power_basis(x / strike, n_basis)
    elif basis_type == 'hermite':
        # Normalize to standard normal-like range
        mean_price = paths.mean()
        std_price = paths.std()
        basis_func = lambda x: hermite_polynomials((x - mean_price) / std_price, n_basis)
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")

    # Store continuation values for analysis
    continuation_values = torch.zeros_like(paths)

    # Backward induction
    for t in range(n_steps - 1, 0, -1):
        # Current stock prices
        S_t = paths[:, t]

        # Intrinsic value at time t
        intrinsic = payoff(S_t)

        # Only consider in-the-money paths for regression
        itm_mask = intrinsic > 0

        if itm_mask.sum() > 0:
            # Basis functions for ITM paths
            X = basis_func(S_t[itm_mask])

            # Discounted future cash flows for ITM paths
            Y = cash_flows[itm_mask] * torch.exp(-r * dt * (exercise_times[itm_mask] - t))

            # Least squares regression
            # Solve X @ beta = Y using normal equations
            XtX = X.T @ X
            XtY = X.T @ Y

            # Add small regularization for numerical stability
            reg = 1e-8 * torch.eye(n_basis, device=device)
            beta = torch.linalg.solve(XtX + reg, XtY)

            # Estimate continuation value for all paths
            X_all = basis_func(S_t)
            cont_value = X_all @ beta
            continuation_values[:, t] = cont_value

            # Exercise decision: exercise if intrinsic > continuation
            exercise_now = itm_mask & (intrinsic > cont_value)

            # Update cash flows and exercise times for those exercising
            cash_flows[exercise_now] = intrinsic[exercise_now]
            exercise_times[exercise_now] = t

    # Compute option value at time 0
    discount_factors = torch.exp(-r * dt * exercise_times)
    prices = cash_flows * discount_factors

    return prices, exercise_times, continuation_values


def generate_american_option_dataset(
    bs_params: BSParams,
    n_samples: int,
    american_params: Optional[AmericanOptionParams] = None,
    compute_greeks: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate dataset for American option pricing with DML.

    Creates a dataset of American option prices and their sensitivities
    using the Longstaff-Schwartz method combined with automatic differentiation
    for Greek calculation.

    Parameters:
        bs_params: Black-Scholes parameters
        n_samples: Number of samples to generate
        american_params: American option specific parameters
        compute_greeks: Whether to compute Greeks (delta, gamma)

    Returns:
        Tuple of:
            - features: Input features (S0, K, r, sigma, T)
            - prices: American option prices
            - deltas: Option deltas (if compute_greeks=True)
            - gammas: Option gammas (if compute_greeks=True)
    """
    device = get_device()

    if american_params is None:
        american_params = AmericanOptionParams()

    # Generate varied parameters for dataset
    S0_range = (80, 120)
    K_range = (90, 110)
    r_range = (0.01, 0.1)
    sigma_range = (0.1, 0.4)
    T_range = (0.25, 2.0)

    # Sample parameters
    S0_samples = torch.rand(n_samples) * (S0_range[1] - S0_range[0]) + S0_range[0]
    K_samples = torch.rand(n_samples) * (K_range[1] - K_range[0]) + K_range[0]
    r_samples = torch.rand(n_samples) * (r_range[1] - r_range[0]) + r_range[0]
    sigma_samples = torch.rand(n_samples) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    T_samples = torch.rand(n_samples) * (T_range[1] - T_range[0]) + T_range[0]

    # Move to device
    S0_samples = S0_samples.to(device)
    K_samples = K_samples.to(device)
    r_samples = r_samples.to(device)
    sigma_samples = sigma_samples.to(device)
    T_samples = T_samples.to(device)

    # Number of time steps and paths for LSM
    n_steps = 50
    n_paths = 1000

    prices = []
    deltas = []
    gammas = []

    for i in range(n_samples):
        S0 = S0_samples[i].requires_grad_(True) if compute_greeks else S0_samples[i]
        K = K_samples[i]
        r = r_samples[i]
        sigma = sigma_samples[i]
        T = T_samples[i]

        # Simulate paths
        dt = T / n_steps
        paths = simulate_bs_paths(
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            n_steps=n_steps,
            n_paths=n_paths
        )

        # Price American option
        option_prices, _, _ = longstaff_schwartz_american(
            paths=paths,
            strike=K.item(),
            r=r.item(),
            dt=dt.item(),
            option_type=american_params.exercise_type,
            n_basis=american_params.n_basis,
            basis_type=american_params.basis_type
        )

        # Average price across paths
        price = option_prices.mean()
        prices.append(price)

        if compute_greeks:
            # Compute delta using automatic differentiation
            delta = torch.autograd.grad(
                price,
                S0,
                create_graph=True,
                retain_graph=True
            )[0]
            deltas.append(delta)

            # Compute gamma (second derivative)
            gamma = torch.autograd.grad(
                delta,
                S0,
                retain_graph=False
            )[0]
            gammas.append(gamma)

    # Stack results
    features = torch.stack([S0_samples, K_samples, r_samples, sigma_samples, T_samples], dim=1)
    prices = torch.stack(prices)

    if compute_greeks:
        deltas = torch.stack(deltas)
        gammas = torch.stack(gammas)
    else:
        deltas = torch.zeros_like(prices)
        gammas = torch.zeros_like(prices)

    return features, prices, deltas, gammas


def finite_difference_american_greeks(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = 'put',
    epsilon: float = 0.01,
    n_paths: int = 10000,
    n_steps: int = 50
) -> Tuple[float, float, float]:
    """Compute American option Greeks using finite differences.

    This function provides a benchmark for validating the automatic differentiation
    approach by computing Greeks using central finite differences.

    Parameters:
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        option_type: 'call' or 'put'
        epsilon: Perturbation size for finite differences
        n_paths: Number of Monte Carlo paths
        n_steps: Number of time steps

    Returns:
        Tuple of (price, delta, gamma)
    """
    device = get_device()

    def price_american(spot):
        """Helper function to price American option at given spot."""
        spot_tensor = torch.tensor(spot, device=device, dtype=torch.float32)
        paths = simulate_bs_paths(spot_tensor, r, sigma, T, n_steps, n_paths)
        prices, _, _ = longstaff_schwartz_american(
            paths, K, r, T/n_steps, option_type
        )
        return prices.mean().item()

    # Central price
    V = price_american(S0)

    # Delta using central difference
    V_up = price_american(S0 + epsilon)
    V_down = price_american(S0 - epsilon)
    delta = (V_up - V_down) / (2 * epsilon)

    # Gamma using central difference
    gamma = (V_up - 2*V + V_down) / (epsilon**2)

    return V, delta, gamma


def validate_american_implementation():
    """Validate the American option implementation against known benchmarks.

    This function runs several test cases with known or approximated solutions
    to ensure the implementation is correct.
    """
    device = get_device()

    # Test case 1: Deep ITM put should have delta close to -1
    print("Test 1: Deep ITM American Put")
    S0, K, r, sigma, T = 50.0, 100.0, 0.05, 0.2, 1.0
    price, delta, gamma = finite_difference_american_greeks(S0, K, r, sigma, T, 'put')
    print(f"  Price: {price:.4f}, Delta: {delta:.4f}, Gamma: {gamma:.6f}")
    assert abs(delta + 1) < 0.1, "Deep ITM put delta should be close to -1"

    # Test case 2: Deep OTM put should have small price and delta
    print("\nTest 2: Deep OTM American Put")
    S0, K, r, sigma, T = 100.0, 50.0, 0.05, 0.2, 1.0
    price, delta, gamma = finite_difference_american_greeks(S0, K, r, sigma, T, 'put')
    print(f"  Price: {price:.4f}, Delta: {delta:.4f}, Gamma: {gamma:.6f}")
    assert price < 0.01, "Deep OTM put should have very small price"
    assert abs(delta) < 0.01, "Deep OTM put should have very small delta"

    # Test case 3: ATM put
    print("\nTest 3: ATM American Put")
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
    price, delta, gamma = finite_difference_american_greeks(S0, K, r, sigma, T, 'put')
    print(f"  Price: {price:.4f}, Delta: {delta:.4f}, Gamma: {gamma:.6f}")
    # ATM put delta should be around -0.5 (slightly different for American)
    assert -0.7 < delta < -0.3, "ATM put delta should be around -0.5"

    # Test case 4: American > European for puts (early exercise premium)
    print("\nTest 4: American vs European Put Premium")
    from .bs_analytics import bs_call_price
    S0_t = torch.tensor(S0, device=device)
    # For puts, we need to implement European put pricing
    # Using put-call parity: P = C - S + K*exp(-rT)
    euro_call = bs_call_price(S0_t, torch.tensor(K), torch.tensor(r),
                              torch.tensor(sigma), torch.tensor(T))
    euro_put = euro_call - S0_t + K * torch.exp(-torch.tensor(r) * torch.tensor(T))
    euro_put_price = euro_put.item()

    print(f"  American Put: {price:.4f}")
    print(f"  European Put: {euro_put_price:.4f}")
    print(f"  Early Exercise Premium: {price - euro_put_price:.4f}")
    assert price >= euro_put_price, "American put should be >= European put"

    print("\n✓ All American option tests passed!")


if __name__ == "__main__":
    # Run validation tests
    validate_american_implementation()

    # Generate sample dataset
    print("\n" + "="*50)
    print("Generating sample American option dataset...")

    bs_params = BSParams(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
    american_params = AmericanOptionParams(
        exercise_type='put',
        n_basis=4,
        basis_type='laguerre'
    )

    features, prices, deltas, gammas = generate_american_option_dataset(
        bs_params=bs_params,
        n_samples=100,
        american_params=american_params,
        compute_greeks=True
    )

    print(f"\nDataset generated:")
    print(f"  Features shape: {features.shape}")
    print(f"  Prices shape: {prices.shape}")
    print(f"  Deltas shape: {deltas.shape}")
    print(f"  Gammas shape: {gammas.shape}")
    print(f"\nSample statistics:")
    print(f"  Price: mean={prices.mean():.4f}, std={prices.std():.4f}")
    print(f"  Delta: mean={deltas.mean():.4f}, std={deltas.std():.4f}")
    print(f"  Gamma: mean={gammas.mean():.6f}, std={gammas.std():.6f}")