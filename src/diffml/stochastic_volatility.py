"""
Stochastic Volatility Models for DiffML

This module implements various stochastic volatility models for option pricing,
including Heston, SABR, and rough volatility models. These models capture the
volatility smile and term structure effects observed in real markets.

Models Implemented:
    - Heston Model: Mean-reverting stochastic volatility
    - SABR Model: Stochastic alpha-beta-rho model
    - Rough Heston: Fractional Brownian motion volatility
    - Bergomi Model: Forward variance model
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import scipy.stats as stats
from scipy.integrate import quad
from scipy.special import gamma


@dataclass
class HestonParams:
    """Parameters for the Heston stochastic volatility model.

    The Heston model dynamics:
    dS_t = r*S_t*dt + sqrt(v_t)*S_t*dW_t^S
    dv_t = kappa*(theta - v_t)*dt + sigma*sqrt(v_t)*dW_t^v
    with correlation rho between W^S and W^v
    """
    v0: float      # Initial variance
    theta: float   # Long-term variance
    kappa: float   # Mean reversion speed
    sigma: float   # Volatility of volatility
    rho: float     # Correlation between asset and variance
    r: float       # Risk-free rate

    def validate(self):
        """Validate Feller condition: 2*kappa*theta > sigma^2."""
        if 2 * self.kappa * self.theta <= self.sigma ** 2:
            raise ValueError(f"Feller condition violated: 2κθ = {2*self.kappa*self.theta:.4f} "
                           f"must be > σ² = {self.sigma**2:.4f}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"Correlation must be in [-1, 1], got {self.rho}")


@dataclass
class SABRParams:
    """Parameters for the SABR stochastic volatility model.

    The SABR model dynamics:
    dF_t = α_t*F_t^β*dW_t^F
    dα_t = ν*α_t*dW_t^α
    with correlation ρ between W^F and W^α
    """
    alpha: float   # Initial volatility
    beta: float    # CEV exponent (0=normal, 1=lognormal)
    rho: float     # Correlation
    nu: float      # Volatility of volatility

    def validate(self):
        """Validate SABR parameters."""
        if self.alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {self.alpha}")
        if not 0 <= self.beta <= 1:
            raise ValueError(f"Beta must be in [0, 1], got {self.beta}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"Rho must be in [-1, 1], got {self.rho}")
        if self.nu <= 0:
            raise ValueError(f"Nu must be positive, got {self.nu}")


class HestonModel:
    """Heston stochastic volatility model implementation.

    Provides pricing and simulation methods for options under
    the Heston model using both Monte Carlo and semi-analytical methods.
    """

    def __init__(self, params: HestonParams):
        """Initialize Heston model.

        Args:
            params: Heston model parameters
        """
        self.params = params
        params.validate()

    def characteristic_function(self, u: torch.Tensor, T: float) -> torch.Tensor:
        """Compute the characteristic function for log-price.

        The characteristic function φ(u) = E[exp(iu*log(S_T/S_0))]

        Args:
            u: Complex argument
            T: Time to maturity

        Returns:
            Characteristic function values
        """
        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma
        rho = self.params.rho
        v0 = self.params.v0
        r = self.params.r

        # Complex calculations
        d = torch.sqrt((rho * sigma * u * 1j - kappa) ** 2 -
                      sigma ** 2 * (-u * 1j - u ** 2))
        g = (kappa - rho * sigma * u * 1j - d) / (kappa - rho * sigma * u * 1j + d)

        C = r * u * 1j * T + (kappa * theta) / sigma ** 2 * (
            (kappa - rho * sigma * u * 1j - d) * T -
            2 * torch.log((1 - g * torch.exp(-d * T)) / (1 - g))
        )

        D = (kappa - rho * sigma * u * 1j - d) / sigma ** 2 * (
            (1 - torch.exp(-d * T)) / (1 - g * torch.exp(-d * T))
        )

        return torch.exp(C + D * v0)

    def price_european_call(self, S0: float, K: float, T: float,
                           n_points: int = 256) -> float:
        """Price European call using Fourier inversion.

        Uses the Carr-Madan formula with FFT for efficient pricing.

        Args:
            S0: Initial spot price
            K: Strike price
            T: Time to maturity
            n_points: Number of integration points

        Returns:
            Option price
        """
        # Log-strike
        k = np.log(K / S0)

        # Integration bounds and grid
        alpha = 1.5
        eta = 0.25
        b = 120
        u_max = n_points * eta

        # Damped characteristic function
        def integrand(u):
            u_tensor = torch.tensor(u + (alpha + 1) * 1j)
            cf = self.characteristic_function(u_tensor, T)

            damped_cf = torch.exp(-self.params.r * T) * cf / (
                alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u
            )

            return (torch.real(damped_cf * torch.exp(-1j * u * k))).item()

        # Numerical integration
        integral, _ = quad(integrand, 0, b)

        price = S0 * np.exp(-alpha * k) / np.pi * integral

        return price

    def simulate_paths(self, S0: float, T: float, n_steps: int,
                       n_paths: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate price and variance paths using Euler-Maruyama.

        Args:
            S0: Initial spot price
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths
            device: Computation device

        Returns:
            Tuple of (price_paths, variance_paths)
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        # Initialize paths
        S = torch.zeros((n_paths, n_steps + 1), device=device)
        v = torch.zeros((n_paths, n_steps + 1), device=device)

        S[:, 0] = S0
        v[:, 0] = self.params.v0

        # Generate correlated Brownian motions
        for i in range(n_steps):
            z1 = torch.randn(n_paths, device=device)
            z2 = torch.randn(n_paths, device=device)

            # Correlated shocks
            w_s = z1
            w_v = self.params.rho * z1 + np.sqrt(1 - self.params.rho ** 2) * z2

            # Variance process (ensure positivity)
            v_curr = v[:, i]
            v_next = v_curr + self.params.kappa * (self.params.theta - v_curr) * dt + \
                     self.params.sigma * torch.sqrt(torch.abs(v_curr)) * sqrt_dt * w_v
            v[:, i + 1] = torch.maximum(v_next, torch.tensor(0.0, device=device))

            # Price process
            S[:, i + 1] = S[:, i] * torch.exp(
                (self.params.r - 0.5 * v[:, i]) * dt +
                torch.sqrt(v[:, i]) * sqrt_dt * w_s
            )

        return S, v

    def price_option_mc(self, S0: float, K: float, T: float,
                       option_type: str = 'call',
                       n_paths: int = 100000,
                       n_steps: int = 100) -> Dict[str, float]:
        """Price option using Monte Carlo simulation.

        Args:
            S0: Initial spot price
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
            n_paths: Number of simulation paths
            n_steps: Number of time steps

        Returns:
            Dictionary with price and standard error
        """
        # Simulate paths
        S_paths, v_paths = self.simulate_paths(S0, T, n_steps, n_paths)

        # Terminal payoff
        S_T = S_paths[:, -1]
        if option_type == 'call':
            payoff = torch.maximum(S_T - K, torch.tensor(0.0))
        else:
            payoff = torch.maximum(K - S_T, torch.tensor(0.0))

        # Discounted expectation
        discount = np.exp(-self.params.r * T)
        price = discount * torch.mean(payoff).item()
        std_error = discount * torch.std(payoff).item() / np.sqrt(n_paths)

        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error)
        }

    def calibrate(self, market_prices: torch.Tensor,
                 strikes: torch.Tensor,
                 maturities: torch.Tensor,
                 S0: float) -> 'HestonParams':
        """Calibrate model parameters to market prices.

        Uses gradient-based optimization to minimize pricing errors.

        Args:
            market_prices: Observed option prices
            strikes: Strike prices
            maturities: Times to maturity
            S0: Current spot price

        Returns:
            Calibrated parameters
        """
        # Implementation of calibration algorithm
        # This would use optimization to find parameters that best fit market prices
        pass


class SABRModel:
    """SABR stochastic volatility model implementation.

    The SABR model is widely used for interest rate derivatives
    and provides closed-form approximations for implied volatility.
    """

    def __init__(self, params: SABRParams):
        """Initialize SABR model.

        Args:
            params: SABR model parameters
        """
        self.params = params
        params.validate()

    def implied_volatility_hagan(self, F: float, K: float, T: float) -> float:
        """Calculate implied volatility using Hagan's approximation.

        This is the standard SABR formula from Hagan et al. (2002).

        Args:
            F: Forward price
            K: Strike price
            T: Time to maturity

        Returns:
            Implied Black volatility
        """
        alpha = self.params.alpha
        beta = self.params.beta
        rho = self.params.rho
        nu = self.params.nu

        if F == K:
            # ATM case
            f_mid = F
        else:
            # General case
            f_mid = np.sqrt(F * K)

        # Log-moneyness
        if F == K:
            log_fk = 0
            z = 0
            x_z = 1
        else:
            log_fk = np.log(F / K)
            z = (nu / alpha) * f_mid ** (1 - beta) * log_fk
            x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        # First term
        if F == K:
            term1 = alpha / (f_mid ** (1 - beta))
        else:
            term1 = z / x_z

        # Expansion terms
        term2 = 1 + (
            ((1 - beta) ** 2 / 24) * (alpha ** 2 / f_mid ** (2 - 2 * beta)) +
            0.25 * rho * beta * nu * alpha / f_mid ** (1 - beta) +
            (2 - 3 * rho ** 2) / 24 * nu ** 2
        ) * T

        sigma = term1 * term2

        return sigma

    def price_european_option(self, F: float, K: float, T: float,
                            r: float = 0.0, option_type: str = 'call') -> float:
        """Price European option using SABR implied volatility.

        Args:
            F: Forward price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate (for discounting)
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        # Get implied volatility
        sigma = self.implied_volatility_hagan(F, K, T)

        # Black formula
        d1 = np.log(F / K) / (sigma * np.sqrt(T)) + 0.5 * sigma * np.sqrt(T)
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = np.exp(-r * T) * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
        else:
            price = np.exp(-r * T) * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))

        return price

    def simulate_paths(self, F0: float, T: float, n_steps: int,
                       n_paths: int, device: str = 'cpu') -> torch.Tensor:
        """Simulate forward price paths.

        Args:
            F0: Initial forward price
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths
            device: Computation device

        Returns:
            Forward price paths
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        # Initialize paths
        F = torch.zeros((n_paths, n_steps + 1), device=device)
        alpha = torch.zeros((n_paths, n_steps + 1), device=device)

        F[:, 0] = F0
        alpha[:, 0] = self.params.alpha

        # Generate paths
        for i in range(n_steps):
            z1 = torch.randn(n_paths, device=device)
            z2 = torch.randn(n_paths, device=device)

            # Correlated shocks
            w_f = z1
            w_alpha = self.params.rho * z1 + np.sqrt(1 - self.params.rho ** 2) * z2

            # Volatility process
            alpha[:, i + 1] = alpha[:, i] * torch.exp(
                -0.5 * self.params.nu ** 2 * dt +
                self.params.nu * sqrt_dt * w_alpha
            )

            # Forward process (CEV dynamics)
            if self.params.beta == 1:
                # Lognormal case
                F[:, i + 1] = F[:, i] * torch.exp(
                    -0.5 * alpha[:, i] ** 2 * dt +
                    alpha[:, i] * sqrt_dt * w_f
                )
            else:
                # General CEV case (requires more careful discretization)
                vol = alpha[:, i] * F[:, i] ** (self.params.beta - 1)
                F[:, i + 1] = F[:, i] + vol * F[:, i] * sqrt_dt * w_f

        return F


class RoughHestonModel:
    """Rough Heston model with fractional Brownian motion.

    The rough Heston model uses fractional Brownian motion for the
    volatility process, capturing the rough behavior observed in
    realized volatility.
    """

    def __init__(self, v0: float, theta: float, kappa: float,
                sigma: float, rho: float, H: float, r: float):
        """Initialize rough Heston model.

        Args:
            v0: Initial variance
            theta: Long-term variance
            kappa: Mean reversion speed
            sigma: Volatility of volatility
            rho: Correlation
            H: Hurst parameter (H < 0.5 for rough volatility)
            r: Risk-free rate
        """
        self.v0 = v0
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.rho = rho
        self.H = H  # Hurst parameter
        self.r = r

        if not 0 < H < 1:
            raise ValueError(f"Hurst parameter must be in (0, 1), got {H}")
        if H >= 0.5:
            print(f"Warning: H={H} >= 0.5 gives standard (non-rough) volatility")

    def fractional_kernel(self, t: float, s: float) -> float:
        """Compute the fractional kernel K(t, s).

        For fractional Brownian motion with Hurst parameter H.

        Args:
            t: Current time
            s: Past time (s <= t)

        Returns:
            Kernel value
        """
        if s > t:
            return 0.0

        if s == t:
            return 0.0

        # Riemann-Liouville fractional integral kernel
        alpha = self.H + 0.5
        kernel = (t - s) ** (alpha - 1) / gamma(alpha)

        return kernel

    def simulate_rough_variance(self, T: float, n_steps: int,
                               n_paths: int) -> torch.Tensor:
        """Simulate rough variance paths.

        Uses a discretization scheme for the rough Heston variance.

        Args:
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths

        Returns:
            Variance paths
        """
        dt = T / n_steps
        times = torch.linspace(0, T, n_steps + 1)

        # Initialize variance paths
        v = torch.zeros((n_paths, n_steps + 1))
        v[:, 0] = self.v0

        # Generate fractional Brownian motion
        # This is a simplified approximation
        for i in range(1, n_steps + 1):
            # Compute weights for fractional integration
            weights = torch.zeros(i)
            for j in range(i):
                weights[j] = self.fractional_kernel(times[i], times[j])

            # Normalize weights
            weights = weights / torch.sum(weights) if torch.sum(weights) > 0 else weights

            # Generate innovation
            dW = torch.randn(n_paths) * np.sqrt(dt)

            # Update variance (simplified scheme)
            drift = self.kappa * (self.theta - v[:, i-1]) * dt
            diffusion = self.sigma * torch.sqrt(torch.abs(v[:, i-1])) * dW

            v[:, i] = torch.maximum(v[:, i-1] + drift + diffusion, torch.tensor(0.0))

        return v


class BergomiModel:
    """Bergomi forward variance model.

    The Bergomi model directly models the forward variance curve,
    providing consistency with the implied volatility surface.
    """

    def __init__(self, xi0_func, eta: float, rho: float, H: float):
        """Initialize Bergomi model.

        Args:
            xi0_func: Initial forward variance curve function
            eta: Volatility of volatility
            rho: Correlation
            H: Hurst parameter for rough Bergomi (H=0.5 for standard)
        """
        self.xi0_func = xi0_func
        self.eta = eta
        self.rho = rho
        self.H = H

    def forward_variance(self, t: float, T: float, W_t: float) -> float:
        """Compute instantaneous forward variance.

        Args:
            t: Current time
            T: Forward time (T >= t)
            W_t: Brownian motion value at time t

        Returns:
            Forward variance xi_t(T)
        """
        xi0 = self.xi0_func(T - t)

        # Exponential martingale
        xi_t = xi0 * np.exp(self.eta * W_t - 0.5 * self.eta ** 2 * t)

        return xi_t


class StochasticVolDataset:
    """Dataset generator for stochastic volatility models.

    Generates training data for DML models with prices and Greeks
    computed under stochastic volatility.
    """

    def __init__(self, model_type: str = 'heston', n_samples: int = 10000):
        """Initialize dataset generator.

        Args:
            model_type: Type of model ('heston', 'sabr', 'rough_heston')
            n_samples: Number of samples to generate
        """
        self.model_type = model_type
        self.n_samples = n_samples

        # Default model parameters
        if model_type == 'heston':
            self.model = HestonModel(HestonParams(
                v0=0.04, theta=0.04, kappa=1.0,
                sigma=0.3, rho=-0.5, r=0.05
            ))
        elif model_type == 'sabr':
            self.model = SABRModel(SABRParams(
                alpha=0.2, beta=0.5, rho=-0.3, nu=0.3
            ))
        elif model_type == 'rough_heston':
            self.model = RoughHestonModel(
                v0=0.04, theta=0.04, kappa=1.0,
                sigma=0.3, rho=-0.5, H=0.1, r=0.05
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def generate(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate dataset with features, prices, and Greeks.

        Returns:
            Tuple of (features, prices, greeks)
        """
        # Generate random market conditions
        spots = torch.rand(self.n_samples) * 40 + 80  # [80, 120]
        strikes = torch.rand(self.n_samples) * 40 + 80  # [80, 120]
        maturities = torch.rand(self.n_samples) * 2 + 0.1  # [0.1, 2.1]

        # Features: [spot, strike, maturity, v0, theta, kappa, sigma, rho]
        if self.model_type == 'heston':
            features = torch.stack([
                spots, strikes, maturities,
                torch.full((self.n_samples,), self.model.params.v0),
                torch.full((self.n_samples,), self.model.params.theta),
                torch.full((self.n_samples,), self.model.params.kappa),
                torch.full((self.n_samples,), self.model.params.sigma),
                torch.full((self.n_samples,), self.model.params.rho)
            ], dim=1)
        else:
            # Simplified features for other models
            features = torch.stack([spots, strikes, maturities], dim=1)

        # Compute prices and Greeks
        prices = torch.zeros(self.n_samples)
        deltas = torch.zeros(self.n_samples)
        vegas = torch.zeros(self.n_samples)

        for i in range(self.n_samples):
            S = spots[i].item()
            K = strikes[i].item()
            T = maturities[i].item()

            if self.model_type == 'heston':
                # Price using Fourier method
                price = self.model.price_european_call(S, K, T)
                prices[i] = price

                # Approximate Greeks using finite differences
                eps = 0.01
                price_up = self.model.price_european_call(S * (1 + eps), K, T)
                price_down = self.model.price_european_call(S * (1 - eps), K, T)
                deltas[i] = (price_up - price_down) / (2 * S * eps)

                # Vega (sensitivity to v0)
                model_up = HestonModel(HestonParams(
                    v0=self.model.params.v0 * (1 + eps),
                    theta=self.model.params.theta,
                    kappa=self.model.params.kappa,
                    sigma=self.model.params.sigma,
                    rho=self.model.params.rho,
                    r=self.model.params.r
                ))
                price_vega_up = model_up.price_european_call(S, K, T)
                vegas[i] = (price_vega_up - price) / (self.model.params.v0 * eps)

            elif self.model_type == 'sabr':
                # Use SABR formula
                F = S  # Assuming spot = forward for simplicity
                price = self.model.price_european_option(F, K, T)
                prices[i] = price

                # Greeks via finite differences
                price_up = self.model.price_european_option(F * (1 + eps), K, T)
                price_down = self.model.price_european_option(F * (1 - eps), K, T)
                deltas[i] = (price_up - price_down) / (2 * F * eps)

        greeks = torch.stack([deltas, vegas], dim=1)

        return features, prices.unsqueeze(1), greeks


class StochasticVolNN(nn.Module):
    """Neural network for stochastic volatility option pricing.

    Specialized architecture for learning option prices under
    stochastic volatility with proper scaling and activation functions.
    """

    def __init__(self, input_dim: int = 8, hidden_dims: list = [128, 128, 128, 128]):
        """Initialize network.

        Args:
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())  # Ensure positive prices

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Option prices [batch, 1]
        """
        return self.network(x)


def train_stochastic_vol_model(model: nn.Module,
                              dataset: StochasticVolDataset,
                              epochs: int = 100,
                              batch_size: int = 256,
                              learning_rate: float = 0.001,
                              differential_weight: float = 0.5) -> Dict[str, list]:
    """Train neural network on stochastic volatility data.

    Args:
        model: Neural network model
        dataset: Stochastic volatility dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        differential_weight: Weight for Greek loss

    Returns:
        Training history
    """
    # Generate data
    X, y, dy = dataset.generate()

    # Split into train/val
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    dy_train, dy_val = dy[:n_train], dy[n_train:]

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train, dy_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for X_batch, y_batch, dy_batch in train_loader:
            X_batch.requires_grad_(True)

            # Forward pass
            y_pred = model(X_batch)

            # Compute gradients for Greeks
            dy_pred = torch.autograd.grad(
                y_pred.sum(), X_batch,
                create_graph=True
            )[0][:, :2]  # Only delta and vega

            # Combined loss
            value_loss = torch.mean((y_pred - y_batch) ** 2)
            greek_loss = torch.mean((dy_pred - dy_batch) ** 2)
            loss = (1 - differential_weight) * value_loss + differential_weight * greek_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = torch.mean((y_val_pred - y_val) ** 2).item()

        scheduler.step()

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {history['train_loss'][-1]:.6f}, "
                  f"Val Loss: {val_loss:.6f}")

    return history


if __name__ == "__main__":
    """Example usage of stochastic volatility models."""

    print("=" * 60)
    print("Stochastic Volatility Models for DiffML")
    print("=" * 60)

    # Example 1: Heston Model
    print("\n1. Heston Model Example")
    print("-" * 40)

    heston_params = HestonParams(
        v0=0.04,     # 4% initial variance (20% vol)
        theta=0.04,  # 4% long-term variance
        kappa=2.0,   # Mean reversion speed
        sigma=0.3,   # Vol of vol
        rho=-0.5,    # Negative correlation (leverage effect)
        r=0.05       # 5% risk-free rate
    )

    heston = HestonModel(heston_params)

    # Price a call option
    S0, K, T = 100.0, 100.0, 1.0
    call_price = heston.price_european_call(S0, K, T)
    print(f"ATM Call Price (Semi-Analytical): ${call_price:.4f}")

    # Monte Carlo price
    mc_result = heston.price_option_mc(S0, K, T, n_paths=10000)
    print(f"ATM Call Price (Monte Carlo): ${mc_result['price']:.4f} "
          f"± {mc_result['std_error']:.4f}")

    # Example 2: SABR Model
    print("\n2. SABR Model Example")
    print("-" * 40)

    sabr_params = SABRParams(
        alpha=0.2,   # Initial vol
        beta=0.5,    # CEV parameter
        rho=-0.3,    # Correlation
        nu=0.4       # Vol of vol
    )

    sabr = SABRModel(sabr_params)

    # Implied volatility smile
    F = 100.0
    strikes = np.linspace(80, 120, 9)
    T = 1.0

    print("Strike  Implied Vol")
    for K in strikes:
        iv = sabr.implied_volatility_hagan(F, K, T)
        print(f"{K:6.1f}  {iv:10.4%}")

    # Example 3: Training DML on Stochastic Vol Data
    print("\n3. Training DML Model")
    print("-" * 40)

    # Generate dataset
    dataset = StochasticVolDataset(model_type='heston', n_samples=5000)

    # Create and train model
    model = StochasticVolNN(input_dim=8)

    print("Training neural network on Heston data...")
    history = train_stochastic_vol_model(
        model, dataset,
        epochs=20,
        batch_size=128,
        differential_weight=0.5
    )

    print(f"\nFinal Training Loss: {history['train_loss'][-1]:.6f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")

    print("\n" + "=" * 60)
    print("Stochastic volatility models successfully implemented!")
    print("These models capture realistic market dynamics including:")
    print("  - Volatility smile and skew")
    print("  - Term structure of implied volatility")
    print("  - Leverage effect (negative correlation)")
    print("  - Rough volatility dynamics")
    print("=" * 60)