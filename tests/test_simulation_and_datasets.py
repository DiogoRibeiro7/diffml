"""Tests for Monte Carlo simulation and dataset generation functions.

This module tests the correctness of simulation functions and
all dataset generation functions for different option types.
"""


import pytest
import torch

from diffml.config import BSParams, set_default_dtype
from diffml.datasets_barrier import make_barrier_dataset
from diffml.datasets_basket import make_basket_digital_dataset
from diffml.datasets_digital import make_digital_dataset
from diffml.datasets_gamma_portfolio import (
    make_portfolio_gamma_dataset,
)
from diffml.datasets_smoothing import make_smoothed_digital_dataset
from diffml.simulation import (
    simulate_bs_terminal,
    simulate_bs_two_step,
)


@pytest.fixture(autouse=True)
def setup_precision() -> None:
    """Set default dtype to float64 for all tests."""
    set_default_dtype()


@pytest.fixture
def bs_params() -> BSParams:
    """Create standard Black-Scholes parameters for testing."""
    return BSParams(r=0.05, sigma=0.2, T=0.25)


class TestSimulation:
    """Test suite for Monte Carlo simulation functions."""

    def test_simulate_bs_terminal_shapes(self, bs_params: BSParams) -> None:
        """Test that simulate_bs_terminal returns correct shapes."""
        m = 8  # Number of initial spots
        n_paths = 10
        x0 = torch.linspace(90.0, 110.0, m, dtype=torch.float64).reshape(-1, 1)

        ST, xi = simulate_bs_terminal(x0, bs_params, n_paths, seed=42)

        # Check shapes
        assert ST.shape == (m, n_paths)
        assert xi.shape == (m, n_paths)

        # Check types
        assert isinstance(ST, torch.Tensor)
        assert isinstance(xi, torch.Tensor)
        assert ST.dtype == torch.float64
        assert xi.dtype == torch.float64

    def test_simulate_bs_terminal_values(self, bs_params: BSParams) -> None:
        """Test that simulated values are reasonable."""
        x0 = torch.tensor([[100.0]], dtype=torch.float64)
        n_paths = 1000

        ST, xi = simulate_bs_terminal(x0, bs_params, n_paths, seed=42)

        # Terminal values should be positive
        assert torch.all(ST > 0)

        # Mean should be approximately S0 * exp(r*T) for risk-neutral
        expected_mean = x0.item() * torch.exp(
            torch.tensor(bs_params.r * bs_params.T)
        ).item()
        actual_mean = ST.mean().item()
        assert abs(actual_mean - expected_mean) / expected_mean < 0.1  # 10% tolerance

        # Check that xi are standard normal
        assert abs(xi.mean().item()) < 0.2  # Should be close to 0
        assert abs(xi.std().item() - 1.0) < 0.2  # Should be close to 1

    def test_simulate_bs_two_step_shapes(self, bs_params: BSParams) -> None:
        """Test that simulate_bs_two_step returns correct shapes."""
        m = 8
        n_paths = 10
        x0 = torch.linspace(90.0, 110.0, m, dtype=torch.float64).reshape(-1, 1)
        T1 = bs_params.T / 2.0
        T2 = bs_params.T

        S1, S2, xi1, xi2 = simulate_bs_two_step(x0, bs_params, T1, T2, n_paths, seed=42)

        # Check shapes
        assert S1.shape == (m, n_paths)
        assert S2.shape == (m, n_paths)
        assert xi1.shape == (m, n_paths)
        assert xi2.shape == (m, n_paths)

        # Check types
        for tensor in [S1, S2, xi1, xi2]:
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float64

    def test_simulate_bs_two_step_values(self, bs_params: BSParams) -> None:
        """Test that two-step simulation produces reasonable values."""
        x0 = torch.tensor([[100.0]], dtype=torch.float64)
        T1 = bs_params.T / 2.0
        T2 = bs_params.T
        n_paths = 1000

        S1, S2, xi1, xi2 = simulate_bs_two_step(x0, bs_params, T1, T2, n_paths, seed=42)

        # All values should be positive
        assert torch.all(S1 > 0)
        assert torch.all(S2 > 0)

        # S1 should be intermediate between S0 and S2 on average
        assert x0.item() < S1.mean().item() < S2.mean().item()

        # Check normal random variables
        for xi in [xi1, xi2]:
            assert abs(xi.mean().item()) < 0.2
            assert abs(xi.std().item() - 1.0) < 0.2


class TestDigitalDataset:
    """Test suite for digital option dataset generation."""

    def test_make_digital_dataset_shapes(self, bs_params: BSParams) -> None:
        """Test that make_digital_dataset returns correct shapes."""
        m = 16
        K = 100.0
        n_paths = 5

        x, price, delta_pw, delta_lrm = make_digital_dataset(
            m=m,
            K=K,
            params=bs_params,
            x_min=80.0,
            x_max=120.0,
            n_paths_per_x=n_paths,
            seed=42
        )

        # Check shapes
        assert x.shape == (m, 1)
        assert price.shape == (m, 1)
        assert delta_pw.shape == (m, 1)
        assert delta_lrm.shape == (m, 1)

        # Check types
        for tensor in [x, price, delta_pw, delta_lrm]:
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float64

    def test_make_digital_dataset_values(self, bs_params: BSParams) -> None:
        """Test that digital dataset values are reasonable."""
        m = 32
        K = 100.0

        x, price, delta_pw, delta_lrm = make_digital_dataset(
            m=m,
            K=K,
            params=bs_params,
            x_min=80.0,
            x_max=120.0,
            n_paths_per_x=20,
            seed=42
        )

        # Prices should be between 0 and discount factor
        discount = torch.exp(torch.tensor(-bs_params.r * bs_params.T))
        assert torch.all(price >= 0)
        assert torch.all(price <= discount * 1.1)  # Small tolerance

        # Pathwise delta should be zeros (discontinuous payoff)
        assert torch.allclose(delta_pw, torch.zeros_like(delta_pw))

        # LRM delta should be finite and mostly positive
        assert torch.all(torch.isfinite(delta_lrm))
        assert torch.sum(delta_lrm > 0) > m // 2  # Most should be positive

        # No NaNs or Infs
        for tensor in [x, price, delta_pw, delta_lrm]:
            assert torch.all(torch.isfinite(tensor))


class TestBarrierDataset:
    """Test suite for barrier option dataset generation."""

    def test_make_barrier_dataset_shapes(self, bs_params: BSParams) -> None:
        """Test that make_barrier_dataset returns correct shapes."""
        m = 16
        K = 1.0
        B = 0.85
        T1 = bs_params.T / 2.0
        T2 = bs_params.T
        n_paths = 5

        x, price, delta_pw, delta_lrm = make_barrier_dataset(
            m=m,
            K=K,
            B=B,
            params=bs_params,
            T1=T1,
            T2=T2,
            x_min=0.7,
            x_max=1.3,
            n_paths_per_x=n_paths,
            seed=42
        )

        # Check shapes
        assert x.shape == (m, 1)
        assert price.shape == (m, 1)
        assert delta_pw.shape == (m, 1)
        assert delta_lrm.shape == (m, 1)

    def test_make_barrier_dataset_values(self, bs_params: BSParams) -> None:
        """Test that barrier dataset values are reasonable."""
        m = 32
        K = 1.0
        B = 0.85
        T1 = bs_params.T / 2.0
        T2 = bs_params.T

        x, price, delta_pw, delta_lrm = make_barrier_dataset(
            m=m,
            K=K,
            B=B,
            params=bs_params,
            T1=T1,
            T2=T2,
            x_min=0.7,
            x_max=1.3,
            n_paths_per_x=20,
            seed=42
        )

        # Prices should be non-negative and less than spot
        assert torch.all(price >= 0)
        assert torch.all(price <= x * 1.1)  # Rough upper bound

        # Deltas should be finite
        assert torch.all(torch.isfinite(delta_pw))
        assert torch.all(torch.isfinite(delta_lrm))

        # Barrier option price should be less than vanilla for same strike
        # (down-and-out has knock-out feature)
        assert torch.all(price < x)  # Very rough check


class TestBasketDataset:
    """Test suite for basket option dataset generation."""

    def test_make_basket_dataset_shapes(self) -> None:
        """Test that make_basket_digital_dataset returns correct shapes."""
        m = 16
        d = 5  # Small dimension for testing
        K = 1.0
        sigma = 0.2
        T = 0.25
        n_paths = 5

        x, price, delta_pw, delta_lrm = make_basket_digital_dataset(
            m=m,
            d=d,
            K=K,
            sigma=sigma,
            T=T,
            x_min=0.5,
            x_max=1.5,
            n_paths_per_x=n_paths,
            seed=42
        )

        # Check shapes
        assert x.shape == (m, d)  # Multi-dimensional input
        assert price.shape == (m, 1)
        assert delta_pw.shape == (m, d)  # Vector delta
        assert delta_lrm.shape == (m, d)  # Vector delta

    def test_make_basket_dataset_values(self) -> None:
        """Test that basket dataset values are reasonable."""
        m = 32
        d = 10
        K = 1.0
        sigma = 0.2
        T = 0.25

        x, price, delta_pw, delta_lrm = make_basket_digital_dataset(
            m=m,
            d=d,
            K=K,
            sigma=sigma,
            T=T,
            x_min=0.5,
            x_max=1.5,
            n_paths_per_x=20,
            seed=42
        )

        # Prices should be between 0 and 1 (digital option)
        assert torch.all(price >= 0)
        assert torch.all(price <= 1.0)

        # Pathwise delta should be zeros (discontinuous)
        assert torch.allclose(delta_pw, torch.zeros_like(delta_pw))

        # LRM delta should be finite
        assert torch.all(torch.isfinite(delta_lrm))

        # Average spot should correlate with price (roughly)
        avg_spot = x.mean(dim=1, keepdim=True)
        # Higher average spot -> higher digital price
        correlation = torch.corrcoef(
            torch.cat([avg_spot.squeeze(), price.squeeze()])
        )[0, 1]
        assert correlation > 0.5  # Should be positively correlated


class TestSmoothingDataset:
    """Test suite for smoothed digital option dataset generation."""

    def test_make_smoothed_dataset_shapes(self, bs_params: BSParams) -> None:
        """Test that make_smoothed_digital_dataset returns correct shapes."""
        m = 16
        K = 100.0
        n_paths = 5
        eps_multiplier = 1.0

        x, price, delta_pw, delta_lrm = make_smoothed_digital_dataset(
            m=m,
            K=K,
            params=bs_params,
            x_min=80.0,
            x_max=120.0,
            n_paths_per_x=n_paths,
            eps_multiplier=eps_multiplier,
            seed=42
        )

        # Check shapes
        assert x.shape == (m, 1)
        assert price.shape == (m, 1)
        assert delta_pw.shape == (m, 1)
        assert delta_lrm.shape == (m, 1)

    def test_make_smoothed_dataset_values(self, bs_params: BSParams) -> None:
        """Test that smoothed dataset values are reasonable."""
        m = 32
        K = 100.0

        # Test with different smoothing levels
        for eps_mult in [0.5, 1.0, 2.0]:
            x, price, delta_pw, delta_lrm = make_smoothed_digital_dataset(
                m=m,
                K=K,
                params=bs_params,
                x_min=80.0,
                x_max=120.0,
                n_paths_per_x=20,
                eps_multiplier=eps_mult,
                seed=42
            )

            # Prices should be between 0 and discount
            discount = torch.exp(torch.tensor(-bs_params.r * bs_params.T))
            assert torch.all(price >= 0)
            assert torch.all(price <= discount * 1.1)

            # Smoothed payoff should have non-zero pathwise deltas
            # (unlike pure digital)
            assert not torch.allclose(delta_pw, torch.zeros_like(delta_pw))
            assert torch.any(delta_pw > 0)

            # All values should be finite
            for tensor in [x, price, delta_pw, delta_lrm]:
                assert torch.all(torch.isfinite(tensor))


class TestGammaPortfolioDataset:
    """Test suite for gamma portfolio dataset generation."""

    def test_make_portfolio_gamma_dataset_shapes(self, bs_params: BSParams) -> None:
        """Test that make_portfolio_gamma_dataset returns correct shapes."""
        m = 16
        strikes = torch.tensor([0.9, 1.0, 1.1], dtype=torch.float64)
        weights = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
        n_paths = 5

        results = make_portfolio_gamma_dataset(
            m=m,
            params=bs_params,
            strikes=strikes,
            weights=weights,
            x_min=0.5,
            x_max=1.5,
            n_paths_per_x=n_paths,
            seed=42
        )

        x, price_true, delta_true, gamma_true, price_mc, delta_pw, gamma_pwlr = results

        # Check that we get 7 outputs
        assert len(results) == 7

        # Check shapes - all should be (m, 1)
        for tensor in results:
            assert tensor.shape == (m, 1)
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float64

    def test_make_portfolio_gamma_dataset_values(self, bs_params: BSParams) -> None:
        """Test that portfolio gamma dataset values are reasonable."""
        m = 32
        # Butterfly spread
        strikes = torch.tensor([0.85, 0.9, 1.15], dtype=torch.float64)
        weights = torch.tensor([1.0, -1.5, 0.75], dtype=torch.float64)

        results = make_portfolio_gamma_dataset(
            m=m,
            params=bs_params,
            strikes=strikes,
            weights=weights,
            x_min=0.5,
            x_max=1.5,
            n_paths_per_x=50,
            seed=42
        )

        x, price_true, delta_true, gamma_true, price_mc, delta_pw, gamma_pwlr = results

        # Analytical and MC prices should be close
        price_diff = (price_true - price_mc).abs()
        assert price_diff.mean() < 0.1  # Reasonable tolerance

        # All values should be finite
        for tensor in results:
            assert torch.all(torch.isfinite(tensor))

        # Butterfly spread has positive gamma near center strike
        # Find spots near center strike (0.9)
        near_center = (x - 0.9).abs() < 0.1
        if near_center.any():
            gamma_near_center = gamma_true[near_center]
            assert torch.any(gamma_near_center > 0)

        # Portfolio value should be non-negative for butterfly
        assert torch.all(price_true >= -0.01)  # Small tolerance for numerical errors


class TestDatasetRobustness:
    """Test robustness of dataset generation functions."""

    def test_all_datasets_with_small_sizes(self, bs_params: BSParams) -> None:
        """Test that all datasets work with very small sizes."""
        m = 2  # Minimal size
        n_paths = 2

        # Digital
        x1, p1, d1, d2 = make_digital_dataset(
            m=m, K=100.0, params=bs_params,
            x_min=90.0, x_max=110.0, n_paths_per_x=n_paths, seed=1
        )
        assert x1.shape[0] == m

        # Barrier
        x2, p2, d3, d4 = make_barrier_dataset(
            m=m, K=1.0, B=0.85, params=bs_params,
            T1=bs_params.T/2, T2=bs_params.T,
            x_min=0.7, x_max=1.3, n_paths_per_x=n_paths, seed=2
        )
        assert x2.shape[0] == m

        # Basket
        x3, p3, d5, d6 = make_basket_digital_dataset(
            m=m, d=3, K=1.0, sigma=0.2, T=0.25,
            x_min=0.5, x_max=1.5, n_paths_per_x=n_paths, seed=3
        )
        assert x3.shape[0] == m

        # Smoothed
        x4, p4, d7, d8 = make_smoothed_digital_dataset(
            m=m, K=100.0, params=bs_params,
            x_min=90.0, x_max=110.0, n_paths_per_x=n_paths,
            eps_multiplier=1.0, seed=4
        )
        assert x4.shape[0] == m

        # Portfolio gamma
        strikes = torch.tensor([0.9, 1.0, 1.1], dtype=torch.float64)
        weights = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)
        results = make_portfolio_gamma_dataset(
            m=m, params=bs_params, strikes=strikes, weights=weights,
            x_min=0.5, x_max=1.5, n_paths_per_x=n_paths, seed=5
        )
        assert results[0].shape[0] == m

    def test_datasets_deterministic_with_seed(self, bs_params: BSParams) -> None:
        """Test that datasets are deterministic when using same seed."""
        m = 8
        n_paths = 10
        seed = 12345

        # Generate twice with same seed
        x1, p1, _, _ = make_digital_dataset(
            m=m, K=100.0, params=bs_params,
            x_min=90.0, x_max=110.0, n_paths_per_x=n_paths, seed=seed
        )

        x2, p2, _, _ = make_digital_dataset(
            m=m, K=100.0, params=bs_params,
            x_min=90.0, x_max=110.0, n_paths_per_x=n_paths, seed=seed
        )

        # Should be identical
        assert torch.allclose(x1, x2)
        assert torch.allclose(p1, p2)

        # Generate with different seed
        x3, p3, _, _ = make_digital_dataset(
            m=m, K=100.0, params=bs_params,
            x_min=90.0, x_max=110.0, n_paths_per_x=n_paths, seed=seed+1
        )

        # Should be different (prices will differ due to MC randomness)
        assert not torch.allclose(p1, p3, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
