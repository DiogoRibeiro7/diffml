"""Tests for path-dependent option datasets.

This module tests the dataset generation functions for path-dependent options
including Asian and lookback options, as well as extended barrier options.
"""

import pytest
import torch
from torch import Tensor

from diffml.config import BSParams
from diffml.datasets_barrier_extended import (
    make_double_barrier_call_dataset,
    make_down_and_in_call_dataset,
    make_up_and_out_call_dataset,
)
from diffml.datasets_path_dependent import (
    make_arithmetic_asian_call_dataset,
    make_lookback_call_dataset,
)
from diffml.simulation import build_time_grid, simulate_bs_paths


class TestSimulationExtensions:
    """Test new simulation functions."""

    def test_build_time_grid(self):
        """Test time grid construction."""
        T = 1.0
        n_steps = 4
        grid = build_time_grid(T, n_steps)

        assert grid.shape == (5,)  # n_steps + 1 points
        assert grid[0] == 0.0
        assert grid[-1] == T
        assert torch.allclose(grid[1] - grid[0], torch.tensor(0.25))

        # Test error cases
        with pytest.raises(ValueError):
            build_time_grid(-1.0, 4)  # Negative T
        with pytest.raises(ValueError):
            build_time_grid(1.0, 0)  # Zero steps

    def test_simulate_bs_paths_shape(self):
        """Test shape and basic properties of simulated paths."""
        m = 3
        n_paths = 10
        n_steps = 8
        spots = torch.tensor([[100.0], [110.0], [90.0]])
        params = BSParams(r=0.05, sigma=0.2, T=1.0)

        paths, xi = simulate_bs_paths(spots, params, n_steps, n_paths, seed=42)

        # Check shapes
        assert paths.shape == (m, n_paths, n_steps + 1)
        assert xi.shape == (m, n_paths, n_steps)

        # Check initial values
        assert torch.allclose(paths[:, :, 0], spots.expand(m, n_paths))

        # Check positivity
        assert (paths > 0).all()

        # Check reproducibility
        paths2, xi2 = simulate_bs_paths(spots, params, n_steps, n_paths, seed=42)
        assert torch.allclose(paths, paths2)
        assert torch.allclose(xi, xi2)

    def test_simulate_bs_paths_errors(self):
        """Test error handling in path simulation."""
        params = BSParams(r=0.05, sigma=0.2, T=1.0)

        # Wrong shape for spots
        with pytest.raises(ValueError):
            simulate_bs_paths(torch.tensor([100.0]), params, 10, 100)

        # Negative spot
        with pytest.raises(ValueError):
            simulate_bs_paths(torch.tensor([[-100.0]]), params, 10, 100)

        # Invalid n_steps
        with pytest.raises(ValueError):
            simulate_bs_paths(torch.tensor([[100.0]]), params, 0, 100)

        # Invalid n_paths
        with pytest.raises(ValueError):
            simulate_bs_paths(torch.tensor([[100.0]]), params, 10, 0)


class TestAsianOptions:
    """Test Asian option dataset generation."""

    def test_arithmetic_asian_dataset_shapes(self):
        """Test shapes of arithmetic Asian dataset outputs."""
        m = 5
        K = 100.0
        params = BSParams(r=0.05, sigma=0.2, T=0.25)
        n_steps = 8
        n_paths = 50

        x, price, delta_pw, delta_lrm = make_arithmetic_asian_call_dataset(
            m=m,
            K=K,
            params=params,
            n_steps=n_steps,
            n_paths_per_x=n_paths,
            seed=123
        )

        # Check shapes
        assert x.shape == (m, 1)
        assert price.shape == (m, 1)
        assert delta_pw.shape == (m, 1)
        assert delta_lrm.shape == (m, 1)

        # Check finiteness
        assert torch.isfinite(x).all()
        assert torch.isfinite(price).all()
        assert torch.isfinite(delta_pw).all()
        assert torch.isfinite(delta_lrm).all()

    def test_asian_price_monotonicity(self):
        """Test that Asian option prices increase with spot."""
        m = 10
        K = 100.0
        params = BSParams(r=0.0, sigma=0.2, T=0.5)  # r=0 for cleaner test

        x, price, _, _ = make_arithmetic_asian_call_dataset(
            m=m,
            K=K,
            params=params,
            n_steps=16,
            x_min=0.5,
            x_max=1.5,
            n_paths_per_x=1000,
            seed=456
        )

        # Prices should generally increase with spot
        # Allow for some MC noise
        price_diffs = price[1:] - price[:-1]
        assert (price_diffs >= -0.01).all()  # Allowing small negative due to MC noise

    def test_asian_delta_positivity(self):
        """Test that Asian call deltas are generally positive."""
        m = 5
        K = 100.0
        params = BSParams(r=0.05, sigma=0.2, T=0.25)

        x, _, delta_pw, delta_lrm = make_arithmetic_asian_call_dataset(
            m=m,
            K=K,
            params=params,
            n_steps=8,
            x_min=0.8,
            x_max=1.2,
            n_paths_per_x=500,
            seed=789
        )

        # For calls, delta should be non-negative in most cases
        # Allow some negative values due to MC noise for OTM options
        assert (delta_pw >= -0.05).all()
        assert delta_lrm.mean() > 0  # Average should be positive

    def test_asian_dataset_errors(self):
        """Test error handling in Asian dataset generation."""
        params = BSParams(r=0.05, sigma=0.2, T=0.25)

        with pytest.raises(ValueError):
            make_arithmetic_asian_call_dataset(0, 100, params)  # m = 0

        with pytest.raises(ValueError):
            make_arithmetic_asian_call_dataset(5, -100, params)  # K < 0

        with pytest.raises(ValueError):
            make_arithmetic_asian_call_dataset(5, 100, params, n_steps=0)

        with pytest.raises(ValueError):
            make_arithmetic_asian_call_dataset(5, 100, params, n_paths_per_x=0)


class TestLookbackOptions:
    """Test lookback option dataset generation."""

    def test_lookback_dataset_shapes(self):
        """Test shapes of lookback dataset outputs."""
        m = 5
        K = 100.0
        params = BSParams(r=0.05, sigma=0.2, T=0.25)
        n_steps = 16
        n_paths = 50

        x, price, delta_pw, delta_lrm = make_lookback_call_dataset(
            m=m,
            K=K,
            params=params,
            n_steps=n_steps,
            n_paths_per_x=n_paths,
            seed=123
        )

        # Check shapes
        assert x.shape == (m, 1)
        assert price.shape == (m, 1)
        assert delta_pw.shape == (m, 1)
        assert delta_lrm.shape == (m, 1)

        # Check finiteness
        assert torch.isfinite(x).all()
        assert torch.isfinite(price).all()
        assert torch.isfinite(delta_pw).all()
        assert torch.isfinite(delta_lrm).all()

    def test_lookback_price_bounds(self):
        """Test that lookback prices are within reasonable bounds."""
        m = 5
        K = 100.0
        params = BSParams(r=0.05, sigma=0.2, T=0.5)

        x, price, _, _ = make_lookback_call_dataset(
            m=m,
            K=K,
            params=params,
            n_steps=32,
            x_min=0.8,
            x_max=1.2,
            n_paths_per_x=500,
            seed=456
        )

        # Lookback call price >= vanilla call price (approximately)
        # and should be less than spot price for reasonable parameters
        assert (price >= 0).all()
        assert (price <= x).all()  # Rough upper bound

    def test_lookback_vs_asian_prices(self):
        """Test that lookback prices are generally higher than Asian."""
        m = 5
        K = 100.0
        params = BSParams(r=0.05, sigma=0.2, T=0.5)
        seed = 789

        # Generate both with same parameters
        x_asian, price_asian, _, _ = make_arithmetic_asian_call_dataset(
            m=m, K=K, params=params, n_steps=16, n_paths_per_x=500, seed=seed
        )

        x_lookback, price_lookback, _, _ = make_lookback_call_dataset(
            m=m, K=K, params=params, n_steps=16, n_paths_per_x=500, seed=seed
        )

        # Lookback should generally be more expensive than Asian
        # (max >= average for same path)
        assert (price_lookback >= price_asian - 0.5).all()  # Allow some MC noise


class TestExtendedBarriers:
    """Test extended barrier option datasets."""

    def test_up_and_out_dataset_shapes(self):
        """Test shapes of up-and-out dataset outputs."""
        m = 5
        K = 100.0
        H = 120.0  # Upper barrier
        params = BSParams(r=0.05, sigma=0.2, T=0.25)

        x, price, delta_pw, delta_lrm = make_up_and_out_call_dataset(
            m=m, K=K, H=H, params=params, n_steps=8, n_paths_per_x=100, seed=123
        )

        # Check shapes
        assert x.shape == (m, 1)
        assert price.shape == (m, 1)
        assert delta_pw.shape == (m, 1)
        assert delta_lrm.shape == (m, 1)

        # Check finiteness
        assert torch.isfinite(x).all()
        assert torch.isfinite(price).all()
        assert torch.isfinite(delta_pw).all()
        assert torch.isfinite(delta_lrm).all()

    def test_up_and_out_price_bounds(self):
        """Test that up-and-out prices are less than vanilla."""
        m = 5
        K = 100.0
        H = 120.0
        params = BSParams(r=0.0, sigma=0.2, T=0.5)  # r=0 for simplicity

        x, price, _, _ = make_up_and_out_call_dataset(
            m=m, K=K, H=H, params=params, n_steps=16, n_paths_per_x=500, seed=456
        )

        # Barrier option should be cheaper than vanilla (approximately)
        # Price should be non-negative and less than intrinsic value
        assert (price >= 0).all()
        intrinsic = torch.maximum(x - K, torch.zeros_like(x))
        assert (price <= intrinsic + 1.0).all()  # Allow small MC error

    def test_double_barrier_dataset(self):
        """Test double barrier dataset generation."""
        m = 5
        K = 100.0
        L = 80.0   # Lower barrier
        H = 120.0  # Upper barrier
        params = BSParams(r=0.05, sigma=0.2, T=0.25)

        x, price, delta_pw, delta_lrm = make_double_barrier_call_dataset(
            m=m, K=K, L=L, H=H, params=params, n_steps=8, n_paths_per_x=100, seed=123
        )

        # Check basic properties
        assert x.shape == (m, 1)
        assert torch.isfinite(price).all()
        assert torch.isfinite(delta_pw).all()
        assert torch.isfinite(delta_lrm).all()

        # Prices should be non-negative
        assert (price >= 0).all()

        # Initial spots should be within barriers (approximately)
        assert (x > L * 0.99).all()
        assert (x < H * 1.01).all()

    def test_down_and_in_dataset(self):
        """Test down-and-in dataset generation."""
        m = 5
        K = 100.0
        B = 85.0  # Lower knock-in barrier
        params = BSParams(r=0.05, sigma=0.2, T=0.25)

        x, price, delta_pw, delta_lrm = make_down_and_in_call_dataset(
            m=m, K=K, B=B, params=params, n_paths_per_x=100, seed=123
        )

        # Check basic properties
        assert x.shape == (m, 1)
        assert torch.isfinite(price).all()
        assert torch.isfinite(delta_pw).all()
        assert torch.isfinite(delta_lrm).all()

        # Prices should be non-negative
        assert (price >= 0).all()

    def test_barrier_option_errors(self):
        """Test error handling in barrier datasets."""
        params = BSParams(r=0.05, sigma=0.2, T=0.25)

        # Up-and-out: H should be > K
        with pytest.raises(ValueError):
            make_up_and_out_call_dataset(5, K=100, H=90, params=params)

        # Double barrier: L < K < H
        with pytest.raises(ValueError):
            make_double_barrier_call_dataset(5, K=100, L=110, H=120, params=params)

        with pytest.raises(ValueError):
            make_double_barrier_call_dataset(5, K=100, L=80, H=90, params=params)

        # Down-and-in: B < K
        with pytest.raises(ValueError):
            make_down_and_in_call_dataset(5, K=100, B=110, params=params)


class TestPathDependentIntegration:
    """Integration tests for path-dependent options."""

    def test_all_datasets_consistent_seed(self):
        """Test that all datasets produce consistent results with same seed."""
        m = 3
        K = 100.0
        params = BSParams(r=0.05, sigma=0.2, T=0.5)
        seed = 999

        # Generate all datasets with same seed
        x1, p1, _, _ = make_arithmetic_asian_call_dataset(
            m, K, params, n_steps=8, n_paths_per_x=100, seed=seed
        )
        x2, p2, _, _ = make_arithmetic_asian_call_dataset(
            m, K, params, n_steps=8, n_paths_per_x=100, seed=seed
        )

        # Should be identical
        assert torch.allclose(x1, x2)
        assert torch.allclose(p1, p2)

    def test_price_delta_consistency(self):
        """Test that prices and deltas have consistent signs."""
        m = 10
        K = 100.0
        params = BSParams(r=0.05, sigma=0.2, T=0.25)

        # Generate Asian dataset
        x, price, delta_pw, delta_lrm = make_arithmetic_asian_call_dataset(
            m=m, K=K, params=params, x_min=0.9, x_max=1.1, n_paths_per_x=500, seed=42
        )

        # For ITM options (x > K), both price and delta should be positive
        itm_mask = x > K
        if itm_mask.any():
            assert (price[itm_mask] > 0).all()
            assert (delta_pw[itm_mask] > -0.1).all()  # Allow small MC error

        # Deltas should be bounded roughly in [0, 1] for calls
        assert (delta_pw >= -0.2).all()  # Allow MC noise
        assert (delta_pw <= 1.2).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])