"""Tests for Black-Scholes analytical formulas.

This module tests the correctness of Black-Scholes pricing functions
for digital and vanilla options.
"""

from typing import Tuple

import pytest
import torch

from diffml.bs_analytics import (
    bs_call_gamma,
    bs_call_price,
    bs_digital_delta,
    bs_digital_price,
)
from diffml.config import BSParams, set_default_dtype


@pytest.fixture(autouse=True)
def setup_precision() -> None:
    """Set default dtype to float64 for all tests."""
    set_default_dtype()


@pytest.fixture
def bs_params() -> BSParams:
    """Create standard Black-Scholes parameters for testing."""
    return BSParams(r=0.05, sigma=0.2, T=0.25)


@pytest.fixture
def spot_prices() -> torch.Tensor:
    """Create a range of spot prices for testing."""
    return torch.linspace(80.0, 120.0, 10, dtype=torch.float64).reshape(-1, 1)


class TestDigitalOptions:
    """Test suite for digital option pricing functions."""

    def test_digital_price_shape(
        self, spot_prices: torch.Tensor, bs_params: BSParams
    ) -> None:
        """Test that bs_digital_price returns correct shape."""
        K = 100.0
        price = bs_digital_price(spot_prices, K, bs_params)

        assert isinstance(price, torch.Tensor)
        assert price.shape == spot_prices.shape
        assert price.dtype == torch.float64

    def test_digital_delta_shape(
        self, spot_prices: torch.Tensor, bs_params: BSParams
    ) -> None:
        """Test that bs_digital_delta returns correct shape."""
        K = 100.0
        delta = bs_digital_delta(spot_prices, K, bs_params)

        assert isinstance(delta, torch.Tensor)
        assert delta.shape == spot_prices.shape
        assert delta.dtype == torch.float64

    def test_digital_price_limits(self, bs_params: BSParams) -> None:
        """Test digital price behavior at extreme spot values.

        For S >> K: digital price should be close to discount factor e^(-rT)
        For S << K: digital price should be close to 0
        """
        K = 100.0
        discount = torch.exp(torch.tensor(-bs_params.r * bs_params.T))

        # Test S >> K (deep in the money)
        s_high = torch.tensor([[1000.0]], dtype=torch.float64)
        price_high = bs_digital_price(s_high, K, bs_params)
        assert torch.allclose(price_high, discount, rtol=1e-2)

        # Test S << K (deep out of the money)
        s_low = torch.tensor([[10.0]], dtype=torch.float64)
        price_low = bs_digital_price(s_low, K, bs_params)
        assert price_low.item() < 0.01  # Should be close to 0

    def test_digital_price_range(
        self, spot_prices: torch.Tensor, bs_params: BSParams
    ) -> None:
        """Test that digital prices are in valid range [0, discount]."""
        K = 100.0
        price = bs_digital_price(spot_prices, K, bs_params)
        discount = torch.exp(torch.tensor(-bs_params.r * bs_params.T))

        assert torch.all(price >= 0)
        assert torch.all(price <= discount * 1.01)  # Small tolerance

    def test_digital_delta_sign_and_magnitude(
        self, spot_prices: torch.Tensor, bs_params: BSParams
    ) -> None:
        """Test that digital delta has correct sign and reasonable magnitude."""
        K = 100.0
        delta = bs_digital_delta(spot_prices, K, bs_params)

        # Delta should be non-negative for a digital call
        assert torch.all(delta >= 0)

        # Delta should be finite
        assert torch.all(torch.isfinite(delta))

        # Delta should be largest near the strike
        spot_near_strike = torch.tensor([[99.0], [100.0], [101.0]], dtype=torch.float64)
        delta_near = bs_digital_delta(spot_near_strike, K, bs_params)

        spot_far = torch.tensor([[80.0], [120.0]], dtype=torch.float64)
        delta_far = bs_digital_delta(spot_far, K, bs_params)

        # Delta near strike should be larger than far from strike
        assert delta_near.max() > delta_far.max()


class TestVanillaOptions:
    """Test suite for vanilla option pricing functions."""

    def test_call_price_shape(
        self, spot_prices: torch.Tensor, bs_params: BSParams
    ) -> None:
        """Test that bs_call_price returns correct shape."""
        K = 100.0
        price = bs_call_price(spot_prices, K, bs_params)

        assert isinstance(price, torch.Tensor)
        assert price.shape == spot_prices.shape
        assert price.dtype == torch.float64

    def test_call_gamma_shape(
        self, spot_prices: torch.Tensor, bs_params: BSParams
    ) -> None:
        """Test that bs_call_gamma returns correct shape."""
        K = 100.0
        gamma = bs_call_gamma(spot_prices, K, bs_params)

        assert isinstance(gamma, torch.Tensor)
        assert gamma.shape == spot_prices.shape
        assert gamma.dtype == torch.float64

    def test_call_price_limits(self, bs_params: BSParams) -> None:
        """Test call price behavior at extreme spot values.

        For S >> K: call price should be close to S - K*e^(-rT)
        For S << K: call price should be close to 0
        """
        K = 100.0
        discount = torch.exp(torch.tensor(-bs_params.r * bs_params.T))

        # Test S >> K (deep in the money)
        s_high = torch.tensor([[500.0]], dtype=torch.float64)
        price_high = bs_call_price(s_high, K, bs_params)
        intrinsic = s_high - K * discount
        assert torch.allclose(price_high, intrinsic, rtol=1e-2)

        # Test S << K (deep out of the money)
        s_low = torch.tensor([[10.0]], dtype=torch.float64)
        price_low = bs_call_price(s_low, K, bs_params)
        assert price_low.item() < 0.01  # Should be close to 0

    def test_call_gamma_properties(
        self, spot_prices: torch.Tensor, bs_params: BSParams
    ) -> None:
        """Test that call gamma has expected properties.

        Gamma should:
        - Be positive for all spot prices
        - Peak near the strike (at-the-money)
        - Approach zero for deep ITM/OTM
        """
        K = 100.0
        gamma = bs_call_gamma(spot_prices, K, bs_params)

        # Gamma should be positive
        assert torch.all(gamma >= 0)

        # Gamma should be finite
        assert torch.all(torch.isfinite(gamma))

        # Test that gamma peaks near the strike
        spot_atm = torch.tensor([[100.0]], dtype=torch.float64)
        gamma_atm = bs_call_gamma(spot_atm, K, bs_params)

        spot_otm = torch.tensor([[80.0]], dtype=torch.float64)
        gamma_otm = bs_call_gamma(spot_otm, K, bs_params)

        spot_itm = torch.tensor([[120.0]], dtype=torch.float64)
        gamma_itm = bs_call_gamma(spot_itm, K, bs_params)

        # Gamma at-the-money should be larger than away from money
        assert gamma_atm > gamma_otm
        assert gamma_atm > gamma_itm

        # Test that gamma approaches zero for extreme values
        spot_extreme_low = torch.tensor([[50.0]], dtype=torch.float64)
        gamma_extreme_low = bs_call_gamma(spot_extreme_low, K, bs_params)
        assert gamma_extreme_low.item() < 0.001

        spot_extreme_high = torch.tensor([[200.0]], dtype=torch.float64)
        gamma_extreme_high = bs_call_gamma(spot_extreme_high, K, bs_params)
        assert gamma_extreme_high.item() < 0.001

    def test_call_price_monotonicity(
        self, bs_params: BSParams
    ) -> None:
        """Test that call price is monotonically increasing in spot."""
        K = 100.0
        spots = torch.linspace(90.0, 110.0, 20, dtype=torch.float64).reshape(-1, 1)
        prices = bs_call_price(spots, K, bs_params)

        # Check that prices are monotonically increasing
        price_diffs = prices[1:] - prices[:-1]
        assert torch.all(price_diffs >= 0)

    def test_multiple_strikes(
        self, spot_prices: torch.Tensor, bs_params: BSParams
    ) -> None:
        """Test that functions work with tensor strikes."""
        # Test with multiple strikes
        strikes = torch.tensor([90.0, 100.0, 110.0], dtype=torch.float64)

        # Single spot, multiple strikes
        spot_single = torch.tensor([[100.0]], dtype=torch.float64)
        prices = bs_call_price(spot_single, strikes, bs_params)
        assert prices.shape == (1, 3)

        gammas = bs_call_gamma(spot_single, strikes, bs_params)
        assert gammas.shape == (1, 3)


class TestNumericalStability:
    """Test numerical stability of pricing functions."""

    def test_no_nans_or_infs(self, bs_params: BSParams) -> None:
        """Test that functions don't produce NaN or Inf values."""
        # Test with various spot values including edge cases
        spots = torch.tensor(
            [[0.01], [50.0], [100.0], [200.0], [1000.0]],
            dtype=torch.float64
        )
        K = 100.0

        # Test digital options
        digital_price = bs_digital_price(spots, K, bs_params)
        assert torch.all(torch.isfinite(digital_price))

        digital_delta = bs_digital_delta(spots, K, bs_params)
        assert torch.all(torch.isfinite(digital_delta))

        # Test vanilla options
        call_price = bs_call_price(spots, K, bs_params)
        assert torch.all(torch.isfinite(call_price))

        call_gamma = bs_call_gamma(spots, K, bs_params)
        assert torch.all(torch.isfinite(call_gamma))

    def test_zero_volatility_edge_case(self) -> None:
        """Test behavior with very small volatility."""
        params = BSParams(r=0.05, sigma=0.001, T=0.25)  # Very small sigma
        K = 100.0
        spots = torch.tensor([[99.0], [100.0], [101.0]], dtype=torch.float64)

        # Should still produce finite values
        price = bs_digital_price(spots, K, params)
        assert torch.all(torch.isfinite(price))

        delta = bs_digital_delta(spots, K, params)
        assert torch.all(torch.isfinite(delta))

    def test_zero_time_edge_case(self) -> None:
        """Test behavior at expiry (T close to 0)."""
        params = BSParams(r=0.05, sigma=0.2, T=0.001)  # Very small T
        K = 100.0

        # At expiry, digital should be 0 or 1 (discounted)
        spot_itm = torch.tensor([[101.0]], dtype=torch.float64)
        price_itm = bs_digital_price(spot_itm, K, params)
        assert price_itm.item() > 0.9  # Should be close to 1

        spot_otm = torch.tensor([[99.0]], dtype=torch.float64)
        price_otm = bs_digital_price(spot_otm, K, params)
        assert price_otm.item() < 0.1  # Should be close to 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])