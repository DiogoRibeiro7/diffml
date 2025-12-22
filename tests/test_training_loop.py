"""Tests for the training loop and neural network components.

This module tests the training loop functionality, ensuring models
can be trained without errors and show learning progress.
"""

from typing import cast

import pytest
import torch
from torch.utils.data import TensorDataset

from diffml.config import (
    BSParams,
    TrainingConfig,
    get_device,
    set_default_dtype,
)
from diffml.datasets_digital import make_digital_dataset
from diffml.losses import dml_loss
from diffml.networks import PricingNet
from diffml.training import (
    Mode,
    nn_value_delta_gamma,
    rmse,
    train_model,
)


@pytest.fixture(autouse=True)
def setup_precision() -> None:
    """Set default dtype to float64 for all tests."""
    set_default_dtype()


@pytest.fixture
def device() -> torch.device:
    """Get compute device for tests."""
    return get_device()


@pytest.fixture
def tiny_dataset() -> tuple[TensorDataset, torch.Tensor, torch.Tensor]:
    """Create a tiny toy dataset for quick testing."""
    # Create small digital option dataset
    params = BSParams(r=0.05, sigma=0.2, T=0.25)
    m = 16  # Small number of samples
    n_paths = 10  # Few paths for speed

    x, price, delta_pw, delta_lrm = make_digital_dataset(
        m=m,
        K=100.0,
        params=params,
        x_min=90.0,
        x_max=110.0,
        n_paths_per_x=n_paths,
        seed=42
    )

    # Create dataset
    dataset = TensorDataset(x, price, delta_pw, delta_lrm)

    # Also return test data for evaluation
    x_test, price_test, _, _ = make_digital_dataset(
        m=8,
        K=100.0,
        params=params,
        x_min=85.0,
        x_max=115.0,
        n_paths_per_x=20,
        seed=123
    )

    return dataset, x_test, price_test


class TestPricingNet:
    """Test suite for PricingNet neural network."""

    def test_network_creation(self) -> None:
        """Test that PricingNet can be created with various configurations."""
        # Default configuration
        net1 = PricingNet()
        assert net1.input_dim == 1
        assert net1.hidden_dim == 20
        assert net1.n_hidden == 4

        # Custom configuration
        net2 = PricingNet(input_dim=5, hidden_dim=32, n_hidden=2)
        assert net2.input_dim == 5
        assert net2.hidden_dim == 32
        assert net2.n_hidden == 2

    def test_network_forward_pass(self, device: torch.device) -> None:
        """Test that network forward pass works correctly."""
        net = PricingNet(input_dim=1, hidden_dim=10, n_hidden=2)
        net = net.to(device)

        # Test with batch of inputs
        batch_size = 8
        x = torch.randn(batch_size, 1, device=device, dtype=torch.float64)

        # Forward pass
        output = net(x)

        # Check output shape
        assert output.shape == (batch_size, 1)
        assert output.dtype == torch.float64

        # Check that outputs are finite
        assert torch.all(torch.isfinite(output))

    def test_network_validation(self) -> None:
        """Test that network validates input parameters."""
        # Should raise error for invalid dimensions
        with pytest.raises(ValueError):
            PricingNet(input_dim=0)

        with pytest.raises(ValueError):
            PricingNet(hidden_dim=0)

        with pytest.raises(ValueError):
            PricingNet(n_hidden=0)

        with pytest.raises(ValueError):
            PricingNet(input_dim=-1)


class TestValueDeltaGamma:
    """Test suite for nn_value_delta_gamma function."""

    def test_value_only(self, device: torch.device) -> None:
        """Test computing only values."""
        net = PricingNet().to(device)
        x = torch.linspace(90.0, 110.0, 10, device=device, dtype=torch.float64).reshape(-1, 1)

        value, delta, gamma = nn_value_delta_gamma(
            net, x, compute_delta=False, compute_gamma=False
        )

        # Check shapes
        assert value.shape == x.shape
        assert delta is None
        assert gamma is None

        # Check values are finite
        assert torch.all(torch.isfinite(value))

    def test_value_and_delta(self, device: torch.device) -> None:
        """Test computing values and first derivatives."""
        net = PricingNet().to(device)
        x = torch.linspace(90.0, 110.0, 10, device=device, dtype=torch.float64).reshape(-1, 1)
        x.requires_grad_(True)

        value, delta, gamma = nn_value_delta_gamma(
            net, x, compute_delta=True, compute_gamma=False
        )

        # Check shapes
        assert value.shape == x.shape
        assert delta is not None
        assert delta.shape == x.shape
        assert gamma is None

        # Check values are finite
        assert torch.all(torch.isfinite(value))
        assert torch.all(torch.isfinite(delta))

    def test_value_delta_gamma(self, device: torch.device) -> None:
        """Test computing values and both derivatives."""
        net = PricingNet().to(device)
        x = torch.linspace(90.0, 110.0, 10, device=device, dtype=torch.float64).reshape(-1, 1)
        x.requires_grad_(True)

        value, delta, gamma = nn_value_delta_gamma(
            net, x, compute_delta=True, compute_gamma=True
        )

        # Check shapes
        assert value.shape == x.shape
        assert delta is not None
        assert gamma is not None
        assert delta.shape == x.shape
        assert gamma.shape == x.shape

        # Check values are finite
        assert torch.all(torch.isfinite(value))
        assert torch.all(torch.isfinite(delta))
        assert torch.all(torch.isfinite(gamma))

    def test_multi_dimensional_input(self, device: torch.device) -> None:
        """Test with multi-dimensional input."""
        d = 5
        net = PricingNet(input_dim=d).to(device)
        batch_size = 8
        x = torch.randn(batch_size, d, device=device, dtype=torch.float64)
        x.requires_grad_(True)

        value, delta, gamma = nn_value_delta_gamma(
            net, x, compute_delta=True, compute_gamma=False
        )

        # Check shapes
        assert value.shape == (batch_size, 1)
        assert delta is not None
        assert delta.shape == (batch_size, d)  # Gradient w.r.t. each input
        assert gamma is None


class TestLossFunctions:
    """Test suite for loss functions."""

    def test_dml_loss_price_only(self) -> None:
        """Test DML loss with price only."""
        batch_size = 16
        pred_price = torch.randn(batch_size, 1, dtype=torch.float64)
        true_price = torch.randn(batch_size, 1, dtype=torch.float64)

        loss = dml_loss(
            pred_price=pred_price,
            true_price=true_price,
            lambda_delta=0.0,
            lambda_gamma=0.0
        )

        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # MSE is non-negative
        assert torch.isfinite(loss)

    def test_dml_loss_with_delta(self) -> None:
        """Test DML loss with price and delta."""
        batch_size = 16
        pred_price = torch.randn(batch_size, 1, dtype=torch.float64)
        true_price = torch.randn(batch_size, 1, dtype=torch.float64)
        pred_delta = torch.randn(batch_size, 1, dtype=torch.float64)
        true_delta = torch.randn(batch_size, 1, dtype=torch.float64)

        loss = dml_loss(
            pred_price=pred_price,
            true_price=true_price,
            pred_delta_scalar=pred_delta,
            true_delta_scalar=true_delta,
            lambda_delta=1.0,
            lambda_gamma=0.0
        )

        assert loss.ndim == 0
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_dml_loss_with_gamma(self) -> None:
        """Test DML loss with price, delta, and gamma."""
        batch_size = 16
        pred_price = torch.randn(batch_size, 1, dtype=torch.float64)
        true_price = torch.randn(batch_size, 1, dtype=torch.float64)
        pred_delta = torch.randn(batch_size, 1, dtype=torch.float64)
        true_delta = torch.randn(batch_size, 1, dtype=torch.float64)
        pred_gamma = torch.randn(batch_size, 1, dtype=torch.float64)
        true_gamma = torch.randn(batch_size, 1, dtype=torch.float64)

        loss = dml_loss(
            pred_price=pred_price,
            true_price=true_price,
            pred_delta_scalar=pred_delta,
            true_delta_scalar=true_delta,
            pred_gamma=pred_gamma,
            true_gamma=true_gamma,
            lambda_delta=1.0,
            lambda_gamma=0.5
        )

        assert loss.ndim == 0
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_dml_loss_vector_delta(self) -> None:
        """Test DML loss with vector delta."""
        batch_size = 16
        d = 5
        pred_price = torch.randn(batch_size, 1, dtype=torch.float64)
        true_price = torch.randn(batch_size, 1, dtype=torch.float64)
        pred_delta = torch.randn(batch_size, d, dtype=torch.float64)
        true_delta = torch.randn(batch_size, d, dtype=torch.float64)

        loss = dml_loss(
            pred_price=pred_price,
            true_price=true_price,
            pred_delta_vector=pred_delta,
            true_delta_vector=true_delta,
            lambda_delta=1.0,
            lambda_gamma=0.0
        )

        assert loss.ndim == 0
        assert loss.item() >= 0
        assert torch.isfinite(loss)


class TestTrainingLoop:
    """Test suite for the complete training loop."""

    def test_train_standard_mode(
        self,
        tiny_dataset: tuple[TensorDataset, torch.Tensor, torch.Tensor],
        device: torch.device
    ) -> None:
        """Test training with standard mode (price only)."""
        dataset, x_test, price_test = tiny_dataset

        # Create model
        model = PricingNet(input_dim=1, hidden_dim=10, n_hidden=2)

        # Training config with small epochs
        config = TrainingConfig(
            n_epochs=20,
            batch_size=8,
            lr_initial=1e-3,
            lr_min=1e-5,
            lambda_delta=0.0,
            lambda_gamma=0.0
        )

        # Get initial loss
        model.eval()
        with torch.no_grad():
            pred_initial, _, _ = nn_value_delta_gamma(model, x_test, False, False)
            loss_initial = rmse(pred_initial, price_test)

        # Train model
        model_trained = train_model(
            model=model,
            dataset=dataset,
            config=config,
            mode="standard",
            device=device
        )

        # Get final loss
        model_trained.eval()
        with torch.no_grad():
            pred_final, _, _ = nn_value_delta_gamma(model_trained, x_test, False, False)
            loss_final = rmse(pred_final, price_test)

        # Training should reduce loss (with some tolerance for small datasets)
        assert loss_final < loss_initial * 1.5  # Generous tolerance

        # Model should produce finite outputs
        assert torch.all(torch.isfinite(pred_final))

    def test_train_delta_lrm_mode(
        self,
        tiny_dataset: tuple[TensorDataset, torch.Tensor, torch.Tensor],
        device: torch.device
    ) -> None:
        """Test training with delta LRM mode."""
        dataset, x_test, _ = tiny_dataset

        # Create model
        model = PricingNet(input_dim=1, hidden_dim=10, n_hidden=2)

        # Training config with delta regularization
        config = TrainingConfig(
            n_epochs=20,
            batch_size=8,
            lr_initial=1e-3,
            lr_min=1e-5,
            lambda_delta=1.0,
            lambda_gamma=0.0
        )

        # Train model
        model_trained = train_model(
            model=model,
            dataset=dataset,
            config=config,
            mode="delta_lrm",
            device=device
        )

        # Check that model produces finite outputs
        model_trained.eval()
        with torch.no_grad():
            pred_price, pred_delta, _ = nn_value_delta_gamma(
                model_trained, x_test, compute_delta=True, compute_gamma=False
            )

        assert torch.all(torch.isfinite(pred_price))
        assert pred_delta is not None
        assert torch.all(torch.isfinite(pred_delta))

    def test_train_delta_pathwise_mode(
        self,
        tiny_dataset: tuple[TensorDataset, torch.Tensor, torch.Tensor],
        device: torch.device
    ) -> None:
        """Test training with delta pathwise mode."""
        dataset, x_test, _ = tiny_dataset

        model = PricingNet(input_dim=1, hidden_dim=10, n_hidden=2)

        config = TrainingConfig(
            n_epochs=20,
            batch_size=8,
            lr_initial=1e-3,
            lr_min=1e-5,
            lambda_delta=1.0,
            lambda_gamma=0.0
        )

        # Train model
        model_trained = train_model(
            model=model,
            dataset=dataset,
            config=config,
            mode="delta_pathwise",
            device=device
        )

        # Check outputs
        model_trained.eval()
        with torch.no_grad():
            pred_price, pred_delta, _ = nn_value_delta_gamma(
                model_trained, x_test, compute_delta=True, compute_gamma=False
            )

        assert torch.all(torch.isfinite(pred_price))
        assert pred_delta is not None
        assert torch.all(torch.isfinite(pred_delta))

    def test_train_gamma_mode(
        self,
        device: torch.device
    ) -> None:
        """Test training with gamma mode."""
        # Create dataset with gamma labels
        m = 16
        x = torch.linspace(0.8, 1.2, m, dtype=torch.float64).reshape(-1, 1)
        price = torch.randn(m, 1, dtype=torch.float64) * 0.1 + 0.5
        delta = torch.randn(m, 1, dtype=torch.float64) * 0.1
        gamma = torch.randn(m, 1, dtype=torch.float64) * 0.01

        # Gamma mode expects 7-tensor batches matching gamma portfolio format
        dataset = TensorDataset(
            x,           # features
            price,       # true price
            delta,       # analytical delta (used as delta_lrm stand-in)
            gamma,       # analytical gamma (unused placeholder)
            price,       # Monte Carlo price placeholder
            delta,       # pathwise delta placeholder
            gamma        # gamma PWLR labels
        )

        model = PricingNet(input_dim=1, hidden_dim=10, n_hidden=2)

        config = TrainingConfig(
            n_epochs=20,
            batch_size=8,
            lr_initial=1e-3,
            lr_min=1e-5,
            lambda_delta=1.0,
            lambda_gamma=0.5
        )

        # Train model
        model_trained = train_model(
            model=model,
            dataset=dataset,
            config=config,
            mode="gamma_pwlr",
            device=device
        )

        # Check outputs including gamma
        model_trained.eval()
        x_test = torch.linspace(0.9, 1.1, 8, dtype=torch.float64).reshape(-1, 1)
        x_test.requires_grad_(True)

        pred_price, pred_delta, pred_gamma = nn_value_delta_gamma(
            model_trained, x_test, compute_delta=True, compute_gamma=True
        )

        assert torch.all(torch.isfinite(pred_price))
        assert pred_delta is not None
        assert pred_gamma is not None
        assert torch.all(torch.isfinite(pred_delta))
        assert torch.all(torch.isfinite(pred_gamma))

    def test_training_reduces_loss(
        self,
        tiny_dataset: tuple[TensorDataset, torch.Tensor, torch.Tensor],
        device: torch.device
    ) -> None:
        """Test that training actually reduces the loss over epochs."""
        dataset, _, _ = tiny_dataset

        model = PricingNet(input_dim=1, hidden_dim=10, n_hidden=2)

        # Longer training to ensure convergence
        config = TrainingConfig(
            n_epochs=50,
            batch_size=8,
            lr_initial=1e-2,
            lr_min=1e-5,
            lambda_delta=0.0,
            lambda_gamma=0.0
        )

        # Record initial loss
        model.eval()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        x_batch, price_batch, _, _ = next(iter(dataloader))
        x_batch = x_batch.to(device)
        price_batch = price_batch.to(device)

        with torch.no_grad():
            pred_initial = model(x_batch)
            loss_initial = torch.nn.functional.mse_loss(pred_initial, price_batch)

        # Train
        model_trained = train_model(
            model=model,
            dataset=dataset,
            config=config,
            mode="standard",
            device=device
        )

        # Record final loss
        model_trained.eval()
        with torch.no_grad():
            pred_final = model_trained(x_batch)
            loss_final = torch.nn.functional.mse_loss(pred_final, price_batch)

        # Loss should decrease
        assert loss_final < loss_initial

    def test_rmse_function(self) -> None:
        """Test the RMSE calculation function."""
        # Perfect predictions
        pred = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        true = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        assert rmse(pred, true) < 1e-10

        # Known RMSE
        pred = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        true = torch.tensor([[2.0], [3.0], [4.0]], dtype=torch.float64)
        expected_rmse = 1.0  # All errors are 1
        assert abs(rmse(pred, true) - expected_rmse) < 1e-10

        # Multi-dimensional
        pred = torch.randn(10, 5, dtype=torch.float64)
        true = torch.randn(10, 5, dtype=torch.float64)
        error = rmse(pred, true)
        assert isinstance(error, float)
        assert error >= 0


class TestTrainingModes:
    """Test different training modes work correctly."""

    def test_all_modes_run_without_error(
        self,
        tiny_dataset: tuple[TensorDataset, torch.Tensor, torch.Tensor],
        device: torch.device
    ) -> None:
        """Test that all training modes can run without exceptions."""
        dataset, _, _ = tiny_dataset

        config = TrainingConfig(
            n_epochs=5,  # Very few epochs just to test execution
            batch_size=8,
            lr_initial=1e-3,
            lr_min=1e-5,
            lambda_delta=1.0,
            lambda_gamma=0.5
        )

        modes: tuple[Mode, ...] = ("standard", "delta_pathwise", "delta_lrm")

        for mode in modes:
            model = PricingNet(input_dim=1, hidden_dim=10, n_hidden=2)

            # Should not raise any exceptions
            trained_model = train_model(
                model=model,
                dataset=dataset,
                config=config,
                mode=mode,
                device=device
            )

            assert trained_model is not None
            assert isinstance(trained_model, PricingNet)

    def test_invalid_mode_raises_error(
        self,
        tiny_dataset: tuple[TensorDataset, torch.Tensor, torch.Tensor],
        device: torch.device
    ) -> None:
        """Test that invalid training mode raises an error."""
        dataset, _, _ = tiny_dataset
        model = PricingNet()
        config = TrainingConfig(n_epochs=1)

        with pytest.raises(ValueError):
            train_model(
                model=model,
                dataset=dataset,
                config=config,
                mode=cast(Mode, "invalid_mode"),
                device=device
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
