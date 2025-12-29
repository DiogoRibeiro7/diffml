"""Integration tests for end-to-end DiffML workflows.

These tests verify complete experiment pipelines from configuration
loading through training, evaluation, and result generation.
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
import numpy as np
from typing import Dict, Any

# Import core modules
from diffml.config import BSParams, NetworkConfig, TrainingConfig
from diffml.unified_config import (
    UnifiedExperimentConfig,
    load_experiment_config,
    create_default_config
)
from diffml.networks import FeedForwardNet, DifferentialNet
from diffml.training import train_model
from diffml.losses import dml_loss
from diffml.simulation import simulate_bs_terminal

# Import datasets
from diffml.datasets_digital import generate_digital_dataset_train
from diffml.datasets_barrier import generate_barrier_dataset_with_paths
from diffml.datasets_american import generate_american_option_dataset
from diffml.datasets_multibarrier import generate_multibarrier_dataset, MultiBarrierParams, BarrierType

# Import optimizations
from diffml.gpu_optimization import GPUOptimizer, GPUConfig, BatchedMonteCarloSimulator

# Fixtures for common test data
@pytest.fixture
def bs_params():
    """Standard Black-Scholes parameters."""
    return BSParams(r=0.05, sigma=0.2, T=1.0)


@pytest.fixture
def network_config():
    """Standard network configuration."""
    return NetworkConfig(
        input_dim=5,
        hidden_dims=[32, 16, 8],
        output_dim=1,
        activation='relu'
    )


@pytest.fixture
def training_config():
    """Standard training configuration."""
    return TrainingConfig(
        n_epochs=10,  # Small for testing
        batch_size=32,
        lr_initial=0.001,
        lambda_delta=1.0
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestDigitalOptionWorkflow:
    """Test complete digital option pricing workflow."""

    def test_end_to_end_digital_training(self, bs_params, network_config, training_config):
        """Test training a model for digital option pricing."""
        # Generate dataset
        S, K, prices, deltas = generate_digital_dataset_train(
            m=1000,
            n_paths=100,
            r=bs_params.r,
            sigma=bs_params.sigma,
            T=bs_params.T,
            x_min=0.8,
            x_max=1.2
        )

        # Create features
        features = torch.stack([
            S, K,
            torch.full_like(S, bs_params.r),
            torch.full_like(S, bs_params.sigma),
            torch.full_like(S, bs_params.T)
        ], dim=1)

        # Create dataset
        dataset = torch.utils.data.TensorDataset(features, prices, deltas)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        # Create model
        model = FeedForwardNet(
            input_dim=network_config.input_dim,
            hidden_dims=network_config.hidden_dims,
            output_dim=network_config.output_dim
        )

        # Train model
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.lr_initial)
        device = torch.device('cpu')  # Use CPU for CI
        model = model.to(device)

        initial_loss = None
        final_loss = None

        for epoch in range(training_config.n_epochs):
            epoch_loss = 0
            for batch_features, batch_prices, batch_deltas in dataloader:
                batch_features = batch_features.to(device)
                batch_prices = batch_prices.to(device)
                batch_deltas = batch_deltas.to(device)

                optimizer.zero_grad()

                # Forward pass
                predictions = model(batch_features)

                # DML loss
                loss = dml_loss(
                    predictions,
                    batch_prices,
                    batch_deltas,
                    batch_features,
                    lambda_val=training_config.lambda_delta
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            if epoch == 0:
                initial_loss = avg_loss
            if epoch == training_config.n_epochs - 1:
                final_loss = avg_loss

        # Verify training improved loss
        assert final_loss < initial_loss, "Model should improve during training"
        assert final_loss < 0.1, "Final loss should be reasonably small"

        # Test inference
        model.eval()
        with torch.no_grad():
            test_input = torch.tensor([[100.0, 100.0, 0.05, 0.2, 1.0]], device=device)
            prediction = model(test_input)
            assert prediction.shape == (1, 1)
            assert 0 <= prediction.item() <= 1, "Digital option price should be between 0 and 1"


class TestAmericanOptionWorkflow:
    """Test American option pricing workflow."""

    def test_american_option_dataset_generation(self, bs_params):
        """Test generating American option dataset."""
        from diffml.datasets_american import AmericanOptionParams

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

        # Validate shapes
        assert features.shape == (100, 5)
        assert prices.shape == (100,)
        assert deltas.shape == (100,)
        assert gammas.shape == (100,)

        # Validate values
        assert torch.all(prices >= 0), "Prices should be non-negative"
        assert torch.all(torch.isfinite(prices)), "Prices should be finite"
        assert torch.all(torch.isfinite(deltas)), "Deltas should be finite"


class TestMultiBarrierWorkflow:
    """Test multi-barrier option workflow."""

    def test_multibarrier_option_pricing(self, bs_params):
        """Test multi-barrier option dataset and pricing."""
        barrier_params = MultiBarrierParams(
            barrier_type=BarrierType.DOUBLE_OUT,
            lower_barrier=85.0,
            upper_barrier=115.0,
            rebate=0.0
        )

        features, prices, deltas, gammas = generate_multibarrier_dataset(
            bs_params=bs_params,
            n_samples=50,
            barrier_params=barrier_params,
            n_paths=1000,
            n_steps=50,
            compute_greeks=True
        )

        # Validate dataset
        assert features.shape == (50, 5)
        assert prices.shape == (50,)

        # Prices should be less than vanilla due to barrier
        assert torch.all(prices >= 0)
        assert torch.all(prices < 20), "Barrier option prices should be reasonable"

    def test_parisian_barrier(self, bs_params):
        """Test Parisian barrier implementation."""
        from diffml.datasets_multibarrier import check_parisian_barrier

        # Create sample paths
        n_paths = 100
        n_steps = 50
        paths = torch.randn(n_paths, n_steps) * 10 + 100

        # Set some paths to breach barrier
        paths[0, :10] = 120  # Consecutive breach
        paths[1, 0] = 120    # Single breach
        paths[1, 1] = 95     # Back inside

        survived = check_parisian_barrier(
            paths,
            barrier=115.0,
            threshold_steps=5,
            barrier_type='up'
        )

        assert survived.shape == (n_paths,)
        assert not survived[0], "Path with long breach should be knocked out"
        assert survived[1], "Path with short breach should survive"


class TestGPUOptimizationWorkflow:
    """Test GPU optimization strategies."""

    def test_batched_monte_carlo(self):
        """Test batched Monte Carlo simulation."""
        simulator = BatchedMonteCarloSimulator(batch_size=1000)

        paths = simulator.simulate_paths_batched(
            S0=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0,
            n_steps=100,
            n_paths=5000,
            antithetic=True,
            seed=42
        )

        assert paths.shape == (5000, 101)

        # Verify properties
        final_prices = paths[:, -1]
        mean_price = final_prices.mean().item()
        expected_mean = 100 * np.exp(0.05 * 1.0)

        # Should be close to expected value
        assert abs(mean_price - expected_mean) / expected_mean < 0.05

    def test_gpu_optimizer_config(self):
        """Test GPU optimizer configuration."""
        config = GPUConfig(
            enable_mixed_precision=True,
            enable_cudnn_benchmark=True,
            batch_size_finder=False
        )

        optimizer = GPUOptimizer(config)
        assert optimizer.config.enable_mixed_precision == True

        # Test mixed precision context
        with optimizer.mixed_precision_context():
            # Operations here would use mixed precision if available
            x = torch.randn(10, 10)
            y = x * 2


class TestConfigurationWorkflow:
    """Test configuration loading and validation."""

    def test_config_creation_and_validation(self):
        """Test creating and validating configurations."""
        config = UnifiedExperimentConfig(
            name="test_experiment",
            seed=42,
            training=TrainingConfig(),
            m_train=1000,
            m_test=200,
            n_paths_train=100,
            n_paths_test=1000,
            K=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0
        )

        # Validate configuration
        config.validate()  # Should not raise

        # Test invalid configuration
        with pytest.raises(ValueError):
            invalid_config = UnifiedExperimentConfig(
                name="invalid",
                seed=42,
                training=TrainingConfig(),
                m_train=-100,  # Invalid: negative
                m_test=200,
                n_paths_train=100,
                n_paths_test=1000
            )

    def test_config_loading_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            'name': 'test',
            'seed': 42,
            'training': {
                'n_epochs': 100,
                'batch_size': 64,
                'lr_initial': 0.001
            },
            'm_train': 5000,
            'm_test': 1000,
            'n_paths_train': 1000,
            'n_paths_test': 10000,
            'K': 100.0
        }

        config = load_experiment_config(config_dict)
        assert config.name == 'test'
        assert config.m_train == 5000
        assert config.training.n_epochs == 100

    def test_default_config_generation(self):
        """Test generating default configurations."""
        for exp_type in ['digital', 'barrier', 'basket', 'american', 'multibarrier']:
            config = create_default_config(exp_type)
            assert config.name == exp_type
            config.validate()  # Should not raise


class TestEndToEndPipeline:
    """Test complete experiment pipeline."""

    def test_complete_experiment_pipeline(self, temp_dir):
        """Test running a complete experiment from config to results."""
        # Step 1: Create configuration
        config = create_default_config('digital')
        config.m_train = 500  # Small for testing
        config.m_test = 100
        config.n_paths_train = 100
        config.n_paths_test = 500
        config.training.n_epochs = 5

        # Step 2: Generate dataset
        S, K, prices, deltas = generate_digital_dataset_train(
            m=config.m_train,
            n_paths=config.n_paths_train,
            r=config.r,
            sigma=config.sigma,
            T=config.T,
            x_min=config.x_min,
            x_max=config.x_max
        )

        features = torch.stack([
            S, K,
            torch.full_like(S, config.r),
            torch.full_like(S, config.sigma),
            torch.full_like(S, config.T)
        ], dim=1)

        # Step 3: Create and train model
        model = DifferentialNet(
            input_dim=5,
            hidden_dims=[32, 16],
            output_dim=1
        )

        dataset = torch.utils.data.TensorDataset(features, prices, deltas)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr_initial)
        device = torch.device('cpu')
        model = model.to(device)

        # Training loop
        train_losses = []
        for epoch in range(config.training.n_epochs):
            epoch_loss = 0
            for batch_features, batch_prices, batch_deltas in dataloader:
                optimizer.zero_grad()
                predictions = model(batch_features)
                loss = dml_loss(
                    predictions,
                    batch_prices,
                    batch_deltas,
                    batch_features,
                    lambda_val=1.0
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_loss)

        # Step 4: Evaluate on test set
        S_test, K_test, prices_test, deltas_test = generate_digital_dataset_train(
            m=config.m_test,
            n_paths=config.n_paths_test,
            r=config.r,
            sigma=config.sigma,
            T=config.T,
            x_min=config.x_min,
            x_max=config.x_max
        )

        features_test = torch.stack([
            S_test, K_test,
            torch.full_like(S_test, config.r),
            torch.full_like(S_test, config.sigma),
            torch.full_like(S_test, config.T)
        ], dim=1)

        model.eval()
        with torch.no_grad():
            predictions_test = model(features_test)
            test_loss = dml_loss(
                predictions_test,
                prices_test,
                deltas_test,
                features_test,
                lambda_val=1.0
            ).item()

        # Step 5: Save results
        results = {
            'config': config.to_dict(),
            'train_losses': train_losses,
            'test_loss': test_loss,
            'final_train_loss': train_losses[-1]
        }

        results_path = temp_dir / 'results.json'
        with open(results_path, 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_results = {
                'config': {k: str(v) if isinstance(v, TrainingConfig) else v
                          for k, v in results['config'].items()},
                'train_losses': results['train_losses'],
                'test_loss': results['test_loss'],
                'final_train_loss': results['final_train_loss']
            }
            json.dump(json_results, f, indent=2)

        # Verify results
        assert results_path.exists()
        assert train_losses[-1] < train_losses[0], "Training should improve loss"
        assert test_loss < 1.0, "Test loss should be reasonable"

        # Step 6: Save model
        model_path = temp_dir / 'model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'test_loss': test_loss
        }, model_path)

        assert model_path.exists()

        # Step 7: Load and verify model
        loaded_model = DifferentialNet(
            input_dim=5,
            hidden_dims=[32, 16],
            output_dim=1
        )
        checkpoint = torch.load(model_path)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model.eval()

        # Verify loaded model produces same predictions
        with torch.no_grad():
            original_pred = model(features_test[:10])
            loaded_pred = loaded_model(features_test[:10])
            assert torch.allclose(original_pred, loaded_pred, atol=1e-6)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_barrier_configuration(self):
        """Test handling of invalid barrier configurations."""
        with pytest.raises(ValueError):
            # Lower barrier > Upper barrier
            params = MultiBarrierParams(
                barrier_type=BarrierType.DOUBLE_OUT,
                lower_barrier=120.0,  # Invalid: higher than upper
                upper_barrier=80.0
            )

    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        with pytest.raises(ValueError):
            # Should fail with m=0
            S, K, prices, deltas = generate_digital_dataset_train(
                m=0,  # Invalid: no samples
                n_paths=100,
                r=0.05,
                sigma=0.2,
                T=1.0
            )

    def test_nan_handling_in_training(self):
        """Test handling of NaN values during training."""
        # Create dataset with NaN
        features = torch.randn(100, 5)
        prices = torch.randn(100)
        deltas = torch.randn(100)

        # Introduce NaN
        prices[50] = float('nan')

        model = FeedForwardNet(5, [10], 1)
        optimizer = torch.optim.Adam(model.parameters())

        # This should handle NaN gracefully
        dataset = torch.utils.data.TensorDataset(features, prices, deltas)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        for batch_features, batch_prices, batch_deltas in dataloader:
            if torch.any(torch.isnan(batch_prices)):
                # Skip batches with NaN
                continue

            optimizer.zero_grad()
            predictions = model(batch_features)
            # Loss computation would fail with NaN
            assert not torch.any(torch.isnan(predictions))


@pytest.mark.slow
class TestPerformance:
    """Performance tests (marked as slow)."""

    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        import time

        start_time = time.time()

        # Generate large dataset
        S, K, prices, deltas = generate_digital_dataset_train(
            m=10000,
            n_paths=1000,
            r=0.05,
            sigma=0.2,
            T=1.0
        )

        generation_time = time.time() - start_time
        assert generation_time < 60, f"Dataset generation too slow: {generation_time:.2f}s"

        # Test model training speed
        features = torch.stack([S[:1000], K[:1000],
                               torch.full((1000,), 0.05),
                               torch.full((1000,), 0.2),
                               torch.full((1000,), 1.0)], dim=1)

        model = FeedForwardNet(5, [32, 16], 1)
        optimizer = torch.optim.Adam(model.parameters())

        start_time = time.time()
        for _ in range(10):
            optimizer.zero_grad()
            pred = model(features)
            loss = pred.mean()
            loss.backward()
            optimizer.step()

        training_time = time.time() - start_time
        assert training_time < 10, f"Training too slow: {training_time:.2f}s"


# Test utilities
def assert_tensor_properties(tensor: torch.Tensor, name: str, **properties):
    """Helper to assert tensor properties.

    Parameters:
        tensor: Tensor to check
        name: Name for error messages
        **properties: Properties to check (shape, dtype, min, max, etc.)
    """
    if 'shape' in properties:
        assert tensor.shape == properties['shape'], \
            f"{name} shape mismatch: {tensor.shape} != {properties['shape']}"

    if 'dtype' in properties:
        assert tensor.dtype == properties['dtype'], \
            f"{name} dtype mismatch: {tensor.dtype} != {properties['dtype']}"

    if 'min' in properties:
        assert tensor.min() >= properties['min'], \
            f"{name} min value {tensor.min()} < {properties['min']}"

    if 'max' in properties:
        assert tensor.max() <= properties['max'], \
            f"{name} max value {tensor.max()} > {properties['max']}"

    if 'finite' in properties and properties['finite']:
        assert torch.all(torch.isfinite(tensor)), f"{name} contains non-finite values"


if __name__ == "__main__":
    pytest.main([__file__, '-v'])