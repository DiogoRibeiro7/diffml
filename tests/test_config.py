"""Tests for configuration module."""

import pytest
import torch

from diffml.config import (
    ExperimentConfig,
    NetworkConfig,
    SimulationConfig,
    TrainingConfig,
    get_device,
    set_random_seeds,
)


def test_network_config():
    """Test NetworkConfig initialization."""
    config = NetworkConfig(
        input_dim=5,
        hidden_dims=[100, 50],
        output_dim=2,
        activation="tanh",
        dropout_rate=0.2,
    )

    assert config.input_dim == 5
    assert config.hidden_dims == [100, 50]
    assert config.output_dim == 2
    assert config.activation == "tanh"
    assert config.dropout_rate == 0.2


def test_network_config_defaults():
    """Test NetworkConfig with default values."""
    config = NetworkConfig()

    assert config.input_dim == 1
    assert config.hidden_dims == [50, 50, 50]
    assert config.output_dim == 1
    assert config.activation == "relu"
    assert config.dropout_rate == 0.0


def test_training_config():
    """Test TrainingConfig initialization."""
    config = TrainingConfig(
        batch_size=128,
        n_epochs=200,
    )

    config.learning_rate = 5e-4
    assert config.batch_size == 128
    assert config.learning_rate == 5e-4
    assert config.lr_initial == 5e-4
    assert config.n_epochs == 200


def test_simulation_config():
    """Test SimulationConfig initialization."""
    config = SimulationConfig(
        n_paths=5000,
        n_timesteps=50,
        seed=123,
        antithetic=False,
    )

    assert config.n_paths == 5000
    assert config.n_timesteps == 50
    assert config.seed == 123
    assert config.antithetic is False


def test_experiment_config():
    """Test ExperimentConfig initialization."""
    config = ExperimentConfig(name="test_experiment")

    assert config.name == "test_experiment"
    assert isinstance(config.network, NetworkConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.simulation, SimulationConfig)
    assert config.output_dir == "output"


def test_get_device():
    """Test device selection."""
    # Test auto-select
    device = get_device()
    assert isinstance(device, torch.device)

    # Test CPU selection
    cpu_device = get_device("cpu")
    assert cpu_device.type == "cpu"

    # Test CUDA selection (will fall back to CPU if not available)
    cuda_device = get_device("cuda")
    if torch.cuda.is_available():
        assert cuda_device.type == "cuda"
    else:
        assert cuda_device.type == "cpu"

    # Unknown preference should raise
    with pytest.raises(ValueError):
        get_device("tpu")


def test_set_random_seeds():
    """Test random seed setting."""
    # This should run without errors
    set_random_seeds(42)

    # Verify reproducibility
    import random

    import numpy as np

    set_random_seeds(42)
    random_val1 = random.random()
    np_val1 = np.random.random()
    torch_val1 = torch.rand(1).item()

    set_random_seeds(42)
    random_val2 = random.random()
    np_val2 = np.random.random()
    torch_val2 = torch.rand(1).item()

    assert random_val1 == random_val2
    assert np_val1 == np_val2
    assert torch_val1 == torch_val2


if __name__ == "__main__":
    test_network_config()
    test_network_config_defaults()
    test_training_config()
    test_simulation_config()
    test_experiment_config()
    test_get_device()
    test_set_random_seeds()
    print("All configuration tests passed!")
