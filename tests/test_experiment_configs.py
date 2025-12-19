"""Tests for experiment configuration system.

This module tests the configuration loading, experiment registry,
and configuration-based experiment execution.
"""

import tempfile
from pathlib import Path

import pytest

from diffml.config import TrainingConfig
from diffml.config_experiments import (
    ExperimentConfig,
    load_experiment_config,
    save_experiment_config,
)
from diffml.experiments_registry import (
    EXPERIMENT_REGISTRY,
    clear_registry,
    get_experiment,
    list_registered_experiments,
    register_experiment,
)
from diffml.run_experiment import run_experiment_from_dict, validate_config


class TestExperimentConfig:
    """Test ExperimentConfig dataclass."""

    def test_experiment_config_creation(self):
        """Test creating an ExperimentConfig."""
        training = TrainingConfig(n_epochs=100, batch_size=32)
        config = ExperimentConfig(
            name="test",
            seed=42,
            training=training,
            m_train=50,
            m_test=10,
            n_paths_train=100,
            n_paths_test=500,
        )

        assert config.name == "test"
        assert config.seed == 42
        assert config.training.n_epochs == 100
        assert config.m_train == 50
        assert config.K is None  # Optional field

    def test_experiment_config_validation(self):
        """Test ExperimentConfig validation."""
        training = TrainingConfig()

        # Invalid m_train
        with pytest.raises(ValueError):
            ExperimentConfig(
                name="test",
                seed=42,
                training=training,
                m_train=0,  # Invalid
                m_test=10,
                n_paths_train=100,
                n_paths_test=500,
            )

        # Invalid x_min >= x_max
        with pytest.raises(ValueError):
            ExperimentConfig(
                name="test",
                seed=42,
                training=training,
                m_train=50,
                m_test=10,
                n_paths_train=100,
                n_paths_test=500,
                x_min=1.5,
                x_max=0.5,  # Invalid
            )

    def test_load_experiment_config(self):
        """Test loading configuration from TOML."""
        toml_content = """
[experiment]
name = "digital"
seed = 123

[training]
n_epochs = 500
batch_size = 128
lr_initial = 0.001
lambda_delta = 0.5

[dataset]
m_train = 256
m_test = 64
n_paths_train = 1000
n_paths_test = 5000
x_min = 0.6
x_max = 1.4

[payoff]
K = 100.0

[black_scholes]
r = 0.05
sigma = 0.25
T = 0.5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            config = load_experiment_config(temp_path)

            assert config.name == "digital"
            assert config.seed == 123
            assert config.training.n_epochs == 500
            assert config.m_train == 256
            assert config.K == 100.0
            assert config.sigma == 0.25
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_load_roundtrip(self):
        """Test saving and loading configuration maintains data."""
        # Create a config
        training = TrainingConfig(n_epochs=200, lambda_delta=0.75)
        config = ExperimentConfig(
            name="test_roundtrip",
            seed=999,
            training=training,
            m_train=100,
            m_test=25,
            n_paths_train=500,
            n_paths_test=2000,
            K=110.0,
            B=85.0,
            d=15,
            eps_multipliers=[0.5, 1.0, 2.0],
        )

        # Save it
        with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as f:
            temp_path = f.name

        try:
            # Note: save_experiment_config requires the 'toml' package
            # Skip this test if not available
            try:
                save_experiment_config(config, temp_path)
            except ImportError:
                pytest.skip("toml package not available for saving")

            # Load it back
            loaded_config = load_experiment_config(temp_path)

            # Check key fields
            assert loaded_config.name == config.name
            assert loaded_config.seed == config.seed
            assert loaded_config.training.n_epochs == config.training.n_epochs
            assert loaded_config.K == config.K
            assert loaded_config.B == config.B
            assert loaded_config.d == config.d
            assert loaded_config.eps_multipliers == config.eps_multipliers

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestExperimentRegistry:
    """Test experiment registry functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        clear_registry()

    def test_register_experiment(self):
        """Test registering an experiment."""

        @register_experiment("test_exp")
        def dummy_experiment(config: ExperimentConfig) -> None:
            pass

        assert "test_exp" in EXPERIMENT_REGISTRY
        assert EXPERIMENT_REGISTRY["test_exp"] == dummy_experiment

    def test_register_duplicate_raises(self):
        """Test that registering duplicate names raises error."""

        @register_experiment("duplicate")
        def exp1(config: ExperimentConfig) -> None:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_experiment("duplicate")
            def exp2(config: ExperimentConfig) -> None:
                pass

    def test_get_experiment(self):
        """Test retrieving registered experiment."""

        @register_experiment("retrieve_test")
        def exp(config: ExperimentConfig) -> None:
            pass

        retrieved = get_experiment("retrieve_test")
        assert retrieved == exp

        with pytest.raises(KeyError, match="not found"):
            get_experiment("nonexistent")

    def test_list_registered_experiments(self):
        """Test listing registered experiments."""
        assert list_registered_experiments() == []

        @register_experiment("exp_b")
        def exp_b(config: ExperimentConfig) -> None:
            pass

        @register_experiment("exp_a")
        def exp_a(config: ExperimentConfig) -> None:
            pass

        # Should be sorted
        assert list_registered_experiments() == ["exp_a", "exp_b"]


class TestExperimentRunner:
    """Test experiment running functionality."""

    def setup_method(self):
        """Setup for tests."""
        clear_registry()

    def teardown_method(self):
        """Cleanup after tests."""
        clear_registry()

    def test_run_experiment_from_dict(self):
        """Test running experiment from dictionary config."""
        # Register a simple test experiment
        @register_experiment("test_runner")
        def test_experiment(config: ExperimentConfig) -> None:
            # Just check that config was passed correctly
            assert config.name == "test_runner"
            assert config.seed == 999
            assert config.m_train == 10

        config_dict = {
            "experiment": {"name": "test_runner", "seed": 999},
            "training": {"n_epochs": 10, "batch_size": 5},
            "dataset": {
                "m_train": 10,
                "m_test": 5,
                "n_paths_train": 10,
                "n_paths_test": 20,
            },
        }

        # This should run without error
        run_experiment_from_dict(config_dict)

    def test_validate_config_valid(self):
        """Test validating a valid configuration."""
        # Register a test experiment
        @register_experiment("validation_test")
        def exp(config: ExperimentConfig) -> None:
            pass

        # Create a valid config file
        toml_content = """
[experiment]
name = "validation_test"
seed = 42

[training]
n_epochs = 10

[dataset]
m_train = 10
m_test = 5
n_paths_train = 10
n_paths_test = 20
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            assert validate_config(temp_path) is True
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_validate_config_invalid_experiment(self):
        """Test validating config with unregistered experiment."""
        # Create config with non-existent experiment
        toml_content = """
[experiment]
name = "nonexistent"
seed = 42

[training]
n_epochs = 10

[dataset]
m_train = 10
m_test = 5
n_paths_train = 10
n_paths_test = 20
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            assert validate_config(temp_path) is False
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestConfigFileIntegration:
    """Test integration with actual config files."""

    def test_load_digital_config(self):
        """Test loading the digital_default.toml config."""
        config_path = Path("configs/digital_default.toml")
        if not config_path.exists():
            pytest.skip("Config file not found")

        config = load_experiment_config(str(config_path))
        assert config.name == "digital"
        assert config.K == 1.0
        assert config.training.n_epochs == 2000

    def test_load_barrier_config(self):
        """Test loading the barrier_default.toml config."""
        config_path = Path("configs/barrier_default.toml")
        if not config_path.exists():
            pytest.skip("Config file not found")

        config = load_experiment_config(str(config_path))
        assert config.name == "barrier"
        assert config.K == 1.0
        assert config.B == 0.85

    def test_load_basket_config(self):
        """Test loading the basket_digital_default.toml config."""
        config_path = Path("configs/basket_digital_default.toml")
        if not config_path.exists():
            pytest.skip("Config file not found")

        config = load_experiment_config(str(config_path))
        assert config.name == "basket"
        assert config.d == 20
        assert config.m_train == 2048  # More samples for high-dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
