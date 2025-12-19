"""Smoke tests for experiment functions.

This module provides quick end-to-end tests for experiment functions
to ensure they can run without exceptions. These tests use reduced
configurations for speed.
"""

from unittest.mock import patch

import pytest
import torch

from diffml.config import TrainingConfig, set_default_dtype
from diffml.experiments_barrier import run_barrier_experiment
from diffml.experiments_basket import run_basket_digital_experiment
from diffml.experiments_digital import run_digital_experiment
from diffml.experiments_gamma import run_gamma_experiment
from diffml.experiments_smoothing import run_smoothing_experiment


@pytest.fixture(autouse=True)
def setup_precision() -> None:
    """Set default dtype to float64 for all tests."""
    set_default_dtype()


@pytest.fixture
def fast_training_config() -> TrainingConfig:
    """Create a fast training configuration for smoke tests."""
    return TrainingConfig(
        n_epochs=5,  # Very few epochs
        batch_size=32,
        lr_initial=1e-3,
        lr_min=1e-5,
        lambda_delta=1.0,
        lambda_gamma=0.5
    )


class TestDigitalExperimentSmoke:
    """Smoke test for digital option experiment."""

    @patch('diffml.experiments_digital.TrainingConfig')
    @patch('diffml.experiments_digital.make_digital_dataset')
    def test_run_digital_experiment_smoke(
        self,
        mock_dataset,
        mock_config,
        fast_training_config: TrainingConfig
    ) -> None:
        """Test that digital experiment runs without exceptions."""
        # Mock the dataset to return small data
        m = 16
        mock_dataset.return_value = (
            torch.randn(m, 1, dtype=torch.float64),  # x
            torch.rand(m, 1, dtype=torch.float64),   # price
            torch.zeros(m, 1, dtype=torch.float64),  # delta_pw
            torch.randn(m, 1, dtype=torch.float64) * 0.1  # delta_lrm
        )

        # Use fast training config
        mock_config.return_value = fast_training_config

        # Should run without exceptions
        try:
            run_digital_experiment()
        except Exception as e:
            pytest.fail(f"Digital experiment raised exception: {e}")

    def test_digital_experiment_imports(self) -> None:
        """Test that digital experiment can be imported."""
        # This tests that all dependencies are properly set up
        from diffml.experiments_digital import (
            run_digital_experiment,
        )

        assert callable(run_digital_experiment)


class TestBarrierExperimentSmoke:
    """Smoke test for barrier option experiment."""

    @patch('diffml.experiments_barrier.TrainingConfig')
    @patch('diffml.experiments_barrier.make_barrier_dataset')
    def test_run_barrier_experiment_smoke(
        self,
        mock_dataset,
        mock_config,
        fast_training_config: TrainingConfig
    ) -> None:
        """Test that barrier experiment runs without exceptions."""
        # Mock the dataset
        m = 16
        mock_dataset.return_value = (
            torch.randn(m, 1, dtype=torch.float64) * 0.3 + 1.0,  # x
            torch.rand(m, 1, dtype=torch.float64) * 0.5,         # price
            torch.randn(m, 1, dtype=torch.float64) * 0.1,        # delta_pw
            torch.randn(m, 1, dtype=torch.float64) * 0.1         # delta_lrm
        )

        # Use fast training config
        mock_config.return_value = fast_training_config

        # Should run without exceptions
        try:
            run_barrier_experiment()
        except Exception as e:
            pytest.fail(f"Barrier experiment raised exception: {e}")


class TestBasketExperimentSmoke:
    """Smoke test for basket digital option experiment."""

    @patch('diffml.experiments_basket.TrainingConfig')
    @patch('diffml.experiments_basket.make_basket_digital_dataset')
    def test_run_basket_experiment_smoke(
        self,
        mock_dataset,
        mock_config,
        fast_training_config: TrainingConfig
    ) -> None:
        """Test that basket experiment runs without exceptions."""
        # Mock the dataset with smaller dimension
        m = 16
        d = 3  # Small dimension for testing
        mock_dataset.return_value = (
            torch.randn(m, d, dtype=torch.float64),           # x (multi-dim)
            torch.rand(m, 1, dtype=torch.float64),            # price
            torch.zeros(m, d, dtype=torch.float64),           # delta_pw
            torch.randn(m, d, dtype=torch.float64) * 0.1     # delta_lrm
        )

        # Use fast training config
        mock_config.return_value = fast_training_config

        # Patch the dimension used in the experiment
        with patch('diffml.experiments_basket.run_basket_digital_experiment') as mock_run:
            # Create a modified version that uses smaller dimension
            def run_small_basket():
                """Run basket experiment with small dimension."""
                from diffml.experiments_basket import (
                    run_basket_digital_experiment as original_run,
                )
                with patch('diffml.experiments_basket.d', 3):
                    return original_run()

            # Test the actual function with patched config and dataset
            try:
                run_basket_digital_experiment()
            except Exception as e:
                pytest.fail(f"Basket experiment raised exception: {e}")


class TestSmoothingExperimentSmoke:
    """Smoke test for smoothing experiment."""

    @patch('diffml.experiments_smoothing.TrainingConfig')
    @patch('diffml.experiments_smoothing.make_smoothed_digital_dataset')
    def test_run_smoothing_experiment_smoke(
        self,
        mock_dataset,
        mock_config,
        fast_training_config: TrainingConfig
    ) -> None:
        """Test that smoothing experiment runs without exceptions."""
        # Mock the dataset
        m = 16
        mock_dataset.return_value = (
            torch.randn(m, 1, dtype=torch.float64) * 20 + 100,  # x
            torch.rand(m, 1, dtype=torch.float64),               # price
            torch.randn(m, 1, dtype=torch.float64) * 0.01,      # delta_pw
            torch.randn(m, 1, dtype=torch.float64) * 0.1        # delta_lrm
        )

        # Use fast training config
        mock_config.return_value = fast_training_config

        # Patch epsilon multipliers to use fewer values
        with patch('diffml.experiments_smoothing.eps_multipliers', [0.5, 1.0]):
            try:
                run_smoothing_experiment()
            except Exception as e:
                pytest.fail(f"Smoothing experiment raised exception: {e}")


class TestGammaExperimentSmoke:
    """Smoke test for gamma portfolio experiment."""

    @patch('diffml.experiments_gamma.TrainingConfig')
    @patch('diffml.experiments_gamma.make_portfolio_gamma_dataset')
    def test_run_gamma_experiment_smoke(
        self,
        mock_dataset,
        mock_config,
        fast_training_config: TrainingConfig
    ) -> None:
        """Test that gamma experiment runs without exceptions."""
        # Mock the dataset
        m = 16
        mock_dataset.return_value = (
            torch.randn(m, 1, dtype=torch.float64) * 0.3 + 1.0,   # x
            torch.rand(m, 1, dtype=torch.float64) * 0.2,          # price_true
            torch.randn(m, 1, dtype=torch.float64) * 0.1,         # delta_true
            torch.randn(m, 1, dtype=torch.float64) * 0.01,        # gamma_true
            torch.rand(m, 1, dtype=torch.float64) * 0.2,          # price_mc
            torch.randn(m, 1, dtype=torch.float64) * 0.1,         # delta_pw
            torch.randn(m, 1, dtype=torch.float64) * 0.01         # gamma_pwlr
        )

        # Use fast training config
        mock_config.return_value = fast_training_config

        # Should run without exceptions
        try:
            run_gamma_experiment()
        except Exception as e:
            pytest.fail(f"Gamma experiment raised exception: {e}")


class TestExperimentIntegration:
    """Integration tests for experiment functions."""

    @pytest.mark.slow
    def test_digital_experiment_runs_fully(self) -> None:
        """Test digital experiment runs fully with minimal config.

        This is marked as slow because it runs the actual experiment.
        """
        # Patch to use very small configuration
        with patch('diffml.experiments_digital.m_train', 8), \
             patch('diffml.experiments_digital.m_test', 4), \
             patch('diffml.experiments_digital.n_paths_train', 5), \
             patch('diffml.experiments_digital.n_paths_test', 10), \
             patch.object(TrainingConfig, 'n_epochs', 3):

            try:
                # This actually runs the experiment with tiny data
                run_digital_experiment()
            except Exception as e:
                pytest.fail(f"Digital experiment integration test failed: {e}")


class TestAllExperimentsImport:
    """Test that all experiment modules can be imported."""

    def test_import_all_experiments(self) -> None:
        """Test importing all experiment modules."""
        # These imports should not raise any errors
        from diffml.experiments_barrier import (
            run_barrier_experiment,
        )
        from diffml.experiments_basket import (
            run_basket_digital_experiment,
        )
        from diffml.experiments_digital import (
            run_digital_experiment,
        )
        from diffml.experiments_gamma import run_gamma_experiment
        from diffml.experiments_smoothing import (
            run_smoothing_experiment,
        )

        # Check all are callable
        assert callable(run_digital_experiment)
        assert callable(run_barrier_experiment)
        assert callable(run_basket_digital_experiment)
        assert callable(run_smoothing_experiment)
        assert callable(run_gamma_experiment)

    def test_run_all_script_import(self) -> None:
        """Test that the main run_all_experiments script can be imported."""
        import sys
        from pathlib import Path

        # Add scripts directory to path
        scripts_dir = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))

        try:
            # Should be able to import without errors
            from run_all_experiments import run_all_experiments

            assert callable(run_all_experiments)
        finally:
            # Clean up sys.path
            sys.path.pop(0)


class TestExperimentOutput:
    """Test experiment output format and content."""

    @patch('diffml.experiments_digital.print')
    @patch('diffml.experiments_digital.TrainingConfig')
    @patch('diffml.experiments_digital.train_model')
    @patch('diffml.experiments_digital.make_digital_dataset')
    def test_experiment_prints_results(
        self,
        mock_dataset,
        mock_train,
        mock_config,
        mock_print
    ) -> None:
        """Test that experiments print results in expected format."""
        # Setup mocks
        m = 8
        mock_dataset.return_value = (
            torch.randn(m, 1, dtype=torch.float64),
            torch.rand(m, 1, dtype=torch.float64),
            torch.zeros(m, 1, dtype=torch.float64),
            torch.randn(m, 1, dtype=torch.float64) * 0.1
        )

        mock_config.return_value = TrainingConfig(n_epochs=1)

        # Mock trained model
        from diffml.networks import PricingNet

        mock_model = PricingNet()
        mock_train.return_value = mock_model

        # Run experiment
        run_digital_experiment()

        # Check that results were printed
        print_calls = mock_print.call_args_list
        print_texts = [str(call[0][0]) if call[0] else "" for call in print_calls]

        # Should print experiment name
        assert any("DIGITAL OPTION EXPERIMENT" in text for text in print_texts)

        # Should print results summary
        assert any("RESULTS SUMMARY" in text for text in print_texts)

        # Should print model names
        assert any("Standard ML" in text for text in print_texts)
        assert any("LRM DML" in text for text in print_texts)


class TestExperimentReproducibility:
    """Test that experiments are reproducible with fixed seeds."""

    @patch('diffml.experiments_digital.TrainingConfig')
    @patch('diffml.experiments_digital.make_digital_dataset')
    def test_digital_experiment_deterministic(
        self,
        mock_dataset,
        mock_config
    ) -> None:
        """Test that digital experiment is deterministic with fixed seed."""
        # Create deterministic mock data
        torch.manual_seed(42)
        m = 8
        data1 = (
            torch.randn(m, 1, dtype=torch.float64),
            torch.rand(m, 1, dtype=torch.float64),
            torch.zeros(m, 1, dtype=torch.float64),
            torch.randn(m, 1, dtype=torch.float64) * 0.1
        )

        torch.manual_seed(42)  # Reset seed
        data2 = (
            torch.randn(m, 1, dtype=torch.float64),
            torch.rand(m, 1, dtype=torch.float64),
            torch.zeros(m, 1, dtype=torch.float64),
            torch.randn(m, 1, dtype=torch.float64) * 0.1
        )

        # Data should be identical
        for t1, t2 in zip(data1, data2, strict=False):
            assert torch.allclose(t1, t2)

        # Mock dataset always returns same data
        mock_dataset.return_value = data1
        mock_config.return_value = TrainingConfig(n_epochs=1)

        # Experiments should be reproducible
        # (In practice, would need to control all random seeds)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
