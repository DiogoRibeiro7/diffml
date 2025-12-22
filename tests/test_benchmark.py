"""Tests for benchmark and sensitivity analysis utilities."""

from __future__ import annotations

from diffml_article_replication.benchmark import (
    _DigitalBenchmarkConfig,
    run_digital_benchmark,
    run_lambda_delta_sweep,
)

from diffml.config import TrainingConfig


class TestBenchmarkUtilities:
    """Ensure benchmark helpers execute with tiny configurations."""

    tiny_config = _DigitalBenchmarkConfig(
        m_train=32,
        m_test=32,
        n_paths_train=100,
        n_paths_test=200,
        training=TrainingConfig(
            n_epochs=10,
            batch_size=32,
            lr_initial=1e-3,
            lr_min=1e-4,
            lambda_delta=0.0,
            lambda_gamma=0.0,
        ),
    )

    def test_run_digital_benchmark_baseline_only(self) -> None:
        """Benchmark should run and produce metrics for requested modes."""
        results = run_digital_benchmark(
            seeds=[0],
            model_modes=["baseline_poly"],
            config=self.tiny_config,
        )
        assert "baseline_poly" in results
        assert results["baseline_poly"]["price_rmse_mean"] >= 0.0

    def test_lambda_delta_sweep(self) -> None:
        """Lambda delta sweep returns entries for each requested value."""
        lambda_values = [0.0, 0.5]
        results = run_lambda_delta_sweep(
            lambda_values=lambda_values,
            seed=0,
            config=self.tiny_config,
        )
        assert set(results) == set(lambda_values)
        for metrics in results.values():
            assert "price_rmse" in metrics
            assert "delta_rmse" in metrics
