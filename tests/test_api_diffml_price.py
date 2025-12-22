"""Tests for the high-level diffml_article_replication API."""

from __future__ import annotations

import torch
from torch import Tensor

from diffml.config import TrainingConfig
from diffml_article_replication.api import diffml_price
from diffml_article_replication.simulator_api import SimulationResult


class _DeterministicDigitalSimulator:
    """Toy simulator returning analytic price/delta labels."""

    default_n_paths = 4

    def __init__(self, strike: float = 0.0) -> None:
        self.strike = strike

    def simulate(self, x: Tensor, n_paths: int, seed: int | None = None) -> SimulationResult:
        with torch.no_grad():
            features = x.to(dtype=torch.float64)
            logits = (features - self.strike) * 5.0
            probabilities = torch.sigmoid(logits)
            payoffs = probabilities.repeat(1, n_paths)
            lrm = (probabilities * (1 - probabilities) * 5.0).repeat(1, n_paths)

        return SimulationResult(
            paths=features.repeat(1, n_paths, 1),
            payoffs=payoffs,
            pathwise_deltas=torch.zeros_like(payoffs),
            lrm_deltas=lrm,
        )


def _tiny_training_config() -> TrainingConfig:
    return TrainingConfig(
        n_epochs=8,
        batch_size=8,
        lr_initial=5e-3,
        lr_min=5e-4,
    )


def test_diffml_price_standard_shapes() -> None:
    simulator = _DeterministicDigitalSimulator()
    x_train = torch.linspace(-1.0, 1.0, 16).reshape(-1, 1)
    x_test = torch.linspace(-0.5, 0.5, 4).reshape(-1, 1)

    prices, deltas, gamma = diffml_price(
        simulator=simulator,
        x_train=x_train,
        x_test=x_test,
        mode="standard",
        training_config=_tiny_training_config(),
        seed=1,
    )

    assert prices.shape == (x_test.shape[0], 1)
    assert deltas.shape == x_test.shape
    assert gamma is None
    assert torch.isfinite(prices).all()
    assert torch.isfinite(deltas).all()


def test_diffml_price_delta_lrm_runs() -> None:
    simulator = _DeterministicDigitalSimulator()
    x_train = torch.linspace(-1.0, 1.0, 32).reshape(-1, 1)
    x_test = torch.tensor([[-0.2], [0.0], [0.3]])

    prices, deltas, gamma = diffml_price(
        simulator=simulator,
        x_train=x_train,
        x_test=x_test,
        mode="delta_lrm",
        training_config=_tiny_training_config(),
        lambda_delta=0.5,
        seed=2,
    )

    assert prices.shape == (x_test.shape[0], 1)
    assert deltas.shape == x_test.shape
    assert gamma is None
    assert torch.isfinite(prices).all()
    assert torch.isfinite(deltas).all()
