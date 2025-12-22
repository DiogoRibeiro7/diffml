"""Tests for performance-oriented utilities and experiments."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from diffml.config import BSParams, TrainingConfig, set_default_dtype
from diffml.datasets_digital import make_digital_dataset
from diffml.experiments_basket import run_basket_high_dim_experiment
from diffml.networks import PricingNet
from diffml.simulation import simulate_bs_terminal_shared
from diffml.training import maybe_mixed_precision, train_model


def _tiny_digital_dataset(
    m: int = 16,
    n_paths: int = 4,
    seed: int = 321,
) -> TensorDataset:
    set_default_dtype()
    params = BSParams(r=0.05, sigma=0.2, T=0.25)
    x, price, delta_pw, delta_lrm = make_digital_dataset(
        m=m,
        K=1.0,
        params=params,
        x_min=0.8,
        x_max=1.2,
        n_paths_per_x=n_paths,
        seed=seed,
        use_shared_paths=True,
    )
    return TensorDataset(x, price, delta_pw, delta_lrm)


def test_simulate_bs_terminal_shared_shapes(disable_cuda: None) -> None:
    """Shared-path simulator should reuse shocks while preserving vectorization."""
    set_default_dtype()
    params = BSParams(r=0.01, sigma=0.2, T=0.25)
    spots = torch.linspace(80.0, 120.0, 4, dtype=torch.float64).reshape(-1, 1)
    ST, xi = simulate_bs_terminal_shared(spots, params, n_paths=6, seed=123)
    assert ST.shape == (spots.shape[0], 6)
    assert xi.shape == (6,)
    scaling = ST / spots
    assert torch.allclose(scaling[0], scaling[1])


def test_run_basket_high_dim_experiment_smoke(disable_cuda: None, capsys: pytest.CaptureFixture[str]) -> None:
    """High-dimensional experiment should complete for a tiny configuration."""
    config = TrainingConfig(
        n_epochs=2,
        batch_size=32,
        lr_initial=1e-3,
        lr_min=1e-4,
        lambda_delta=1.0,
    )
    run_basket_high_dim_experiment(
        dims=[5],
        training_config=config,
        m_train=64,
        m_test=32,
        n_paths_train=4,
        n_paths_test=8,
    )
    captured = capsys.readouterr().out
    assert "High-dimensional basket" in captured
    assert "dimension" in captured


def test_mixed_precision_noop_cpu(disable_cuda: None) -> None:
    """Mixed precision helper should be a no-op on CPU and training should succeed."""
    dataset = _tiny_digital_dataset()
    model = PricingNet(input_dim=1, hidden_dim=10, n_hidden=2)
    config = TrainingConfig(
        n_epochs=2,
        batch_size=8,
        lr_initial=1e-3,
        lr_min=1e-4,
        lambda_delta=1.0,
        use_mixed_precision=True,
    )
    train_model(model, dataset, config, mode="delta_lrm", device=torch.device("cpu"))
    tensor = torch.tensor([1.0])
    with maybe_mixed_precision(torch.device("cpu")):
        assert torch.allclose(tensor, tensor)


@pytest.mark.gpu
def test_mixed_precision_training_cuda() -> None:
    """If CUDA is available, ensure mixed precision training runs on a tiny dataset."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    dataset = _tiny_digital_dataset()
    model = PricingNet(input_dim=1, hidden_dim=10, n_hidden=2)
    device = torch.device("cuda")
    config = TrainingConfig(
        n_epochs=2,
        batch_size=8,
        lr_initial=1e-3,
        lr_min=1e-4,
        lambda_delta=1.0,
        use_mixed_precision=True,
    )
    train_model(model.to(device), dataset, config, mode="delta_lrm", device=device)
