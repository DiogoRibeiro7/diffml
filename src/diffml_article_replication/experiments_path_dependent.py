"""Path-dependent option experiments for article replication."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from diffml.config import BSParams, TrainingConfig, get_device, set_default_dtype
from diffml.datasets_path_dependent import (
    make_arithmetic_asian_call_dataset,
    make_lookback_call_dataset,
)
from diffml.networks import PricingNet
from diffml.training import Mode, nn_value_delta_gamma, rmse, train_model

DatasetBuilder = Callable[..., tuple[Tensor, Tensor, Tensor, Tensor]]


def _compute_rmse(pred: Tensor, target: Tensor) -> float:
    """Return root-mean-squared error between two tensors."""
    return rmse(pred, target)


def _print_rmse_table(title: str, rows: Sequence[tuple[str, float, float]]) -> None:
    """Print a paper-style RMSE table."""
    print(f"\n{title}")
    header = f"{'model':<20} {'price_rmse':>12} {'delta_rmse':>12}"
    print(header)
    print("-" * len(header))
    for model_name, price_rmse, delta_rmse in rows:
        print(f"{model_name:<20} {price_rmse:12.6f} {delta_rmse:12.6f}")


def _train_and_evaluate_variant(
    *,
    model_name: str,
    mode: Mode,
    lambda_delta: float,
    dataset: TensorDataset,
    base_config: TrainingConfig,
    x_test: Tensor,
    price_ref: Tensor,
    delta_ref: Tensor,
    device: torch.device,
) -> tuple[str, float, float]:
    """Train the requested model variant and return its RMSE metrics."""
    config = replace(base_config, lambda_delta=lambda_delta)
    model = PricingNet(input_dim=x_test.shape[1], hidden_dim=32, n_hidden=3)
    train_model(model, dataset, config, mode=mode, device=device)
    model.eval()
    with torch.no_grad():
        pred_price, pred_delta, _ = nn_value_delta_gamma(
            model,
            x_test.to(device),
            compute_delta=True,
            compute_gamma=False,
        )
    if pred_delta is None:
        raise RuntimeError("Expected delta predictions for path-dependent experiment.")

    price_rmse = _compute_rmse(pred_price.cpu(), price_ref.cpu())
    delta_rmse = _compute_rmse(pred_delta.cpu(), delta_ref.cpu())
    return model_name, price_rmse, delta_rmse


def _run_path_dependent_experiment(
    *,
    title: str,
    dataset_fn: DatasetBuilder,
    train_kwargs: dict[str, object],
    test_kwargs: dict[str, object],
    training_config: TrainingConfig,
    lambda_pathwise: float,
    lambda_lrm: float,
) -> None:
    """Shared orchestration for path-dependent experiments."""
    set_default_dtype()
    device = get_device()

    x_train, price_train, delta_pw_train, delta_lrm_train = dataset_fn(**train_kwargs)
    dataset = TensorDataset(x_train, price_train, delta_pw_train, delta_lrm_train)

    x_test, price_ref, _delta_pw_ref, delta_lrm_ref = dataset_fn(**test_kwargs)
    delta_reference = delta_lrm_ref

    rows: list[tuple[str, float, float]] = []
    rows.append(
        _train_and_evaluate_variant(
            model_name="standard",
            mode="standard",
            lambda_delta=0.0,
            dataset=dataset,
            base_config=training_config,
            x_test=x_test,
            price_ref=price_ref,
            delta_ref=delta_reference,
            device=device,
        )
    )

    rows.append(
        _train_and_evaluate_variant(
            model_name="dml_pathwise",
            mode="delta_pathwise",
            lambda_delta=lambda_pathwise,
            dataset=dataset,
            base_config=training_config,
            x_test=x_test,
            price_ref=price_ref,
            delta_ref=delta_reference,
            device=device,
        )
    )

    rows.append(
        _train_and_evaluate_variant(
            model_name="dml_lrm",
            mode="delta_lrm",
            lambda_delta=lambda_lrm,
            dataset=dataset,
            base_config=training_config,
            x_test=x_test,
            price_ref=price_ref,
            delta_ref=delta_reference,
            device=device,
        )
    )

    _print_rmse_table(title, rows)


def run_arithmetic_asian_experiment(
    *,
    K: float = 1.0,
    params: BSParams | None = None,
    m_train: int = 512,
    m_test: int = 200,
    n_steps: int = 16,
    n_paths_train: int = 10,
    n_paths_test: int = 2000,
    x_min: float = 0.5,
    x_max: float = 1.5,
    train_seed: int = 1234,
    test_seed: int = 5678,
    training_config: TrainingConfig | None = None,
    lambda_pathwise: float = 1.0,
    lambda_lrm: float = 1.0,
) -> None:
    """Run the arithmetic Asian call pricing experiment."""
    bs_params = params or BSParams(r=0.0, sigma=0.20, T=1.0 / 3.0)
    base_config = training_config or TrainingConfig(
        n_epochs=2000,
        batch_size=256,
        lr_initial=1e-3,
        lr_min=1e-6,
        lambda_delta=1.0,
        lambda_gamma=0.0,
    )

    train_kwargs = {
        "m": m_train,
        "K": K,
        "params": bs_params,
        "n_steps": n_steps,
        "x_min": x_min,
        "x_max": x_max,
        "n_paths_per_x": n_paths_train,
        "seed": train_seed,
    }
    test_kwargs = {
        "m": m_test,
        "K": K,
        "params": bs_params,
        "n_steps": n_steps,
        "x_min": x_min,
        "x_max": x_max,
        "n_paths_per_x": n_paths_test,
        "seed": test_seed,
    }

    _run_path_dependent_experiment(
        title="Arithmetic Asian Call Experiment",
        dataset_fn=make_arithmetic_asian_call_dataset,
        train_kwargs=train_kwargs,
        test_kwargs=test_kwargs,
        training_config=base_config,
        lambda_pathwise=lambda_pathwise,
        lambda_lrm=lambda_lrm,
    )


def run_lookback_call_experiment(
    *,
    K: float = 1.0,
    params: BSParams | None = None,
    m_train: int = 512,
    m_test: int = 200,
    n_steps: int = 32,
    n_paths_train: int = 10,
    n_paths_test: int = 3000,
    x_min: float = 0.5,
    x_max: float = 1.5,
    train_seed: int = 2468,
    test_seed: int = 9753,
    training_config: TrainingConfig | None = None,
    lambda_pathwise: float = 1.0,
    lambda_lrm: float = 1.0,
) -> None:
    """Run the fixed-strike lookback call pricing experiment."""
    bs_params = params or BSParams(r=0.0, sigma=0.20, T=1.0 / 3.0)
    base_config = training_config or TrainingConfig(
        n_epochs=2000,
        batch_size=256,
        lr_initial=1e-3,
        lr_min=1e-6,
        lambda_delta=1.0,
        lambda_gamma=0.0,
    )

    train_kwargs = {
        "m": m_train,
        "K": K,
        "params": bs_params,
        "n_steps": n_steps,
        "x_min": x_min,
        "x_max": x_max,
        "n_paths_per_x": n_paths_train,
        "seed": train_seed,
    }
    test_kwargs = {
        "m": m_test,
        "K": K,
        "params": bs_params,
        "n_steps": n_steps,
        "x_min": x_min,
        "x_max": x_max,
        "n_paths_per_x": n_paths_test,
        "seed": test_seed,
    }

    _run_path_dependent_experiment(
        title="Lookback Call Experiment",
        dataset_fn=make_lookback_call_dataset,
        train_kwargs=train_kwargs,
        test_kwargs=test_kwargs,
        training_config=base_config,
        lambda_pathwise=lambda_pathwise,
        lambda_lrm=lambda_lrm,
    )


__all__ = ["run_arithmetic_asian_experiment", "run_lookback_call_experiment"]
