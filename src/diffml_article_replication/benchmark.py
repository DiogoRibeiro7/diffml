"""Benchmark and sensitivity analysis helpers for the digital experiment."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from diffml.bs_analytics import bs_digital_delta, bs_digital_price
from diffml.config import BSParams, TrainingConfig, get_device, set_default_dtype
from diffml.datasets_digital import make_digital_dataset
from diffml.networks import PricingNet
from diffml.training import nn_value_delta_gamma, rmse, train_model

from .baselines import BaselineModel, MLPBaseline, PolynomialRegressionBaseline


@dataclass(frozen=True)
class _DigitalBenchmarkConfig:
    m_train: int = 256
    m_test: int = 256
    n_paths_train: int = 2000
    n_paths_test: int = 5000
    x_min: float = 0.5
    x_max: float = 1.5
    strike: float = 1.0
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            n_epochs=300,
            batch_size=128,
            lr_initial=1e-3,
            lr_min=1e-6,
            lambda_delta=1.0,
            lambda_gamma=0.0,
        )
    )


def run_digital_benchmark(
    seeds: list[int],
    model_modes: list[str],
    *,
    config: _DigitalBenchmarkConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Run the digital experiment for multiple seeds and model modes."""
    if not seeds:
        raise ValueError("seeds list must not be empty")
    if not model_modes:
        raise ValueError("model_modes list must not be empty")

    params = config or _DigitalBenchmarkConfig()
    set_default_dtype()
    device = get_device("cpu")

    metrics: dict[str, dict[str, list[float]]] = {
        mode: {"price_rmse": [], "delta_rmse": []} for mode in model_modes
    }

    for seed in seeds:
        torch.manual_seed(seed)
        data = _generate_digital_data(seed, params)

        for mode in model_modes:
            price_rmse_value, delta_rmse_value = _run_single_model(mode, params, data, device)
            metrics[mode]["price_rmse"].append(price_rmse_value)
            metrics[mode]["delta_rmse"].append(delta_rmse_value)

    return {
        mode: {
            "price_rmse_mean": torch.tensor(values["price_rmse"]).mean().item(),
            "price_rmse_std": torch.tensor(values["price_rmse"]).std(unbiased=False).item(),
            "delta_rmse_mean": torch.tensor(values["delta_rmse"]).mean().item(),
            "delta_rmse_std": torch.tensor(values["delta_rmse"]).std(unbiased=False).item(),
        }
        for mode, values in metrics.items()
    }


def format_benchmark_table(results: dict[str, dict[str, float]]) -> str:
    """Return a formatted string summarising benchmark results."""
    header = "Model".ljust(20) + "Price RMSE (mean±std)".ljust(30) + "Delta RMSE (mean±std)"
    lines = [header, "-" * len(header)]
    for name, stats in results.items():
        price = f"{stats['price_rmse_mean']:.4f} ± {stats['price_rmse_std']:.4f}"
        delta = f"{stats['delta_rmse_mean']:.4f} ± {stats['delta_rmse_std']:.4f}"
        lines.append(f"{name.ljust(20)}{price.ljust(30)}{delta}")
    return "\n".join(lines)


def run_lambda_delta_sweep(
    lambda_values: list[float],
    *,
    seed: int = 0,
    config: _DigitalBenchmarkConfig | None = None,
) -> dict[float, dict[str, float]]:
    """Sweep ``lambda_delta`` for the digital experiment."""
    if not lambda_values:
        raise ValueError("lambda_values must contain at least one entry")

    params = config or _DigitalBenchmarkConfig()
    set_default_dtype()
    torch.manual_seed(seed)
    data = _generate_digital_data(seed, params)

    results: dict[float, dict[str, float]] = {}
    for lambda_delta in lambda_values:
        training_cfg = replace(params.training, lambda_delta=lambda_delta)
        metrics = _train_eval_neural_model(
            mode="delta_lrm",
            training_cfg=training_cfg,
            data=data,
            device=get_device("cpu"),
        )
        results[lambda_delta] = {
            "price_rmse": metrics[0],
            "delta_rmse": metrics[1],
        }
    return results


def _generate_digital_data(seed: int, params: _DigitalBenchmarkConfig):
    bs_params = BSParams(r=0.0, sigma=0.2, T=1.0 / 3.0)
    x_train, price_train, delta_pw_train, delta_lrm_train = make_digital_dataset(
        m=params.m_train,
        K=params.strike,
        params=bs_params,
        x_min=params.x_min * params.strike,
        x_max=params.x_max * params.strike,
        n_paths_per_x=params.n_paths_train,
        seed=seed,
    )
    x_test = torch.linspace(
        params.x_min * params.strike,
        params.x_max * params.strike,
        params.m_test,
        dtype=torch.float64,
    ).reshape(-1, 1)
    price_true = bs_digital_price(x_test, params.strike, bs_params)
    delta_true = bs_digital_delta(x_test, params.strike, bs_params)
    return {
        "train": (x_train, price_train, delta_pw_train, delta_lrm_train),
        "test": (x_test, price_true, delta_true),
    }


def _run_single_model(
    mode: str,
    params: _DigitalBenchmarkConfig,
    data: dict[str, tuple[Tensor, ...]],
    device: torch.device,
) -> tuple[float, float]:
    if mode in {"standard", "delta_pathwise", "delta_lrm"}:
        return _train_eval_neural_model(mode, params.training, data, device)
    if mode == "baseline_poly":
        return _evaluate_baseline(PolynomialRegressionBaseline(degree=3), data)
    if mode == "baseline_mlp":
        return _evaluate_baseline(MLPBaseline(n_epochs=100), data)
    raise ValueError(f"Unknown model mode: {mode}")


def _train_eval_neural_model(
    mode: Literal["standard", "delta_pathwise", "delta_lrm"],
    training_cfg: TrainingConfig,
    data: dict[str, tuple[Tensor, ...]],
    device: torch.device,
) -> tuple[float, float]:
    x_train, price_train, delta_pw_train, delta_lrm_train = data["train"]
    dataset = TensorDataset(x_train, price_train, delta_pw_train, delta_lrm_train)
    model = PricingNet(input_dim=1, hidden_dim=32, n_hidden=3)
    train_model(model, dataset, training_cfg, mode=mode, device=device)
    x_test, price_true, delta_true = data["test"]
    model.eval()
    with torch.no_grad():
        preds_price, preds_delta, _ = nn_value_delta_gamma(
            model,
            x_test.to(device),
            compute_delta=True,
            compute_gamma=False,
            device=device,
        )
    if preds_delta is None:
        raise RuntimeError("Expected delta predictions for neural model")
    price_rmse_value = rmse(preds_price.cpu(), price_true)
    delta_rmse_value = rmse(preds_delta.cpu(), delta_true)
    return float(price_rmse_value), float(delta_rmse_value)


def _evaluate_baseline(
    baseline: BaselineModel,
    data: dict[str, tuple[Tensor, ...]],
) -> tuple[float, float]:
    x_train, price_train, _, _ = data["train"]
    baseline.fit(x_train, price_train)
    x_test, price_true, delta_true = data["test"]
    x_test_var = x_test.clone().requires_grad_(True)
    preds = baseline.predict(x_test_var)
    grads = torch.autograd.grad(preds.sum(), x_test_var, create_graph=False)[0]
    price_rmse_value = rmse(preds.detach(), price_true)
    delta_rmse_value = rmse(grads.detach(), delta_true)
    return float(price_rmse_value), float(delta_rmse_value)


__all__ = [
    "run_digital_benchmark",
    "format_benchmark_table",
    "run_lambda_delta_sweep",
]
