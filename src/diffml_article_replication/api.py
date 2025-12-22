"""High-level helpers for training and pricing via the simulator interface."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from diffml.config import DEFAULT_DTYPE, TrainingConfig, get_device
from diffml.networks import PricingNet
from diffml.training import nn_value_delta_gamma, train_model

from .simulator_api import InstrumentedSimulator

Mode = Literal["standard", "delta_pathwise", "delta_lrm", "gamma_pwlr"]


def diffml_price(
    simulator: InstrumentedSimulator,
    x_train: Tensor,
    x_test: Tensor,
    mode: Mode,
    training_config: TrainingConfig,
    lambda_delta: float = 0.0,
    lambda_gamma: float = 0.0,
    seed: int | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Train a neural network on simulator-generated labels and price ``x_test``."""
    device = get_device()
    dtype = DEFAULT_DTYPE
    x_train_device = x_train.to(device=device, dtype=dtype).detach().clone()
    x_test_device = x_test.to(device=device, dtype=dtype).detach().clone()

    if seed is not None:
        torch.manual_seed(seed)

    n_paths = _resolve_n_paths(simulator, x_train_device)
    with torch.no_grad():
        simulation = simulator.simulate(x_train_device, n_paths=n_paths, seed=seed)

    price_label = _aggregate_payoffs(simulation.payoffs).detach()
    delta_pw = _detach_optional(_aggregate_optional(simulation.pathwise_deltas))
    delta_lrm = _detach_optional(_aggregate_optional(simulation.lrm_deltas))
    gamma_pwlr = _detach_optional(_aggregate_optional(simulation.pwlr_gammas))

    _validate_labels(mode, delta_pw, delta_lrm, gamma_pwlr)

    delta_pw = delta_pw if delta_pw is not None else torch.zeros_like(price_label)
    delta_lrm = delta_lrm if delta_lrm is not None else torch.zeros_like(price_label)

    dataset = _build_dataset(mode, x_train_device, price_label, delta_pw, delta_lrm, gamma_pwlr)
    config = replace(training_config, lambda_delta=lambda_delta, lambda_gamma=lambda_gamma)
    model = PricingNet(input_dim=x_train_device.shape[1]).to(device=device, dtype=dtype)
    trained = train_model(model, dataset, config, mode=mode, device=device)

    compute_gamma = mode == "gamma_pwlr"
    values, deltas, gammas = nn_value_delta_gamma(
        trained,
        x_test_device,
        compute_delta=True,
        compute_gamma=compute_gamma,
    )
    if deltas is None:
        raise RuntimeError("Expected delta predictions from nn_value_delta_gamma")

    gamma_out = gammas if compute_gamma else None
    return values.detach().cpu(), deltas.detach().cpu(), None if gamma_out is None else gamma_out.detach().cpu()


def _resolve_n_paths(simulator: InstrumentedSimulator, x: Tensor) -> int:
    for attribute in ("n_paths", "n_paths_train", "default_n_paths", "paths_per_sample"):
        value = getattr(simulator, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    return max(1, x.shape[0])


def _aggregate_payoffs(payoffs: Tensor) -> Tensor:
    reduced = payoffs.mean(dim=1)
    if reduced.dim() == 1:
        reduced = reduced.unsqueeze(-1)
    return reduced


def _aggregate_optional(values: Tensor | None) -> Tensor | None:
    if values is None:
        return None
    reduced = values.mean(dim=1)
    if reduced.dim() == 1:
        reduced = reduced.unsqueeze(-1)
    return reduced


def _detach_optional(values: Tensor | None) -> Tensor | None:
    if values is None:
        return None
    return values.detach()


def _validate_labels(
    mode: Mode,
    delta_pw: Tensor | None,
    delta_lrm: Tensor | None,
    gamma_pwlr: Tensor | None,
) -> None:
    if mode == "delta_pathwise" and delta_pw is None:
        raise ValueError("delta_pathwise mode requires pathwise delta labels from the simulator")
    if mode in {"delta_lrm", "gamma_pwlr"} and delta_lrm is None:
        raise ValueError(f"{mode} mode requires LRM delta labels from the simulator")
    if mode == "gamma_pwlr" and gamma_pwlr is None:
        raise ValueError("gamma_pwlr mode requires PW-LR gamma labels from the simulator")


def _build_dataset(
    mode: Mode,
    x: Tensor,
    price: Tensor,
    delta_pw: Tensor,
    delta_lrm: Tensor,
    gamma_pwlr: Tensor | None,
) -> TensorDataset:
    if mode == "gamma_pwlr":
        if gamma_pwlr is None:
            raise ValueError("gamma_pwlr mode requires gamma labels")
        return TensorDataset(
            x,
            price,
            delta_lrm,
            gamma_pwlr,
            price,
            delta_pw,
            gamma_pwlr,
        )

    return TensorDataset(x, price, delta_pw, delta_lrm)


__all__ = ["Mode", "diffml_price"]
