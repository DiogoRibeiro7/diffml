"""Simulation adapters exposed to the article-replication workflows.

The goal of this module is to make the Monte Carlo data-generation stage
explicit and testable.  Each simulator follows a tiny protocol that exposes a
``simulate`` method returning raw path data together with the payoff-specific
quantities that Differential ML training expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from diffml.config import BSParams, DEFAULT_DTYPE, get_device
from diffml.simulation import simulate_bs_terminal, simulate_bs_two_step


@dataclass(slots=True)
class SimulationResult:
    """Container returned by :class:`InstrumentedSimulator` objects."""

    paths: Tensor
    payoffs: Tensor
    pathwise_deltas: Tensor | None = None
    lrm_deltas: Tensor | None = None
    pwlr_gammas: Tensor | None = None


class InstrumentedSimulator(Protocol):
    """Protocol describing the minimum surface required by :func:`diffml_price`."""

    def simulate(
        self,
        x: Tensor,
        n_paths: int,
        seed: int | None = None,
    ) -> SimulationResult:  # pragma: no cover - protocol definition only
        ...


def _prepare_inputs(x: Tensor) -> Tensor:
    """Return ``x`` as a float64 tensor on the default training device."""

    if x.dim() != 2:
        raise ValueError(f"Expected 2D input features of shape (m, d), got {x.shape}")
    device = get_device()
    return x.to(device=device, dtype=DEFAULT_DTYPE)


def _discount_factor(rate: float, maturity: float, *, device: torch.device) -> Tensor:
    return torch.exp(torch.tensor(-rate * maturity, dtype=DEFAULT_DTYPE, device=device))


@dataclass(slots=True)
class DigitalCallSimulator:
    """Simulator for single-asset digital call options under Black-Scholes."""

    strike: float
    params: BSParams
    default_n_paths: int = 10_000

    def simulate(self, x: Tensor, n_paths: int, seed: int | None = None) -> SimulationResult:
        if n_paths <= 0:
            if self.default_n_paths <= 0:
                raise ValueError("n_paths must be positive")
            n_paths = self.default_n_paths
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        features = _prepare_inputs(x)
        if features.shape[1] != 1:
            raise ValueError(
                f"DigitalCallSimulator expects 1D inputs, received shape {features.shape}"
            )

        terminal, xi = simulate_bs_terminal(features, self.params, n_paths, seed=seed)
        device = terminal.device
        discount = _discount_factor(self.params.r, self.params.T, device=device)
        payoffs = discount * (terminal > self.strike).to(dtype=terminal.dtype)

        sqrt_T = torch.sqrt(torch.tensor(self.params.T, dtype=DEFAULT_DTYPE, device=device))
        score = xi / (features * self.params.sigma * sqrt_T)
        lrm_deltas = payoffs * score
        zeros = torch.zeros_like(payoffs)

        return SimulationResult(
            paths=terminal.unsqueeze(-1),
            payoffs=payoffs,
            pathwise_deltas=zeros,
            lrm_deltas=lrm_deltas,
        )


@dataclass(slots=True)
class BarrierCallSimulator:
    """Simulator for down-and-out call options observed at ``T1`` then matured at ``T2``."""

    strike: float
    barrier: float
    params: BSParams
    T1: float
    T2: float
    default_n_paths: int = 10_000

    def simulate(self, x: Tensor, n_paths: int, seed: int | None = None) -> SimulationResult:
        if n_paths <= 0:
            if self.default_n_paths <= 0:
                raise ValueError("n_paths must be positive")
            n_paths = self.default_n_paths
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        features = _prepare_inputs(x)
        if features.shape[1] != 1:
            raise ValueError(
                f"BarrierCallSimulator expects 1D inputs, received shape {features.shape}"
            )

        S1, S2, xi1, _ = simulate_bs_two_step(
            features,
            self.params,
            self.T1,
            self.T2,
            n_paths,
            seed=seed,
        )
        device = S1.device
        discount = _discount_factor(self.params.r, self.T2, device=device)
        survived = (S1 > self.barrier).to(dtype=DEFAULT_DTYPE)
        call_payoff = torch.maximum(
            S2 - self.strike,
            torch.tensor(0.0, dtype=DEFAULT_DTYPE, device=device),
        )
        payoffs = discount * survived * call_payoff

        itm = (S2 > self.strike).to(dtype=DEFAULT_DTYPE)
        delta_paths = discount * survived * itm * (S2 / features)
        delta_pw = delta_paths

        if self.T1 > 0:
            sqrt_T1 = torch.sqrt(torch.tensor(self.T1, dtype=DEFAULT_DTYPE, device=device))
            score = xi1 / (features * self.params.sigma * sqrt_T1)
        else:
            score = torch.zeros_like(xi1)
        lrm = payoffs * score

        paths = torch.stack((S1, S2), dim=-1)
        return SimulationResult(
            paths=paths,
            payoffs=payoffs,
            pathwise_deltas=delta_pw,
            lrm_deltas=lrm,
        )


@dataclass(slots=True)
class BasketDigitalSimulator:
    """Simulator for Bachelier basket digital options (multi-dimensional inputs)."""

    strike: float
    sigma: float
    T: float
    r: float = 0.0
    weights: Tensor | None = None
    sigma_vec: Tensor | None = None
    default_n_paths: int = 5_000

    def simulate(self, x: Tensor, n_paths: int, seed: int | None = None) -> SimulationResult:
        if n_paths <= 0:
            if self.default_n_paths <= 0:
                raise ValueError("n_paths must be positive")
            n_paths = self.default_n_paths
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        features = _prepare_inputs(x)
        m, d = features.shape
        device = features.device

        if seed is not None:
            torch.manual_seed(seed)

        if self.weights is None:
            weights = torch.ones(d, dtype=DEFAULT_DTYPE, device=device) / d
        else:
            weights = self.weights.to(device=device, dtype=DEFAULT_DTYPE)
            if weights.shape != (d,):
                raise ValueError(f"weights must have shape ({d},), got {weights.shape}")

        if self.sigma_vec is None:
            sigma_vec = torch.full((d,), self.sigma, dtype=DEFAULT_DTYPE, device=device)
        else:
            sigma_vec = self.sigma_vec.to(device=device, dtype=DEFAULT_DTYPE)
            if sigma_vec.shape != (d,):
                raise ValueError(f"sigma_vec must have shape ({d},), got {sigma_vec.shape}")

        sqrt_T = torch.sqrt(torch.tensor(self.T, dtype=DEFAULT_DTYPE, device=device))
        xi = torch.randn(m, n_paths, d, dtype=DEFAULT_DTYPE, device=device)
        terminal = features.unsqueeze(1) + sigma_vec.view(1, 1, d) * sqrt_T * xi

        basket = torch.sum(weights.view(1, 1, d) * terminal, dim=2)
        discount = _discount_factor(self.r, self.T, device=device)
        indicator = (basket > self.strike).to(dtype=DEFAULT_DTYPE)
        payoffs = discount * indicator

        zeros = torch.zeros_like(terminal)
        scores = xi / (sigma_vec.view(1, 1, d) * sqrt_T)
        lrm = payoffs.unsqueeze(-1) * scores

        return SimulationResult(
            paths=terminal,
            payoffs=payoffs,
            pathwise_deltas=zeros,
            lrm_deltas=lrm,
        )


__all__ = [
    "SimulationResult",
    "InstrumentedSimulator",
    "DigitalCallSimulator",
    "BarrierCallSimulator",
    "BasketDigitalSimulator",
]
