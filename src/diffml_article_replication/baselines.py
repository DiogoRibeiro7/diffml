"""Simple baseline models used for benchmarking experiments."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from diffml.networks import PricingNet


@runtime_checkable
class BaselineModel(Protocol):
    """Protocol describing baseline model behaviour."""

    def fit(self, x: Tensor, y: Tensor) -> None:  # pragma: no cover - protocol
        """Fit the model to the provided inputs."""

    def predict(self, x: Tensor) -> Tensor:  # pragma: no cover - protocol
        """Return predictions for ``x``."""


class PolynomialRegressionBaseline:
    """Least-squares polynomial regression baseline."""

    def __init__(self, degree: int = 3) -> None:
        if degree < 0:
            raise ValueError("degree must be non-negative")
        self.degree = degree
        self._weights: Tensor | None = None

    def _design_matrix(self, x: Tensor) -> Tensor:
        """Create polynomial feature matrix."""
        powers = [torch.ones_like(x)]
        for power in range(1, self.degree + 1):
            powers.append(x ** power)
        return torch.cat(powers, dim=1)

    def fit(self, x: Tensor, y: Tensor) -> None:
        """Fit least-squares weights."""
        features = self._design_matrix(x)
        solution = torch.linalg.lstsq(features, y).solution
        self._weights = solution

    def predict(self, x: Tensor) -> Tensor:
        """Evaluate the polynomial at ``x``."""
        if self._weights is None:  # pragma: no cover - defensive
            raise RuntimeError("PolynomialRegressionBaseline has not been fitted yet")
        features = self._design_matrix(x)
        return features @ self._weights


class MLPBaseline:
    """Two-layer MLP baseline trained with plain MSE loss."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        n_hidden: int = 2,
        n_epochs: int = 200,
        lr: float = 1e-3,
    ) -> None:
        self.model = PricingNet(input_dim=input_dim, hidden_dim=hidden_dim, n_hidden=n_hidden)
        self.n_epochs = n_epochs
        self.lr = lr

    def fit(self, x: Tensor, y: Tensor) -> None:
        """Train the PricingNet using vanilla MSE."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            preds = self.model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

    def predict(self, x: Tensor) -> Tensor:
        """Return network predictions in eval mode."""
        self.model.eval()
        with torch.no_grad():
            return self.model(x)


__all__ = ["BaselineModel", "PolynomialRegressionBaseline", "MLPBaseline"]
