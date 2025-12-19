"""Loss functions for differential machine learning.

This module implements various loss functions for training neural networks
with differential machine learning, including weighted sensitivity losses.
"""


import torch
import torch.nn as nn
from torch import Tensor


def dml_loss(
    pred_price: Tensor,
    true_price: Tensor,
    pred_delta_vector: Tensor | None = None,
    true_delta_vector: Tensor | None = None,
    pred_delta_scalar: Tensor | None = None,
    true_delta_scalar: Tensor | None = None,
    pred_gamma: Tensor | None = None,
    true_gamma: Tensor | None = None,
    lambda_delta: float = 0.0,
    lambda_gamma: float = 0.0,
) -> Tensor:
    """Compute differential machine learning loss with price, delta, and gamma terms.

    Implements the differential ML loss function combining:
    - Base MSE loss on prices
    - Optional delta (first derivative) regularization
    - Optional gamma (second derivative) regularization

    The total loss is:
    L = MSE(pred_price, true_price)
        + lambda_delta * MSE(pred_delta, true_delta)
        + lambda_gamma * MSE(pred_gamma, true_gamma)

    Parameters
    ----------
    pred_price : Tensor
        Predicted prices from neural network, shape (batch_size, 1).
    true_price : Tensor
        True prices (labels), shape (batch_size, 1).
    pred_delta_vector : Tensor | None
        Predicted delta vector (for multi-dim input), shape (batch_size, d).
    true_delta_vector : Tensor | None
        True delta vector labels, shape (batch_size, d).
    pred_delta_scalar : Tensor | None
        Predicted delta scalar (for 1D input or averaged), shape (batch_size, 1).
    true_delta_scalar : Tensor | None
        True delta scalar labels, shape (batch_size, 1).
    pred_gamma : Tensor | None
        Predicted gamma (second derivative), shape (batch_size, 1).
    true_gamma : Tensor | None
        True gamma labels, shape (batch_size, 1).
    lambda_delta : float
        Weight for delta regularization term. Default is 0.0 (no delta term).
    lambda_gamma : float
        Weight for gamma regularization term. Default is 0.0 (no gamma term).

    Returns
    -------
    Tensor
        Scalar loss value combining all terms.

    Raises
    ------
    ValueError
        If lambda_delta > 0 but no delta predictions/labels provided.
        If lambda_gamma > 0 but no gamma predictions/labels provided.
        If both vector and scalar deltas are provided (ambiguous).
        If shapes don't match between predictions and labels.

    Examples
    --------
    >>> # Price-only loss
    >>> loss = dml_loss(pred_price, true_price)

    >>> # Price + delta loss (scalar)
    >>> loss = dml_loss(
    ...     pred_price, true_price,
    ...     pred_delta_scalar=pred_delta, true_delta_scalar=true_delta,
    ...     lambda_delta=1.0
    ... )

    >>> # Price + delta + gamma loss
    >>> loss = dml_loss(
    ...     pred_price, true_price,
    ...     pred_delta_scalar=pred_delta, true_delta_scalar=true_delta,
    ...     pred_gamma=pred_gamma, true_gamma=true_gamma,
    ...     lambda_delta=1.0, lambda_gamma=0.5
    ... )
    """
    # Validate price shapes
    if pred_price.shape != true_price.shape:
        raise ValueError(
            f"Shape mismatch: pred_price {pred_price.shape} vs true_price {true_price.shape}"
        )

    # Base loss: MSE on prices
    mse_loss = nn.MSELoss()
    loss = mse_loss(pred_price, true_price)

    # Add delta regularization if requested
    if lambda_delta > 0:
        # Check that we have delta predictions and labels
        has_vector_delta = (pred_delta_vector is not None) and (true_delta_vector is not None)
        has_scalar_delta = (pred_delta_scalar is not None) and (true_delta_scalar is not None)

        if has_vector_delta and has_scalar_delta:
            raise ValueError(
                "Ambiguous delta specification: both vector and scalar deltas provided. "
                "Please provide only one type of delta."
            )

        if has_vector_delta:
            # Vector delta case (multi-dimensional input)
            if pred_delta_vector.shape != true_delta_vector.shape:
                raise ValueError(
                    f"Shape mismatch: pred_delta_vector {pred_delta_vector.shape} "
                    f"vs true_delta_vector {true_delta_vector.shape}"
                )
            # Add weighted MSE loss on delta vectors
            delta_loss = mse_loss(pred_delta_vector, true_delta_vector)
            loss = loss + lambda_delta * delta_loss

        elif has_scalar_delta:
            # Scalar delta case (1D input or averaged multi-dim)
            if pred_delta_scalar.shape != true_delta_scalar.shape:
                raise ValueError(
                    f"Shape mismatch: pred_delta_scalar {pred_delta_scalar.shape} "
                    f"vs true_delta_scalar {true_delta_scalar.shape}"
                )
            # Add weighted MSE loss on scalar deltas
            delta_loss = mse_loss(pred_delta_scalar, true_delta_scalar)
            loss = loss + lambda_delta * delta_loss

        else:
            raise ValueError(
                f"lambda_delta = {lambda_delta} > 0 but no delta predictions/labels provided. "
                "Please provide either (pred_delta_vector, true_delta_vector) or "
                "(pred_delta_scalar, true_delta_scalar)."
            )

    # Add gamma regularization if requested
    if lambda_gamma > 0:
        if (pred_gamma is None) or (true_gamma is None):
            raise ValueError(
                f"lambda_gamma = {lambda_gamma} > 0 but gamma predictions/labels not provided. "
                "Please provide both pred_gamma and true_gamma."
            )

        if pred_gamma.shape != true_gamma.shape:
            raise ValueError(
                f"Shape mismatch: pred_gamma {pred_gamma.shape} vs true_gamma {true_gamma.shape}"
            )

        # Add weighted MSE loss on gamma
        gamma_loss = mse_loss(pred_gamma, true_gamma)
        loss = loss + lambda_gamma * gamma_loss

    return loss


class DifferentialLoss(nn.Module):
    """Loss function for differential machine learning.

    Combines value loss and sensitivity (derivative) loss with weighting.

    Parameters
    ----------
    value_weight : float
        Weight for the value loss component.
    sensitivity_weight : float
        Weight for the sensitivity loss component.
    value_loss_fn : nn.Module | None
        Loss function for values (default: MSE).
    sensitivity_loss_fn : nn.Module | None
        Loss function for sensitivities (default: MSE).
    """

    def __init__(
        self,
        value_weight: float = 1.0,
        sensitivity_weight: float = 1.0,
        value_loss_fn: nn.Module | None = None,
        sensitivity_loss_fn: nn.Module | None = None,
    ) -> None:
        """Initialize the differential loss."""
        super().__init__()
        self.value_weight = value_weight
        self.sensitivity_weight = sensitivity_weight
        self.value_loss_fn = value_loss_fn or nn.MSELoss()
        self.sensitivity_loss_fn = sensitivity_loss_fn or nn.MSELoss()

    def forward(
        self,
        pred_values: torch.Tensor,
        true_values: torch.Tensor,
        pred_sensitivities: torch.Tensor | None = None,
        true_sensitivities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the differential loss.

        Parameters
        ----------
        pred_values : torch.Tensor
            Predicted values.
        true_values : torch.Tensor
            True values.
        pred_sensitivities : torch.Tensor | None
            Predicted sensitivities.
        true_sensitivities : torch.Tensor | None
            True sensitivities.

        Returns
        -------
        torch.Tensor
            Total loss value.
        """
        # Value loss
        value_loss = self.value_loss_fn(pred_values, true_values)
        total_loss = self.value_weight * value_loss

        # Sensitivity loss (if provided)
        if pred_sensitivities is not None and true_sensitivities is not None:
            sensitivity_loss = self.sensitivity_loss_fn(
                pred_sensitivities, true_sensitivities
            )
            total_loss += self.sensitivity_weight * sensitivity_loss

        return total_loss


class AdaptiveDifferentialLoss(nn.Module):
    """Adaptive differential loss with dynamic weighting.

    Automatically adjusts weights between value and sensitivity losses
    during training based on relative magnitudes.

    Parameters
    ----------
    initial_value_weight : float
        Initial weight for value loss.
    initial_sensitivity_weight : float
        Initial weight for sensitivity loss.
    adaptation_rate : float
        Rate of weight adaptation.
    normalize : bool
        Whether to normalize losses before weighting.
    """

    def __init__(
        self,
        initial_value_weight: float = 1.0,
        initial_sensitivity_weight: float = 1.0,
        adaptation_rate: float = 0.01,
        normalize: bool = True,
    ) -> None:
        """Initialize the adaptive differential loss."""
        super().__init__()
        self.value_weight = initial_value_weight
        self.sensitivity_weight = initial_sensitivity_weight
        self.adaptation_rate = adaptation_rate
        self.normalize = normalize
        self.value_loss_fn = nn.MSELoss()
        self.sensitivity_loss_fn = nn.MSELoss()

        # Running statistics for normalization
        self.value_loss_mean = 1.0
        self.sensitivity_loss_mean = 1.0

    def forward(
        self,
        pred_values: torch.Tensor,
        true_values: torch.Tensor,
        pred_sensitivities: torch.Tensor | None = None,
        true_sensitivities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the adaptive differential loss.

        Parameters
        ----------
        pred_values : torch.Tensor
            Predicted values.
        true_values : torch.Tensor
            True values.
        pred_sensitivities : torch.Tensor | None
            Predicted sensitivities.
        true_sensitivities : torch.Tensor | None
            True sensitivities.

        Returns
        -------
        torch.Tensor
            Total loss value.
        """
        # Compute individual losses
        value_loss = self.value_loss_fn(pred_values, true_values)

        if pred_sensitivities is not None and true_sensitivities is not None:
            sensitivity_loss = self.sensitivity_loss_fn(
                pred_sensitivities, true_sensitivities
            )

            # Update running means
            with torch.no_grad():
                self.value_loss_mean = (
                    1 - self.adaptation_rate
                ) * self.value_loss_mean + self.adaptation_rate * value_loss.item()
                self.sensitivity_loss_mean = (
                    1 - self.adaptation_rate
                ) * self.sensitivity_loss_mean + self.adaptation_rate * sensitivity_loss.item()

            # Normalize if requested
            if self.normalize:
                value_loss = value_loss / (self.value_loss_mean + 1e-8)
                sensitivity_loss = sensitivity_loss / (self.sensitivity_loss_mean + 1e-8)

            # Weighted combination
            total_loss = (
                self.value_weight * value_loss
                + self.sensitivity_weight * sensitivity_loss
            )
        else:
            total_loss = self.value_weight * value_loss

        return total_loss


class HuberDifferentialLoss(nn.Module):
    """Huber loss variant for differential machine learning.

    More robust to outliers than MSE loss.

    Parameters
    ----------
    value_weight : float
        Weight for value loss.
    sensitivity_weight : float
        Weight for sensitivity loss.
    delta : float
        Huber loss delta parameter.
    """

    def __init__(
        self,
        value_weight: float = 1.0,
        sensitivity_weight: float = 1.0,
        delta: float = 1.0,
    ) -> None:
        """Initialize the Huber differential loss."""
        super().__init__()
        self.value_weight = value_weight
        self.sensitivity_weight = sensitivity_weight
        self.value_loss_fn = nn.HuberLoss(delta=delta)
        self.sensitivity_loss_fn = nn.HuberLoss(delta=delta)

    def forward(
        self,
        pred_values: torch.Tensor,
        true_values: torch.Tensor,
        pred_sensitivities: torch.Tensor | None = None,
        true_sensitivities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the Huber differential loss.

        Parameters
        ----------
        pred_values : torch.Tensor
            Predicted values.
        true_values : torch.Tensor
            True values.
        pred_sensitivities : torch.Tensor | None
            Predicted sensitivities.
        true_sensitivities : torch.Tensor | None
            True sensitivities.

        Returns
        -------
        torch.Tensor
            Total loss value.
        """
        value_loss = self.value_loss_fn(pred_values, true_values)
        total_loss = self.value_weight * value_loss

        if pred_sensitivities is not None and true_sensitivities is not None:
            sensitivity_loss = self.sensitivity_loss_fn(
                pred_sensitivities, true_sensitivities
            )
            total_loss += self.sensitivity_weight * sensitivity_loss

        return total_loss


def relative_error(
    pred: torch.Tensor,
    true: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Calculate relative error.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    true : torch.Tensor
        True values.
    epsilon : float
        Small value to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Relative error.
    """
    return torch.abs(pred - true) / (torch.abs(true) + epsilon)
