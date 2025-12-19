"""Tests for loss functions."""

import torch

from diffml.losses import (
    AdaptiveDifferentialLoss,
    DifferentialLoss,
    HuberDifferentialLoss,
    relative_error,
)


def test_differential_loss():
    """Test DifferentialLoss computation."""
    loss_fn = DifferentialLoss(
        value_weight=1.0,
        sensitivity_weight=0.5,
    )

    batch_size = 16
    pred_values = torch.randn(batch_size, 1)
    true_values = torch.randn(batch_size, 1)
    pred_sensitivities = torch.randn(batch_size, 4)
    true_sensitivities = torch.randn(batch_size, 4)

    # Test with sensitivities
    loss = loss_fn(
        pred_values,
        true_values,
        pred_sensitivities,
        true_sensitivities,
    )

    assert loss.shape == torch.Size([]), "Loss should be scalar"
    assert loss.requires_grad, "Loss should require gradients"

    # Test without sensitivities
    loss_no_sens = loss_fn(pred_values, true_values)

    assert loss_no_sens.shape == torch.Size([]), "Loss should be scalar"


def test_adaptive_differential_loss():
    """Test AdaptiveDifferentialLoss."""
    loss_fn = AdaptiveDifferentialLoss(
        initial_value_weight=1.0,
        initial_sensitivity_weight=1.0,
        adaptation_rate=0.1,
        normalize=True,
    )

    batch_size = 16
    pred_values = torch.randn(batch_size, 1)
    true_values = torch.randn(batch_size, 1)
    pred_sensitivities = torch.randn(batch_size, 4)
    true_sensitivities = torch.randn(batch_size, 4)

    # Test multiple forward passes (adaptation should occur)
    for _ in range(5):
        loss = loss_fn(
            pred_values,
            true_values,
            pred_sensitivities,
            true_sensitivities,
        )

        assert loss.shape == torch.Size([]), "Loss should be scalar"
        assert loss.requires_grad, "Loss should require gradients"


def test_huber_differential_loss():
    """Test HuberDifferentialLoss."""
    loss_fn = HuberDifferentialLoss(
        value_weight=1.0,
        sensitivity_weight=1.0,
        delta=1.0,
    )

    batch_size = 16
    pred_values = torch.randn(batch_size, 1)
    true_values = torch.randn(batch_size, 1)
    pred_sensitivities = torch.randn(batch_size, 4)
    true_sensitivities = torch.randn(batch_size, 4)

    loss = loss_fn(
        pred_values,
        true_values,
        pred_sensitivities,
        true_sensitivities,
    )

    assert loss.shape == torch.Size([]), "Loss should be scalar"
    assert loss.requires_grad, "Loss should require gradients"


def test_relative_error():
    """Test relative error calculation."""
    pred = torch.tensor([1.0, 2.0, 3.0])
    true = torch.tensor([1.1, 2.2, 2.9])

    rel_err = relative_error(pred, true)

    assert rel_err.shape == true.shape, "Shape mismatch"
    assert torch.all(rel_err >= 0), "Relative error should be non-negative"

    # Test with zeros
    pred_zero = torch.tensor([0.0, 1.0])
    true_zero = torch.tensor([0.0, 0.0])

    rel_err_zero = relative_error(pred_zero, true_zero, epsilon=1e-8)

    assert torch.all(torch.isfinite(rel_err_zero)), "Should handle zeros properly"


if __name__ == "__main__":
    test_differential_loss()
    test_adaptive_differential_loss()
    test_huber_differential_loss()
    test_relative_error()
    print("All loss function tests passed!")
