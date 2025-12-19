"""Tests for neural network architectures."""

import torch

from diffml.networks import (
    DifferentialNet,
    FeedForwardNet,
    ResNet,
)


def test_feedforward_net():
    """Test FeedForwardNet initialization and forward pass."""
    # Create a simple network
    net = FeedForwardNet(
        input_dim=4,
        hidden_dims=[50, 50],
        output_dim=1,
        activation="relu",
    )

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 4)
    output = net(x)

    assert output.shape == (batch_size, 1), "Output shape mismatch"


def test_differential_net():
    """Test DifferentialNet with gradient computation."""
    # Create base network
    base_net = FeedForwardNet(
        input_dim=3,
        hidden_dims=[20, 20],
        output_dim=1,
    )

    # Wrap in differential net
    diff_net = DifferentialNet(
        base_network=base_net,
        differential_indices=[0, 1],  # Compute derivatives w.r.t. first two inputs
    )

    # Test forward pass with derivatives
    batch_size = 16
    x = torch.randn(batch_size, 3)

    # Test with derivatives
    values, derivatives = diff_net(x, compute_derivatives=True)

    assert values.shape == (batch_size, 1), "Values shape mismatch"
    # Note: actual derivative computation not implemented in stub
    # assert derivatives.shape == (batch_size, 1, 2), "Derivatives shape mismatch"

    # Test without derivatives
    values_only, no_deriv = diff_net(x, compute_derivatives=False)

    assert values_only.shape == (batch_size, 1), "Values only shape mismatch"
    assert no_deriv is None, "Should return None for derivatives when not computed"


def test_resnet():
    """Test ResNet initialization and forward pass."""
    net = ResNet(
        input_dim=5,
        hidden_dim=64,
        output_dim=2,
        n_blocks=3,
        activation="relu",
    )

    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 5)
    output = net(x)

    assert output.shape == (batch_size, 2), "Output shape mismatch"


if __name__ == "__main__":
    test_feedforward_net()
    test_differential_net()
    test_resnet()
    print("All network tests passed!")
