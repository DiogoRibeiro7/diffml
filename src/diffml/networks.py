"""Neural network architectures for differential ML.

This module implements various neural network architectures used in the
differential machine learning experiments.
"""

from typing import Optional

import torch
import torch.nn as nn


class PricingNet(nn.Module):
    """Neural network for option pricing with differential ML.

    Implements a multi-layer perceptron with Softplus activations,
    specifically designed for learning option prices and their sensitivities
    (Greeks) using differential machine learning.

    Parameters
    ----------
    input_dim : int
        Dimension of the input features (e.g., 1 for spot price only).
    hidden_dim : int
        Number of neurons in each hidden layer.
    n_hidden : int
        Number of hidden layers.

    Raises
    ------
    ValueError
        If input_dim, hidden_dim, or n_hidden are not positive integers.

    Examples
    --------
    >>> model = PricingNet(input_dim=1, hidden_dim=20, n_hidden=4)
    >>> x = torch.randn(32, 1)  # batch of 32 spot prices
    >>> price = model(x)  # predicted option prices
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 20,
        n_hidden: int = 4,
    ) -> None:
        """Initialize the pricing network.

        Builds an MLP with n_hidden blocks of (Linear -> Softplus),
        followed by a final linear layer to produce a single output.
        """
        super().__init__()

        # Validate inputs
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_hidden <= 0:
            raise ValueError(f"n_hidden must be positive, got {n_hidden}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden

        # Build the network layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Softplus())

        # Hidden layers
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus())

        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dim, 1))

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1) representing predicted prices.

        Raises
        ------
        ValueError
            If input tensor has incorrect shape.
        """
        # Validate input shape
        if x.dim() != 2:
            raise ValueError(
                f"Input must be 2D tensor (batch_size, input_dim), got shape {x.shape}"
            )
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {x.shape[1]}"
            )

        # Forward pass through the network
        return self.network(x)


class FeedForwardNet(nn.Module):
    """Feedforward neural network for function approximation.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    hidden_dims : List[int]
        List of hidden layer dimensions.
    output_dim : int
        Output dimension.
    activation : str, optional
        Activation function name ('relu', 'tanh', 'sigmoid', 'elu').
    dropout_rate : float, optional
        Dropout rate for regularization.
    batch_norm : bool, optional
        Whether to use batch normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        """Initialize the feedforward network."""
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

        # Build the network
        self.layers = self._build_layers()

    def _get_activation(self) -> nn.Module:
        """Get activation function module.

        Returns
        -------
        nn.Module
            Activation function module.
        """
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
            "selu": nn.SELU(),
        }
        return activations.get(self.activation_name.lower(), nn.ReLU())

    def _build_layers(self) -> nn.Sequential:
        """Build the network layers.

        Returns
        -------
        nn.Sequential
            Sequential container of network layers.
        """
        layers = []
        prev_dim = self.input_dim

        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(self._get_activation())

            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

            prev_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, self.output_dim))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        return self.layers(x)


class DifferentialNet(nn.Module):
    """Neural network with differential outputs for sensitivity learning.

    This network outputs both the function value and its derivatives
    with respect to specified inputs.

    Parameters
    ----------
    base_network : nn.Module
        Base neural network for function approximation.
    differential_indices : Optional[List[int]]
        Indices of inputs for which to compute derivatives.
        If None, computes derivatives for all inputs.
    """

    def __init__(
        self,
        base_network: nn.Module,
        differential_indices: Optional[list[int]] = None,
    ) -> None:
        """Initialize the differential network."""
        super().__init__()
        self.base_network = base_network
        self.differential_indices = differential_indices

    def forward(
        self, x: torch.Tensor, compute_derivatives: bool = True
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional derivative computation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        compute_derivatives : bool, optional
            Whether to compute derivatives.

        Returns
        -------
        tuple[torch.Tensor, Optional[torch.Tensor]]
            Tuple of (values, derivatives) where:
            - values: tensor of shape (batch_size, output_dim)
            - derivatives: tensor of shape (batch_size, output_dim, n_differential_inputs)
              or None if compute_derivatives is False.
        """
        if not compute_derivatives:
            values = self.base_network(x)
            return values, None

        # Enable gradient computation for specified inputs
        x_grad = x.requires_grad_(True)

        # Forward pass
        values = self.base_network(x_grad)

        # Compute derivatives
        if self.differential_indices is not None:
            grad_indices = self.differential_indices
        else:
            grad_indices = list(range(x.shape[1]))

        derivatives = []
        for output_idx in range(values.shape[1]):
            output_derivatives = []
            for input_idx in grad_indices:
                # Compute gradient of output[output_idx] w.r.t. input[input_idx]
                grad = torch.autograd.grad(
                    outputs=values[:, output_idx],
                    inputs=x_grad,
                    grad_outputs=torch.ones_like(values[:, output_idx]),
                    create_graph=True,
                    retain_graph=True,
                )[0][:, input_idx]
                output_derivatives.append(grad)
            derivatives.append(torch.stack(output_derivatives, dim=-1))

        derivatives = torch.stack(derivatives, dim=1)

        return values, derivatives


class ResidualBlock(nn.Module):
    """Residual block for deep networks.

    Parameters
    ----------
    dim : int
        Dimension of the block (input and output).
    activation : str, optional
        Activation function name.
    dropout_rate : float, optional
        Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize the residual block."""
        super().__init__()

        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            activations.get(activation.lower(), nn.ReLU()),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

        self.activation = activations.get(activation.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor with residual connection.
        """
        return self.activation(x + self.block(x))


class ResNet(nn.Module):
    """Residual network for complex function approximation.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    hidden_dim : int
        Hidden layer dimension.
    output_dim : int
        Output dimension.
    n_blocks : int
        Number of residual blocks.
    activation : str, optional
        Activation function name.
    dropout_rate : float, optional
        Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_blocks: int,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize the residual network."""
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, activation, dropout_rate)
            for _ in range(n_blocks)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        x = self.input_proj(x)

        for block in self.res_blocks:
            x = block(x)

        x = self.output_proj(x)

        return x
