"""Training utilities for differential machine learning.

This module provides training loops, callbacks, and utilities for training
neural networks with differential machine learning.
"""

from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from diffml.config import TrainingConfig, get_device
from diffml.losses import dml_loss

# Type alias for training modes
Mode = Literal["standard", "delta_pathwise", "delta_lrm", "gamma_pwlr"]


def nn_value_delta_gamma(
    model: nn.Module,
    x: Tensor,
    compute_delta: bool = True,
    compute_gamma: bool = False,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """Compute neural network value and optionally its derivatives (delta and gamma).

    This function computes the output value from a neural network and optionally
    its first derivative (delta) and second derivative (gamma) with respect to
    the input using automatic differentiation.

    Parameters
    ----------
    model : nn.Module
        The neural network model.
    x : Tensor
        Input tensor of shape (batch_size, input_dim).
    compute_delta : bool, optional
        Whether to compute the first derivative (delta). Default is True.
    compute_gamma : bool, optional
        Whether to compute the second derivative (gamma). Default is False.

    Returns
    -------
    tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        A tuple containing:
        - value: The model output of shape (batch_size, 1).
        - delta: The first derivative if compute_delta is True, else None.
                Shape: (batch_size, input_dim).
        - gamma: The second derivative if compute_gamma is True, else None.
                Shape: (batch_size, 1) for 1D input only.

    Raises
    ------
    ValueError
        If compute_gamma is True and input dimension is not 1.

    Notes
    -----
    - For gamma computation, the input dimension must be 1 (single feature).
    - Gradients are computed using torch.autograd.grad.
    - The function handles gradient graph creation properly for higher-order derivatives.

    Examples
    --------
    >>> model = PricingNet(input_dim=1, hidden_dim=20, n_hidden=4)
    >>> x = torch.randn(32, 1, requires_grad=True)
    >>> value, delta, gamma = nn_value_delta_gamma(model, x, compute_gamma=True)
    """
    # Check gamma computation constraint
    if compute_gamma and x.shape[1] != 1:
        raise ValueError(
            f"Gamma computation is only supported for input_dim == 1, "
            f"but got input with shape {x.shape} (input_dim = {x.shape[1]})"
        )

    # Initialize outputs
    delta = None
    gamma = None

    # Ensure input requires gradients for derivative computation
    if compute_delta or compute_gamma:
        x = x.requires_grad_(True)

    # Forward pass through the model
    value = model(x)

    # Compute delta (first derivative) if requested
    if compute_delta:
        # Compute gradient of output with respect to input
        # grad_outputs is ones to sum over batch dimension
        grad_outputs = torch.ones_like(value)

        # Create graph is True if we need gamma later
        delta = torch.autograd.grad(
            outputs=value,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=compute_gamma,  # Keep graph for second derivative
            retain_graph=compute_gamma,
        )[0]

    # Compute gamma (second derivative) if requested
    if compute_gamma:
        # For 1D input, gamma is the second derivative d²V/dx²
        # We need to compute the derivative of delta with respect to x
        if delta is None:
            # If delta wasn't computed, compute it now
            grad_outputs = torch.ones_like(value)
            delta = torch.autograd.grad(
                outputs=value,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]

        # Compute second derivative
        # Since we have 1D input, delta has shape (batch_size, 1)
        # We compute d(delta)/dx to get gamma
        grad_outputs_2 = torch.ones_like(delta)
        gamma = torch.autograd.grad(
            outputs=delta,
            inputs=x,
            grad_outputs=grad_outputs_2,
            create_graph=False,  # No need for third derivatives
            retain_graph=False,
        )[0]

    return value, delta, gamma


class EarlyStopping:
    """Early stopping callback for training.

    Parameters
    ----------
    patience : int
        Number of epochs to wait before stopping.
    min_delta : float
        Minimum change to qualify as improvement.
    mode : str
        'min' for minimizing metric, 'max' for maximizing.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        """Initialize early stopping."""
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric: float) -> bool:
        """Check if training should stop.

        Parameters
        ----------
        metric : float
            Current metric value.

        Returns
        -------
        bool
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = metric
        elif self._is_improvement(metric):
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_improvement(self, metric: float) -> bool:
        """Check if metric improved.

        Parameters
        ----------
        metric : float
            Current metric value.

        Returns
        -------
        bool
            True if metric improved.
        """
        if self.mode == "min":
            return metric < self.best_score - self.min_delta
        else:
            return metric > self.best_score + self.min_delta


class Trainer:
    """Trainer for differential machine learning models.

    Parameters
    ----------
    model : nn.Module
        Neural network model.
    loss_fn : nn.Module
        Loss function.
    optimizer : optim.Optimizer
        Optimizer.
    device : torch.device
        Device for computation.
    scheduler : Optional[Any]
        Learning rate scheduler.
    early_stopping : Optional[EarlyStopping]
        Early stopping callback.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        scheduler: Optional[Any] = None,
        early_stopping: Optional[EarlyStopping] = None,
    ) -> None:
        """Initialize the trainer."""
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.early_stopping = early_stopping

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        verbose: bool = True,
    ) -> float:
        """Train for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        epoch : int
            Current epoch number.
        verbose : bool
            Whether to show progress bar.

        Returns
        -------
        float
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        iterator = tqdm(train_loader, desc=f"Epoch {epoch}") if verbose else train_loader

        for batch in iterator:
            # Unpack batch (implementation depends on dataset)
            features, targets, sensitivities = batch
            features = features.to(self.device)
            targets = targets.to(self.device)
            sensitivities = sensitivities.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Model output depends on architecture
            # This is a stub - actual implementation will vary
            predictions = self.model(features)
            loss = self.loss_fn(predictions, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / n_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(
        self,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> float:
        """Validate the model.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader.
        verbose : bool
            Whether to show progress.

        Returns
        -------
        float
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                features, targets, sensitivities = batch
                features = features.to(self.device)
                targets = targets.to(self.device)
                sensitivities = sensitivities.to(self.device)

                predictions = self.model(features)
                loss = self.loss_fn(predictions, targets)

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        self.val_losses.append(avg_loss)

        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_epochs: int = 100,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train the model.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : Optional[DataLoader]
            Validation data loader.
        n_epochs : int
            Number of epochs.
        verbose : bool
            Whether to show progress.

        Returns
        -------
        Dict[str, List[float]]
            Training history.
        """
        for epoch in range(1, n_epochs + 1):
            # Training
            train_loss = self.train_epoch(train_loader, epoch, verbose)

            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader, verbose)

                if verbose:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                          f"Val Loss = {val_loss:.4f}")

                # Early stopping
                if self.early_stopping is not None:
                    if self.early_stopping(val_loss):
                        if verbose:
                            print(f"Early stopping triggered at epoch {epoch}")
                        break
            else:
                if verbose:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs,
) -> optim.Optimizer:
    """Create an optimizer.

    Parameters
    ----------
    model : nn.Module
        Model to optimize.
    optimizer_name : str
        Name of optimizer.
    learning_rate : float
        Learning rate.
    weight_decay : float
        Weight decay for L2 regularization.
    **kwargs
        Additional optimizer arguments.

    Returns
    -------
    optim.Optimizer
        Configured optimizer.
    """
    optimizers = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
    }

    optimizer_class = optimizers.get(optimizer_name.lower(), optim.Adam)

    return optimizer_class(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        **kwargs,
    )


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str,
    **kwargs,
) -> Optional[Any]:
    """Create a learning rate scheduler.

    Parameters
    ----------
    optimizer : optim.Optimizer
        Optimizer to schedule.
    scheduler_name : str
        Name of scheduler.
    **kwargs
        Additional scheduler arguments.

    Returns
    -------
    Optional[Any]
        Configured scheduler or None.
    """
    schedulers = {
        "step": optim.lr_scheduler.StepLR,
        "exponential": optim.lr_scheduler.ExponentialLR,
        "cosine": optim.lr_scheduler.CosineAnnealingLR,
        "reduce_on_plateau": optim.lr_scheduler.ReduceLROnPlateau,
    }

    scheduler_class = schedulers.get(scheduler_name.lower())

    if scheduler_class is None:
        return None

    return scheduler_class(optimizer, **kwargs)


def rmse(pred: Tensor, target: Tensor) -> float:
    """Calculate root mean squared error between predictions and targets.

    Parameters
    ----------
    pred : Tensor
        Predicted values.
    target : Tensor
        Target values.

    Returns
    -------
    float
        Root mean squared error.

    Raises
    ------
    ValueError
        If pred and target have different shapes.

    Examples
    --------
    >>> pred = torch.tensor([[1.0], [2.0], [3.0]])
    >>> target = torch.tensor([[1.1], [1.9], [3.2]])
    >>> error = rmse(pred, target)
    >>> print(f"RMSE: {error:.4f}")
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred has shape {pred.shape}, target has shape {target.shape}"
        )

    mse = torch.mean((pred - target) ** 2)
    return float(torch.sqrt(mse).item())


def train_model(
    model: nn.Module,
    dataset: TensorDataset,
    config: TrainingConfig,
    mode: Mode,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Train a neural network model using differential machine learning.

    Implements training with different modes of sensitivity regularization:
    - standard: Price loss only (no delta/gamma regularization)
    - delta_pathwise: Use pathwise delta estimates in loss
    - delta_lrm: Use likelihood ratio method delta estimates in loss
    - gamma_pwlr: Use both LRM delta and pathwise-LR gamma in loss

    Parameters
    ----------
    model : nn.Module
        Neural network model to train (e.g., PricingNet).
    dataset : TensorDataset
        Dataset containing features and labels. Expected formats:
        - Digital/barrier: (x, price, delta_pw, delta_lrm)
        - Basket: (x, price, delta_pw_avg, delta_lrm_avg, delta_true)
        - Gamma portfolio: (x, price_true, delta_true, gamma_true, price_mc, delta_pw, gamma_pwlr)
    config : TrainingConfig
        Training configuration with hyperparameters.
    mode : Mode
        Training mode determining which sensitivities to use:
        - "standard": Price loss only
        - "delta_pathwise": Price + pathwise delta loss
        - "delta_lrm": Price + LRM delta loss
        - "gamma_pwlr": Price + LRM delta + PW-LR gamma loss
    device : Optional[torch.device]
        Device for computation. If None, uses get_device().

    Returns
    -------
    nn.Module
        Trained model (same object, modified in-place).

    Raises
    ------
    ValueError
        If mode is invalid or dataset format doesn't match expected structure.

    Examples
    --------
    >>> from diffml.networks import PricingNet
    >>> model = PricingNet(input_dim=1, hidden_dim=50, n_hidden=4)
    >>> # Assume dataset is prepared
    >>> config = TrainingConfig(n_epochs=100, lr_initial=1e-3, lambda_delta=1.0)
    >>> trained_model = train_model(model, dataset, config, mode="delta_pathwise")
    """
    # Get device if not provided
    if device is None:
        device = get_device()

    # Move model to device
    model = model.to(device)
    model.train()

    # Create data loader with batching and shuffling
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Create optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=config.lr_initial)

    # Create cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.n_epochs,
        eta_min=config.lr_min
    )

    # Training loop over epochs
    for epoch in range(config.n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Iterate over batches
        for batch in dataloader:
            # Move batch to device
            batch = [b.to(device) for b in batch]

            # Unpack batch based on dataset type
            # We infer the dataset type from the number of tensors
            if len(batch) == 4:
                # Digital/barrier format: (x, price, delta_pw, delta_lrm)
                x, true_price, delta_pw, delta_lrm = batch
                gamma_pwlr = None
            elif len(batch) == 5:
                # Basket format: (x, price, delta_pw_avg, delta_lrm_avg, delta_true)
                x, true_price, delta_pw, delta_lrm, _ = batch
                gamma_pwlr = None
            elif len(batch) == 7:
                # Gamma portfolio format: (x, price_true, delta_true, gamma_true, price_mc, delta_pw, gamma_pwlr)
                x, true_price, delta_true, gamma_true, _, delta_pw, gamma_pwlr = batch
                # For gamma portfolio, we use the true analytical labels for training
                delta_lrm = delta_true  # Use analytical delta as LRM stand-in
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} tensors")

            # Zero gradients
            optimizer.zero_grad()

            # Compute neural network outputs based on mode
            if mode == "standard":
                # Standard mode: only compute value (price)
                pred_price, pred_delta, pred_gamma = nn_value_delta_gamma(
                    model, x,
                    compute_delta=False,
                    compute_gamma=False
                )
            elif mode in ["delta_pathwise", "delta_lrm"]:
                # Delta modes: compute value and delta
                pred_price, pred_delta, pred_gamma = nn_value_delta_gamma(
                    model, x,
                    compute_delta=True,
                    compute_gamma=False
                )
            elif mode == "gamma_pwlr":
                # Gamma mode: compute value, delta, and gamma
                # Note: gamma only supported for 1D input
                if x.shape[1] != 1:
                    raise ValueError(
                        f"Gamma mode requires 1D input, but got input with shape {x.shape}"
                    )
                pred_price, pred_delta, pred_gamma = nn_value_delta_gamma(
                    model, x,
                    compute_delta=True,
                    compute_gamma=True
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Handle multi-dimensional input for basket options
            # If x has more than 1 feature, average the delta over features
            if pred_delta is not None and x.shape[1] > 1:
                # Average delta across input dimensions for comparison with scalar labels
                pred_delta_scalar = pred_delta.mean(dim=1, keepdim=True)
            else:
                pred_delta_scalar = pred_delta

            # Select appropriate delta labels based on mode
            if mode == "delta_pathwise":
                true_delta = delta_pw
            elif mode == "delta_lrm":
                true_delta = delta_lrm
            elif mode == "gamma_pwlr":
                true_delta = delta_lrm  # Use LRM delta for gamma mode
            else:
                true_delta = None

            # Compute loss using dml_loss function
            if mode == "standard":
                # Price loss only
                loss = dml_loss(
                    pred_price=pred_price,
                    true_price=true_price,
                    lambda_delta=0.0,
                    lambda_gamma=0.0
                )
            elif mode in ["delta_pathwise", "delta_lrm"]:
                # Price + delta loss
                loss = dml_loss(
                    pred_price=pred_price,
                    true_price=true_price,
                    pred_delta_scalar=pred_delta_scalar,
                    true_delta_scalar=true_delta,
                    lambda_delta=config.lambda_delta,
                    lambda_gamma=0.0
                )
            elif mode == "gamma_pwlr":
                # Price + delta + gamma loss
                loss = dml_loss(
                    pred_price=pred_price,
                    true_price=true_price,
                    pred_delta_scalar=pred_delta_scalar,
                    true_delta_scalar=true_delta,
                    pred_gamma=pred_gamma,
                    true_gamma=gamma_pwlr,
                    lambda_delta=config.lambda_delta,
                    lambda_gamma=config.lambda_gamma
                )

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Accumulate loss for monitoring
            epoch_loss += loss.item()
            n_batches += 1

        # Learning rate scheduler step
        scheduler.step()

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            avg_loss = epoch_loss / n_batches
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch + 1}/{config.n_epochs}] "
                f"Loss: {avg_loss:.6f} "
                f"LR: {current_lr:.6f}"
            )

    # Set model to evaluation mode
    model.eval()

    return model
