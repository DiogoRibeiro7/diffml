"""Dataset generation for basket option experiments.

This module provides functions to generate training and validation datasets
for basket option pricing using differential machine learning.
Uses Bachelier model for multi-dimensional basket digital options.
"""

from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from diffml.config import DEFAULT_DTYPE, BSParams, get_device


def make_basket_digital_dataset(
    m: int,
    d: int,
    K: float,
    params: BSParams,
    w: Optional[Tensor] = None,
    sigma_vec: Optional[Tensor] = None,
    x_low: float = 80.0,
    x_high: float = 120.0,
    n_paths_per_x: int = 10,
    seed: Optional[int] = 1234
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Generate dataset for high-dimensional Bachelier basket digital option.

    Uses Bachelier (arithmetic Brownian motion) model:
    S_T = S_0 + sigma_i * sqrt(T) * Z_i

    The basket value is: B = sum_i w_i * S_T_i
    Payoff: e^(-rT) * 1_{B > K}

    Parameters
    ----------
    m : int
        Number of data points in the dataset.
    d : int
        Number of assets in the basket (dimension).
    K : float
        Strike price of the basket digital option.
    params : BSParams
        Black-Scholes/Bachelier parameters (r, sigma, T).
    w : Optional[Tensor]
        Weights for basket, shape (d,). If None, uses equal weights.
    sigma_vec : Optional[Tensor]
        Volatilities per asset, shape (d,). If None, uses params.sigma for all.
    x_low : float, optional
        Lower bound for initial prices. Default is 80.0.
    x_high : float, optional
        Upper bound for initial prices. Default is 120.0.
    n_paths_per_x : int, optional
        Number of Monte Carlo paths per data point. Default is 10.
    seed : Optional[int], optional
        Random seed for reproducibility. Default is 1234.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        A tuple containing:
        - x: Initial prices of shape (m, d)
        - price_label: Monte Carlo prices of shape (m, 1)
        - avg_delta_pathwise: Pathwise average deltas (zeros) of shape (m, 1)
        - avg_delta_lrm: LRM average deltas of shape (m, 1)
        - avg_delta_true: Analytic average deltas of shape (m, 1)

    Raises
    ------
    ValueError
        If m <= 0, d <= 0, or n_paths_per_x <= 0.
        If x_low >= x_high or K <= 0.
        If weights or sigma_vec have incorrect shapes.

    Examples
    --------
    >>> params = BSParams(r=0.05, sigma=0.2, T=0.25)
    >>> x, prices, delta_pw, delta_lrm, delta_true = make_basket_digital_dataset(
    ...     m=100, d=5, K=500.0, params=params, n_paths_per_x=10000
    ... )
    >>> x.shape, prices.shape
    (torch.Size([100, 5]), torch.Size([100, 1]))
    """
    # Input validation
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    if n_paths_per_x <= 0:
        raise ValueError(f"n_paths_per_x must be positive, got {n_paths_per_x}")
    if x_low >= x_high:
        raise ValueError(f"x_low must be less than x_high, got x_low={x_low}, x_high={x_high}")
    if K <= 0:
        raise ValueError(f"Strike K must be positive, got {K}")

    # Get device and set precision
    device = get_device()

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Generate initial prices uniformly in [x_low, x_high]
    # Shape: (m, d)
    x = torch.rand(m, d, device=device, dtype=DEFAULT_DTYPE) * (x_high - x_low) + x_low

    # Set weights (equal if not provided)
    if w is None:
        w = torch.ones(d, device=device, dtype=DEFAULT_DTYPE) / d
    else:
        w = w.to(device=device, dtype=DEFAULT_DTYPE)
        if w.shape != (d,):
            raise ValueError(f"Weights must have shape ({d},), got {w.shape}")

    # Set volatilities (use params.sigma if not provided)
    if sigma_vec is None:
        sigma_vec = torch.full((d,), params.sigma, device=device, dtype=DEFAULT_DTYPE)
    else:
        sigma_vec = sigma_vec.to(device=device, dtype=DEFAULT_DTYPE)
        if sigma_vec.shape != (d,):
            raise ValueError(f"sigma_vec must have shape ({d},), got {sigma_vec.shape}")

    # Generate standard normal shocks for all paths
    # Shape: (m, n_paths_per_x, d)
    xi = torch.randn(m, n_paths_per_x, d, device=device, dtype=DEFAULT_DTYPE)

    # Compute terminal prices using Bachelier model
    # S_T = S_0 + sigma_i * sqrt(T) * Z_i
    # x has shape (m, d), expand to (m, 1, d) for broadcasting
    # xi has shape (m, n_paths_per_x, d)
    sqrt_T = torch.sqrt(torch.tensor(params.T, dtype=DEFAULT_DTYPE))
    S_T = x.unsqueeze(1) + sigma_vec.unsqueeze(0).unsqueeze(0) * sqrt_T * xi

    # Compute basket value: B = sum_i w_i * S_T_i
    # Shape: (m, n_paths_per_x)
    basket = torch.sum(w.unsqueeze(0).unsqueeze(0) * S_T, dim=2)

    # Digital payoff: 1_{B > K}
    # Shape: (m, n_paths_per_x)
    payoff = (basket > K).to(dtype=DEFAULT_DTYPE)

    # Discount factor
    discount = torch.exp(-params.r * params.T)
    disc_payoff = discount * payoff

    # Price label: Monte Carlo mean
    # Shape: (m, 1)
    price_label = disc_payoff.mean(dim=1, keepdim=True)

    # Pathwise average delta: zeros (discontinuous payoff)
    # Shape: (m, 1)
    avg_delta_pathwise = torch.zeros(m, 1, device=device, dtype=DEFAULT_DTYPE)

    # LRM average delta
    # Score per asset i: xi_i / (sigma_i * sqrt(T))
    # LRM delta per asset: E[payoff * score_i]
    # Average delta: mean over assets
    scores = xi / (sigma_vec.unsqueeze(0).unsqueeze(0) * sqrt_T)  # Shape: (m, n_paths_per_x, d)

    # LRM deltas per asset
    # Shape: (m, d)
    lrm_deltas_per_asset = (disc_payoff.unsqueeze(2) * scores).mean(dim=1)

    # Average LRM delta (mean over assets)
    # Shape: (m, 1)
    avg_delta_lrm = lrm_deltas_per_asset.mean(dim=1, keepdim=True)

    # True average delta using analytic Bachelier formula
    # Basket B is normally distributed with:
    # mean: E[B] = sum_i w_i * x_i
    # variance: Var[B] = T * sum_i w_i^2 * sigma_i^2
    # Derivative of P(B > K) w.r.t. uniform shift in all x_i

    # Compute mean and std of basket
    # Shape: (m,)
    basket_mean = torch.sum(w.unsqueeze(0) * x, dim=1)
    basket_var = params.T * torch.sum((w * sigma_vec) ** 2)
    basket_std = torch.sqrt(basket_var)

    # Standardized strike
    # Shape: (m,)
    z = (K - basket_mean) / basket_std

    # PDF of standard normal at z
    # Shape: (m,)
    pdf_z = torch.exp(-0.5 * z * z) / torch.sqrt(2 * torch.tensor(torch.pi, dtype=DEFAULT_DTYPE))

    # True average delta: discount * pdf(z) * sum(w_i) / (basket_std * d)
    # Since we want derivative w.r.t. uniform shift, it's sum of weights times density
    # Divided by d to get average per asset
    # Shape: (m, 1)
    sum_weights = torch.sum(w)
    avg_delta_true = discount * pdf_z.unsqueeze(1) * sum_weights / (basket_std * d)

    return x, price_label, avg_delta_pathwise, avg_delta_lrm, avg_delta_true


class BasketOptionDataset(Dataset):
    """PyTorch Dataset for basket option pricing.

    Parameters
    ----------
    x : Tensor
        Initial prices of shape (m, d).
    price_label : Tensor
        Option prices of shape (m, 1).
    avg_delta_pathwise : Tensor
        Pathwise average deltas of shape (m, 1).
    avg_delta_lrm : Tensor
        LRM average deltas of shape (m, 1).
    avg_delta_true : Tensor
        True average deltas of shape (m, 1).
    """

    def __init__(
        self,
        x: Tensor,
        price_label: Tensor,
        avg_delta_pathwise: Tensor,
        avg_delta_lrm: Tensor,
        avg_delta_true: Tensor
    ) -> None:
        """Initialize the basket option dataset."""
        self.x = x
        self.price_label = price_label
        self.avg_delta_pathwise = avg_delta_pathwise
        self.avg_delta_lrm = avg_delta_lrm
        self.avg_delta_true = avg_delta_true
        self.n_samples = x.shape[0]

    def __len__(self) -> int:
        """Get dataset size.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            (x, price, avg_delta_pw, avg_delta_lrm, avg_delta_true) for the sample.
        """
        return (
            self.x[idx],
            self.price_label[idx],
            self.avg_delta_pathwise[idx],
            self.avg_delta_lrm[idx],
            self.avg_delta_true[idx]
        )


def create_basket_dataloaders(
    n_train: int = 50000,
    n_val: int = 10000,
    batch_size: int = 256,
    n_assets: int = 5,
    K: float = 500.0,
    params: Optional[BSParams] = None,
    n_paths_per_x: int = 10000,
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders for basket options.

    Parameters
    ----------
    n_train : int
        Number of training samples.
    n_val : int
        Number of validation samples.
    batch_size : int
        Batch size for dataloaders.
    n_assets : int
        Number of assets in the basket.
    K : float
        Strike price.
    params : Optional[BSParams]
        Black-Scholes/Bachelier parameters. If None, uses defaults.
    n_paths_per_x : int
        Number of MC paths per data point.
    **kwargs
        Additional arguments for make_basket_digital_dataset.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training and validation dataloaders.
    """
    if params is None:
        params = BSParams()

    # Generate training data
    x_train, price_train, delta_pw_train, delta_lrm_train, delta_true_train = make_basket_digital_dataset(
        m=n_train,
        d=n_assets,
        K=K,
        params=params,
        n_paths_per_x=n_paths_per_x,
        seed=42,
        **kwargs
    )

    # Generate validation data
    x_val, price_val, delta_pw_val, delta_lrm_val, delta_true_val = make_basket_digital_dataset(
        m=n_val,
        d=n_assets,
        K=K,
        params=params,
        n_paths_per_x=n_paths_per_x,
        seed=123,
        **kwargs
    )

    # Create datasets
    train_dataset = BasketOptionDataset(
        x_train, price_train, delta_pw_train, delta_lrm_train, delta_true_train
    )
    val_dataset = BasketOptionDataset(
        x_val, price_val, delta_pw_val, delta_lrm_val, delta_true_val
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader
