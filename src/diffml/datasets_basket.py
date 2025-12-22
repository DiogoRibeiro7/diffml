"""Dataset generation for basket option experiments.

This module provides functions to generate training and validation datasets
for basket option pricing using differential machine learning.
Uses Bachelier model for multi-dimensional basket digital options.
"""


from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from diffml.config import DEFAULT_DTYPE, BSParams, get_device

BasketSample = tuple[Tensor, Tensor, Tensor, Tensor]


def make_basket_digital_dataset(
    m: int,
    d: int,
    K: float,
    *,
    params: BSParams | None = None,
    sigma: float | None = None,
    T: float | None = None,
    r: float | None = None,
    w: Tensor | None = None,
    sigma_vec: Tensor | None = None,
    x_min: float = 80.0,
    x_max: float = 120.0,
    n_paths_per_x: int = 10,
    seed: int | None = 1234
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
    params : BSParams | None
        Optional Black-Scholes parameters. If None, sigma and T must be provided.
    sigma : float | None
        Bachelier volatility. Required if params is None.
    T : float | None
        Time to maturity. Required if params is None.
    r : float | None
        Risk-free rate used for discounting. Defaults to 0.0 when params is None.
    w : Tensor | None
        Weights for basket, shape (d,). If None, uses equal weights.
    sigma_vec : Tensor | None
        Volatilities per asset, shape (d,). If None, uses params.sigma for all.
    x_min : float, optional
        Lower bound for initial prices. Default is 80.0.
    x_max : float, optional
        Upper bound for initial prices. Default is 120.0.
    n_paths_per_x : int, optional
        Number of Monte Carlo paths per data point. Default is 10.
    seed : int | None, optional
        Random seed for reproducibility. Default is 1234.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor]
        A tuple containing:
        - x: Initial prices of shape (m, d)
        - price_label: Monte Carlo prices of shape (m, 1)
        - delta_pathwise: Pathwise deltas (zeros) of shape (m, d)
        - delta_lrm: LRM deltas of shape (m, d)

    Raises
    ------
    ValueError
        If m <= 0, d <= 0, or n_paths_per_x <= 0.
        If x_min >= x_max or K <= 0.
        If weights or sigma_vec have incorrect shapes.

    Examples
    --------
    >>> x, prices, delta_pw, delta_lrm = make_basket_digital_dataset(
    ...     m=100, d=5, K=500.0, sigma=0.2, T=0.25, n_paths_per_x=10000
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
    if x_min >= x_max:
        raise ValueError(f"x_min must be less than x_max, got x_min={x_min}, x_max={x_max}")
    if K <= 0:
        raise ValueError(f"Strike K must be positive, got {K}")

    # Get device and set precision
    device = get_device()

    if params is not None:
        if any(value is not None for value in (sigma, T, r)):
            raise ValueError("Provide either params or (sigma, T, r), not both.")
        bs_params = params
    else:
        if sigma is None or T is None:
            raise ValueError("sigma and T must be provided when params is None.")
        rate = 0.0 if r is None else r
        bs_params = BSParams(r=rate, sigma=sigma, T=T)

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Generate initial prices uniformly in [x_min, x_max]
    # Shape: (m, d)
    x = torch.rand(m, d, device=device, dtype=DEFAULT_DTYPE) * (x_max - x_min) + x_min

    # Set weights (equal if not provided)
    if w is None:
        w = torch.ones(d, device=device, dtype=DEFAULT_DTYPE) / d
    else:
        w = w.to(device=device, dtype=DEFAULT_DTYPE)
        if w.shape != (d,):
            raise ValueError(f"Weights must have shape ({d},), got {w.shape}")

    # Set volatilities (use params.sigma if not provided)
    if sigma_vec is None:
        sigma_vec = torch.full((d,), bs_params.sigma, device=device, dtype=DEFAULT_DTYPE)
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
    sqrt_T = torch.sqrt(torch.tensor(bs_params.T, dtype=DEFAULT_DTYPE, device=device))
    S_T = x.unsqueeze(1) + sigma_vec.unsqueeze(0).unsqueeze(0) * sqrt_T * xi

    # Compute basket value: B = sum_i w_i * S_T_i
    # Shape: (m, n_paths_per_x)
    basket = torch.sum(w.unsqueeze(0).unsqueeze(0) * S_T, dim=2)

    # Digital payoff: 1_{B > K}
    # Shape: (m, n_paths_per_x)
    payoff = (basket > K).to(dtype=DEFAULT_DTYPE)

    # Discount factor
    discount = torch.exp(torch.tensor(-bs_params.r * bs_params.T, dtype=DEFAULT_DTYPE, device=device))
    disc_payoff = discount * payoff

    # Price label: Monte Carlo mean
    # Shape: (m, 1)
    price_label = disc_payoff.mean(dim=1, keepdim=True)

    # Pathwise delta: zeros (digital payoff is discontinuous)
    # Shape: (m, d)
    delta_pathwise = torch.zeros(m, d, device=device, dtype=DEFAULT_DTYPE)

    # LRM average delta
    # Score per asset i: xi_i / (sigma_i * sqrt(T))
    # LRM delta per asset: E[payoff * score_i]
    # Average delta: mean over assets
    scores = xi / (sigma_vec.unsqueeze(0).unsqueeze(0) * sqrt_T)  # Shape: (m, n_paths_per_x, d)

    # LRM deltas per asset
    # Shape: (m, d)
    delta_lrm = (disc_payoff.unsqueeze(2) * scores).mean(dim=1)

    return x, price_label, delta_pathwise, delta_lrm


class BasketOptionDataset(Dataset[BasketSample]):
    """PyTorch Dataset for basket option pricing.

    Parameters
    ----------
    x : Tensor
        Initial prices of shape (m, d).
    price_label : Tensor
        Option prices of shape (m, 1).
    delta_pathwise : Tensor
        Pathwise deltas of shape (m, d).
    delta_lrm : Tensor
        LRM deltas of shape (m, d).
    """

    def __init__(
        self,
        x: Tensor,
        price_label: Tensor,
        delta_pathwise: Tensor,
        delta_lrm: Tensor
    ) -> None:
        """Initialize the basket option dataset."""
        self.x = x
        self.price_label = price_label
        self.delta_pathwise = delta_pathwise
        self.delta_lrm = delta_lrm
        self.n_samples = x.shape[0]

    def __len__(self) -> int:
        """Get dataset size.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self.n_samples

    def __getitem__(self, idx: int) -> BasketSample:
        """Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor]
            (x, price, delta_pw, delta_lrm) for the sample.
        """
        return (
            self.x[idx],
            self.price_label[idx],
            self.delta_pathwise[idx],
            self.delta_lrm[idx]
        )


def create_basket_dataloaders(
    n_train: int = 50000,
    n_val: int = 10000,
    batch_size: int = 256,
    n_assets: int = 5,
    K: float = 500.0,
    params: BSParams | None = None,
    n_paths_per_x: int = 10000,
    **kwargs: Any,
) -> tuple[DataLoader[BasketSample], DataLoader[BasketSample]]:
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
    params : BSParams | None
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
    use_params = params is not None
    sigma_kwargs_present = any(key in kwargs for key in ("sigma", "T", "r"))
    if not use_params and not sigma_kwargs_present:
        params = BSParams()
        use_params = True

    # Build keyword arguments for dataset generation
    dataset_kwargs = dict(kwargs)
    if use_params:
        dataset_kwargs["params"] = params

    # Generate training data
    x_train, price_train, delta_pw_train, delta_lrm_train = make_basket_digital_dataset(
        m=n_train,
        d=n_assets,
        K=K,
        n_paths_per_x=n_paths_per_x,
        seed=42,
        **dataset_kwargs
    )

    # Generate validation data
    x_val, price_val, delta_pw_val, delta_lrm_val = make_basket_digital_dataset(
        m=n_val,
        d=n_assets,
        K=K,
        n_paths_per_x=n_paths_per_x,
        seed=123,
        **dataset_kwargs
    )

    # Create datasets
    train_dataset = BasketOptionDataset(
        x_train, price_train, delta_pw_train, delta_lrm_train
    )
    val_dataset = BasketOptionDataset(
        x_val, price_val, delta_pw_val, delta_lrm_val
    )

    # Create dataloaders
    train_loader: DataLoader[BasketSample] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader: DataLoader[BasketSample] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader
