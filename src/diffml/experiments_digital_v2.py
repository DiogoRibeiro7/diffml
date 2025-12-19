"""Digital option pricing experiments with configuration support.

This module implements the digital option experiments from the paper,
demonstrating differential ML for discontinuous payoffs, now with
configuration-based setup.
"""


import torch
from torch.utils.data import TensorDataset

from diffml.bs_analytics import bs_digital_delta, bs_digital_price
from diffml.config import BSParams, get_device, set_default_dtype
from diffml.config_experiments import ExperimentConfig
from diffml.datasets_digital import make_digital_dataset
from diffml.experiments_registry import register_experiment
from diffml.networks import PricingNet
from diffml.training import nn_value_delta_gamma, rmse, train_model


@register_experiment("digital")
def run_digital_experiment(config: ExperimentConfig) -> None:
    """Run digital option pricing experiment with configuration.

    This experiment demonstrates how differential ML improves learning of digital
    option prices and deltas. We compare:
    1. Standard ML (price only)
    2. Pathwise DML (using zero pathwise deltas)
    3. LRM DML (using likelihood ratio method deltas)

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration containing all parameters.
    """
    print("\n" + "=" * 80)
    print("DIGITAL OPTION EXPERIMENT (Configured)")
    print("=" * 80)

    # Set default dtype and get device
    set_default_dtype()
    device = get_device()
    print(f"Using device: {device}")
    print(f"Using dtype: {torch.get_default_dtype()}")
    print(f"Random seed: {config.seed}")

    # Set random seed
    torch.manual_seed(config.seed)

    # Black-Scholes parameters
    params = BSParams(r=config.r, sigma=config.sigma, T=config.T)
    K = config.K if config.K is not None else 100.0

    print("\nParameters:")
    print(f"  r = {params.r:.2f}, sigma = {params.sigma:.2f}, T = {params.T:.4f}")
    print(f"  Strike K = {K:.1f}")

    # Training data
    print("\nGenerating training data...")
    x_train, price_train, delta_pw_train, delta_lrm_train = make_digital_dataset(
        m=config.m_train,
        K=K,
        params=params,
        x_min=config.x_min * K,
        x_max=config.x_max * K,
        n_paths_per_x=config.n_paths_train,
        seed=config.seed
    )

    print(f"  Training samples: {config.m_train}")
    print(f"  Paths per sample: {config.n_paths_train}")
    print(f"  Price range: [{price_train.min():.4f}, {price_train.max():.4f}]")
    print(f"  LRM delta range: [{delta_lrm_train.min():.4f}, {delta_lrm_train.max():.4f}]")

    # Test data with analytical values
    print("\nGenerating test data...")
    x_test = torch.linspace(
        config.x_min * K, config.x_max * K, config.m_test,
        device=device, dtype=torch.float64
    ).reshape(-1, 1)

    # Compute analytical prices and deltas
    price_test_true = bs_digital_price(x_test, K, params)
    delta_test_true = bs_digital_delta(x_test, K, params)

    print(f"  Test samples: {config.m_test}")
    print(f"  Spot range: [{x_test.min():.1f}, {x_test.max():.1f}]")

    # Network architecture
    network_config = {
        "input_dim": 1,
        "hidden_dim": 20,
        "n_hidden": 4
    }

    # Store results
    results = {}

    # Train three models with different approaches
    approaches = [
        ("Standard ML", "standard", 0.0),
        ("Pathwise DML", "delta_pathwise", config.training.lambda_delta),
        ("LRM DML", "delta_lrm", config.training.lambda_delta)
    ]

    for name, mode, lambda_delta in approaches:
        print("\n" + "-" * 60)
        print(f"Training: {name}")
        print(f"  Mode: {mode}")
        print(f"  Lambda_delta: {lambda_delta:.2f}")

        # Create fresh model
        model = PricingNet(**network_config)

        # Update training config for this approach
        train_config = config.training
        train_config.lambda_delta = lambda_delta

        # Prepare dataset
        if mode == "delta_pathwise":
            dataset = TensorDataset(x_train, price_train, delta_pw_train, delta_lrm_train)
        else:
            dataset = TensorDataset(x_train, price_train, delta_pw_train, delta_lrm_train)

        # Train model
        model = train_model(model, dataset, train_config, mode=mode)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            x_test_grad = x_test.clone().requires_grad_(True)
            price_pred, delta_pred, _ = nn_value_delta_gamma(
                model, x_test_grad, compute_delta=True, compute_gamma=False
            )

        # Compute errors
        price_rmse = rmse(price_pred, price_test_true)
        delta_rmse = rmse(delta_pred, delta_test_true)

        print("\nTest Results:")
        print(f"  Price RMSE: {price_rmse:.4f}")
        print(f"  Delta RMSE: {delta_rmse:.4f}")

        results[name] = {
            "price_rmse": price_rmse,
            "delta_rmse": delta_rmse,
            "model": model
        }

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Price RMSE':<15} {'Delta RMSE':<15}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<20} {res['price_rmse']:<15.4f} {res['delta_rmse']:<15.4f}")

    # Show improvement ratios
    if "Standard ML" in results and "LRM DML" in results:
        std_delta = results["Standard ML"]["delta_rmse"]
        lrm_delta = results["LRM DML"]["delta_rmse"]
        improvement = (std_delta - lrm_delta) / std_delta * 100
        print(f"\nDelta RMSE improvement (LRM vs Standard): {improvement:.1f}%")

    print("\nExperiment completed successfully!")


@register_experiment("barrier")
def run_barrier_experiment(config: ExperimentConfig) -> None:
    """Run barrier option experiment with configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration containing all parameters.
    """
    print("\n" + "=" * 80)
    print("BARRIER OPTION EXPERIMENT (Configured)")
    print("=" * 80)

    # Set defaults
    set_default_dtype()
    torch.manual_seed(config.seed)

    # Parameters
    params = BSParams(r=config.r, sigma=config.sigma, T=config.T)
    K = config.K if config.K is not None else 1.0
    B = config.B if config.B is not None else 0.85

    print("\nParameters:")
    print(f"  Black-Scholes: r={params.r:.2f}, sigma={params.sigma:.2f}, T={params.T:.4f}")
    print(f"  Strike K = {K:.2f}")
    print(f"  Barrier B = {B:.2f}")

    # Import here to avoid circular dependency
    from diffml.datasets_barrier import make_barrier_dataset

    # Generate training data
    print("\nGenerating training data...")
    x_train, price_train, delta_pw_train, delta_lrm_train = make_barrier_dataset(
        m=config.m_train,
        K=K,
        B=B,
        params=params,
        x_min=config.x_min,
        x_max=config.x_max,
        n_paths_per_x=config.n_paths_train,
        seed=config.seed
    )

    print(f"  Training samples: {config.m_train}")
    print(f"  Price range: [{price_train.min():.4f}, {price_train.max():.4f}]")

    # Create and train model
    model = PricingNet(input_dim=1, hidden_dim=20, n_hidden=4)
    dataset = TensorDataset(x_train, price_train, delta_pw_train, delta_lrm_train)

    # Use pathwise mode for barrier options
    model = train_model(model, dataset, config.training, mode="delta_pathwise")

    print("\nBarrier option experiment completed!")


@register_experiment("basket")
def run_basket_experiment(config: ExperimentConfig) -> None:
    """Run basket digital option experiment with configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration containing all parameters.
    """
    print("\n" + "=" * 80)
    print("BASKET DIGITAL OPTION EXPERIMENT (Configured)")
    print("=" * 80)

    # Set defaults
    set_default_dtype()
    torch.manual_seed(config.seed)

    # Parameters
    params = BSParams(r=config.r, sigma=config.sigma, T=config.T)
    K = config.K if config.K is not None else 100.0
    d = config.d if config.d is not None else 20

    print("\nParameters:")
    print(f"  Black-Scholes: r={params.r:.2f}, sigma={params.sigma:.2f}, T={params.T:.4f}")
    print(f"  Strike K = {K:.1f}")
    print(f"  Dimension d = {d}")

    # Import here to avoid circular dependency
    from diffml.datasets_basket import make_basket_digital_dataset

    # Generate training data
    print("\nGenerating training data...")
    x_train, price_train, delta_pw_train = make_basket_digital_dataset(
        m=config.m_train,
        d=d,
        K=K,
        params=params,
        x_min=config.x_min * K,
        x_max=config.x_max * K,
        n_paths_per_x=config.n_paths_train,
        seed=config.seed
    )

    print(f"  Training samples: {config.m_train}")
    print(f"  Input dimension: {d}")
    print(f"  Price range: [{price_train.min():.4f}, {price_train.max():.4f}]")

    # Create and train model
    model = PricingNet(input_dim=d, hidden_dim=40, n_hidden=4)

    # For basket options, we only have pathwise delta
    delta_lrm_dummy = torch.zeros_like(delta_pw_train)
    dataset = TensorDataset(x_train, price_train, delta_pw_train, delta_lrm_dummy)

    model = train_model(model, dataset, config.training, mode="delta_pathwise")

    print("\nBasket option experiment completed!")


@register_experiment("smoothing")
def run_smoothing_experiment(config: ExperimentConfig) -> None:
    """Run smoothing experiment with configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration containing all parameters.
    """
    print("\n" + "=" * 80)
    print("SMOOTHING EXPERIMENT (Configured)")
    print("=" * 80)

    # Set defaults
    set_default_dtype()
    torch.manual_seed(config.seed)

    # Parameters
    params = BSParams(r=config.r, sigma=config.sigma, T=config.T)
    K = config.K if config.K is not None else 100.0
    eps_multipliers = config.eps_multipliers or [0.2, 0.5, 1.0, 2.0, 5.0]

    print("\nParameters:")
    print(f"  Black-Scholes: r={params.r:.2f}, sigma={params.sigma:.2f}, T={params.T:.4f}")
    print(f"  Strike K = {K:.1f}")
    print(f"  Epsilon multipliers: {eps_multipliers}")

    # Import here
    from diffml.datasets_smoothing import make_smoothed_digital_dataset

    results = {}

    for eps_mult in eps_multipliers:
        print("\n" + "-" * 60)
        print(f"Training with epsilon multiplier = {eps_mult}")

        # Generate data with this smoothing parameter
        x_train, price_train, delta_pw_train, delta_lrm_train = make_smoothed_digital_dataset(
            m=config.m_train,
            K=K,
            params=params,
            eps_multiplier=eps_mult,
            x_min=config.x_min * K,
            x_max=config.x_max * K,
            n_paths_per_x=config.n_paths_train,
            seed=config.seed
        )

        # Train model
        model = PricingNet(input_dim=1, hidden_dim=20, n_hidden=4)
        dataset = TensorDataset(x_train, price_train, delta_pw_train, delta_lrm_train)
        model = train_model(model, dataset, config.training, mode="delta_pathwise")

        results[eps_mult] = model

        print(f"  Training completed for eps_mult = {eps_mult}")

    print("\n" + "=" * 80)
    print(f"Trained {len(results)} models with different smoothing parameters")
    print("\nSmoothing experiment completed!")


@register_experiment("asian")
def run_asian_experiment(config: ExperimentConfig) -> None:
    """Run arithmetic Asian option experiment with configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration containing all parameters.
    """
    print("\n" + "=" * 80)
    print("ASIAN OPTION EXPERIMENT (Configured)")
    print("=" * 80)

    # Set defaults
    set_default_dtype()
    torch.manual_seed(config.seed)

    # Parameters
    params = BSParams(r=config.r, sigma=config.sigma, T=config.T)
    K = config.K if config.K is not None else 100.0
    n_steps = config.n_steps if config.n_steps is not None else 16

    print("\nParameters:")
    print(f"  Black-Scholes: r={params.r:.2f}, sigma={params.sigma:.2f}, T={params.T:.4f}")
    print(f"  Strike K = {K:.1f}")
    print(f"  Time steps = {n_steps}")

    # Import here
    from diffml.datasets_path_dependent import make_arithmetic_asian_call_dataset

    # Generate training data
    print("\nGenerating training data...")
    x_train, price_train, delta_pw_train, delta_lrm_train = make_arithmetic_asian_call_dataset(
        m=config.m_train,
        K=K,
        params=params,
        n_steps=n_steps,
        x_min=config.x_min,
        x_max=config.x_max,
        n_paths_per_x=config.n_paths_train,
        seed=config.seed
    )

    print(f"  Training samples: {config.m_train}")
    print(f"  Price range: [{price_train.min():.4f}, {price_train.max():.4f}]")
    print(f"  Delta PW range: [{delta_pw_train.min():.4f}, {delta_pw_train.max():.4f}]")

    # Create and train model
    model = PricingNet(input_dim=1, hidden_dim=30, n_hidden=4)
    dataset = TensorDataset(x_train, price_train, delta_pw_train, delta_lrm_train)

    # Train with standard ML to avoid gradient issues
    model = train_model(model, dataset, config.training, mode="standard")

    print("\nAsian option experiment completed!")
