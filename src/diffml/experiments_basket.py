"""Basket digital option pricing experiments.

This module implements the basket digital option experiments from the paper,
demonstrating differential ML with Bachelier model on high-dimensional inputs.
"""


import torch
from torch.utils.data import TensorDataset

from diffml.config import TrainingConfig, get_device, set_default_dtype
from diffml.datasets_basket import make_basket_digital_dataset
from diffml.networks import PricingNet
from diffml.training import nn_value_delta_gamma, rmse, train_model


def run_basket_digital_experiment() -> None:
    """Run basket digital option pricing experiment comparing standard ML with DML approaches.

    This experiment demonstrates differential ML on high-dimensional basket digital options
    using the Bachelier model. We compare:
    1. Standard ML (price only)
    2. Pathwise DML (using zero pathwise deltas)
    3. LRM DML (using likelihood ratio method deltas)

    The experiment uses a basket digital option with 20 underlying assets,
    showcasing how DML scales to higher dimensions.
    """
    print("\n" + "=" * 80)
    print("BASKET DIGITAL OPTION EXPERIMENT")
    print("=" * 80)

    # Set default dtype and get device
    set_default_dtype()
    device = get_device()
    print(f"Using device: {device}")
    print(f"Using dtype: {torch.get_default_dtype()}")

    # Parameters for 20-dimensional basket digital
    d = 20  # Number of assets
    K = 1.0  # Strike price for digital payoff
    sigma_bachelier = 0.2  # Bachelier volatility
    T = 1.0 / 3.0  # Maturity

    print("\nParameters:")
    print(f"  Dimension d = {d}")
    print(f"  Strike K = {K:.1f}")
    print(f"  Bachelier sigma = {sigma_bachelier:.2f}, T = {T:.4f}")

    # Training data
    print("\nGenerating training data...")
    m_train = 1024
    n_paths_train = 10

    x_train, price_train, delta_pw_train, delta_lrm_train = make_basket_digital_dataset(
        m=m_train,
        d=d,
        K=K,
        sigma=sigma_bachelier,
        T=T,
        x_min=0.5,
        x_max=1.5,
        n_paths_per_x=n_paths_train,
        seed=1234
    )

    print(f"  Training samples: {m_train}")
    print(f"  Input dimension: {d}")
    print(f"  Paths per sample: {n_paths_train}")
    print(f"  Price range: [{price_train.min():.4f}, {price_train.max():.4f}]")
    print(f"  Pathwise delta norm range: [{delta_pw_train.norm(dim=1).min():.4f}, {delta_pw_train.norm(dim=1).max():.4f}]")
    print(f"  LRM delta norm range: [{delta_lrm_train.norm(dim=1).min():.4f}, {delta_lrm_train.norm(dim=1).max():.4f}]")

    # Test data with large-sample Monte Carlo for "true" values
    print("\nGenerating test data (large-sample Monte Carlo)...")
    m_test = 200
    n_paths_test = 10000  # Large sample for accurate "true" values

    x_test, price_test_true, _, delta_lrm_test_true = make_basket_digital_dataset(
        m=m_test,
        d=d,
        K=K,
        sigma=sigma_bachelier,
        T=T,
        x_min=0.5,
        x_max=1.5,
        n_paths_per_x=n_paths_test,
        seed=5678
    )

    # Use LRM deltas as "true" deltas (more stable than pathwise for digitals)
    delta_test_true = delta_lrm_test_true

    print(f"  Test samples: {m_test}")
    print(f"  Paths per sample: {n_paths_test} (for accurate estimates)")
    print(f"  Input range: [{x_test.min():.2f}, {x_test.max():.2f}]")

    # Training configuration
    config = TrainingConfig(
        n_epochs=2000,
        batch_size=256,
        lr_initial=1e-3,
        lr_min=1e-6,
        lambda_delta=0.0,  # Will be set per model
        lambda_gamma=0.0
    )

    # Network architecture (larger for high-dimensional input)
    network_config = {
        "input_dim": d,
        "hidden_dim": 40,
        "n_hidden": 4
    }

    # Store results
    results = {}

    # 1. Standard ML (price only)
    print("\n" + "-" * 60)
    print("Training Standard ML model (price only)...")
    print("-" * 60)

    model_standard = PricingNet(**network_config)
    # For standard ML, we still need to provide delta data but won't use it
    dataset_standard = TensorDataset(x_train, price_train, delta_pw_train, delta_lrm_train)

    config.lambda_delta = 0.0
    model_standard = train_model(
        model=model_standard,
        dataset=dataset_standard,
        config=config,
        mode="standard",
        device=device
    )

    # Evaluate standard model
    with torch.no_grad():
        pred_price_standard, pred_delta_standard, _ = nn_value_delta_gamma(
            model_standard, x_test, compute_delta=True, compute_gamma=False
        )

    price_rmse_standard = rmse(pred_price_standard, price_test_true)
    # For multi-dimensional delta, compute RMSE over all components
    delta_rmse_standard = rmse(pred_delta_standard, delta_test_true)

    results["Standard ML"] = {
        "Price RMSE": price_rmse_standard,
        "Delta RMSE": delta_rmse_standard
    }

    # 2. Pathwise DML
    print("\n" + "-" * 60)
    print("Training Pathwise DML model...")
    print("-" * 60)

    model_pathwise = PricingNet(**network_config)
    dataset_pathwise = TensorDataset(x_train, price_train, delta_pw_train, delta_lrm_train)

    config.lambda_delta = 1.0
    model_pathwise = train_model(
        model=model_pathwise,
        dataset=dataset_pathwise,
        config=config,
        mode="delta_pathwise",
        device=device
    )

    # Evaluate pathwise model
    with torch.no_grad():
        pred_price_pathwise, pred_delta_pathwise, _ = nn_value_delta_gamma(
            model_pathwise, x_test, compute_delta=True, compute_gamma=False
        )

    price_rmse_pathwise = rmse(pred_price_pathwise, price_test_true)
    delta_rmse_pathwise = rmse(pred_delta_pathwise, delta_test_true)

    results["Pathwise DML"] = {
        "Price RMSE": price_rmse_pathwise,
        "Delta RMSE": delta_rmse_pathwise
    }

    # 3. LRM DML
    print("\n" + "-" * 60)
    print("Training LRM DML model...")
    print("-" * 60)

    model_lrm = PricingNet(**network_config)
    dataset_lrm = TensorDataset(x_train, price_train, delta_pw_train, delta_lrm_train)

    config.lambda_delta = 1.0
    model_lrm = train_model(
        model=model_lrm,
        dataset=dataset_lrm,
        config=config,
        mode="delta_lrm",
        device=device
    )

    # Evaluate LRM model
    with torch.no_grad():
        pred_price_lrm, pred_delta_lrm, _ = nn_value_delta_gamma(
            model_lrm, x_test, compute_delta=True, compute_gamma=False
        )

    price_rmse_lrm = rmse(pred_price_lrm, price_test_true)
    delta_rmse_lrm = rmse(pred_delta_lrm, delta_test_true)

    results["LRM DML"] = {
        "Price RMSE": price_rmse_lrm,
        "Delta RMSE": delta_rmse_lrm
    }

    # Print results table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Price RMSE':>15} {'Delta RMSE':>15}")
    print("-" * 50)

    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['Price RMSE']:>15.6f} {metrics['Delta RMSE']:>15.6f}")

    print("\nObservations:")
    print("- Standard ML learns prices but has poor delta estimates in high dimensions")
    print("- Pathwise DML with zero deltas may degrade performance")
    print("- LRM DML provides better delta estimates for discontinuous payoffs")
    print("- Higher dimensions (d=20) require larger networks and more training")
    print("=" * 80)


if __name__ == "__main__":
    run_basket_digital_experiment()
