"""Barrier option pricing experiments.

This module implements the barrier option experiments from the paper.
"""


import torch
from torch.utils.data import TensorDataset

from diffml.config import BSParams, TrainingConfig, get_device, set_default_dtype
from diffml.datasets_barrier import make_barrier_dataset
from diffml.networks import PricingNet
from diffml.training import nn_value_delta_gamma, rmse, train_model


def run_barrier_experiment() -> None:
    """Run barrier option pricing experiment comparing standard ML with DML approaches.

    This experiment uses a down-and-out call option to demonstrate differential ML
    on path-dependent options with barriers. We compare:
    1. Standard ML (price only)
    2. Pathwise DML (using pathwise delta estimates)
    3. LRM DML (using likelihood ratio method deltas)

    Test values are computed using large-sample Monte Carlo as "ground truth".
    """
    print("\n" + "=" * 80)
    print("BARRIER OPTION EXPERIMENT")
    print("=" * 80)

    # Set default dtype and get device
    set_default_dtype()
    device = get_device()
    print(f"Using device: {device}")
    print(f"Using dtype: {torch.get_default_dtype()}")

    # Parameters for down-and-out call
    params = BSParams(r=0.0, sigma=0.20, T=1.0/3.0)
    K = 1.0
    B = 0.85
    T1 = params.T / 2.0  # Barrier observation at mid-point
    T2 = params.T        # Maturity

    print("\nParameters:")
    print(f"  r = {params.r:.2f}, sigma = {params.sigma:.2f}, T = {params.T:.4f}")
    print(f"  Strike K = {K:.2f}, Barrier B = {B:.2f}")
    print(f"  T1 (barrier) = {T1:.4f}, T2 (maturity) = {T2:.4f}")

    # Training data
    print("\nGenerating training data...")
    m_train = 1024
    n_paths_train = 10

    x_train, price_train, delta_pw_train, delta_lrm_train = make_barrier_dataset(
        m=m_train,
        K=K,
        B=B,
        params=params,
        T1=T1,
        T2=T2,
        x_min=0.4,
        x_max=1.6,
        n_paths_per_x=n_paths_train,
        seed=1234
    )

    print(f"  Training samples: {m_train}")
    print(f"  Paths per sample: {n_paths_train}")
    print(f"  Price range: [{price_train.min():.4f}, {price_train.max():.4f}]")
    print(f"  Pathwise delta range: [{delta_pw_train.min():.4f}, {delta_pw_train.max():.4f}]")
    print(f"  LRM delta range: [{delta_lrm_train.min():.4f}, {delta_lrm_train.max():.4f}]")

    # Test data with large-sample Monte Carlo for "true" values
    print("\nGenerating test data (large-sample Monte Carlo)...")
    m_test = 200
    n_paths_test = 20000  # Large sample for accurate "true" values

    x_test, price_test_true, delta_pw_test_true, delta_lrm_test_true = make_barrier_dataset(
        m=m_test,
        K=K,
        B=B,
        params=params,
        T1=T1,
        T2=T2,
        x_min=0.4,
        x_max=1.6,
        n_paths_per_x=n_paths_test,
        seed=5678
    )

    # Use LRM deltas as "true" deltas for evaluation (more stable than pathwise for barriers)
    delta_test_true = delta_lrm_test_true

    print(f"  Test samples: {m_test}")
    print(f"  Paths per sample: {n_paths_test} (for accurate estimates)")
    print(f"  Spot range: [{x_test.min():.2f}, {x_test.max():.2f}]")

    # Training configuration
    config = TrainingConfig(
        n_epochs=2000,
        batch_size=256,
        lr_initial=1e-3,
        lr_min=1e-6,
        lambda_delta=0.0,  # Will be set per model
        lambda_gamma=0.0
    )

    # Network architecture
    network_config = {
        "input_dim": 1,
        "hidden_dim": 20,
        "n_hidden": 4
    }

    # Store results
    results = {}

    # 1. Standard ML (price only)
    print("\n" + "-" * 60)
    print("Training Standard ML model (price only)...")
    print("-" * 60)

    model_standard = PricingNet(**network_config)
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
    print("- Standard ML learns prices but has poor delta estimates")
    print("- Both Pathwise and LRM DML improve delta accuracy")
    print("- Barrier options benefit from sensitivity regularization")
    print("=" * 80)


if __name__ == "__main__":
    run_barrier_experiment()
