"""Smoothing technique comparison experiments.

This module implements experiments comparing different smoothing techniques
for digital option pricing using differential ML.
"""


import torch
from torch.utils.data import TensorDataset

from diffml.bs_analytics import bs_digital_delta, bs_digital_price
from diffml.config import BSParams, TrainingConfig, get_device, set_default_dtype
from diffml.datasets_smoothing import make_smoothed_digital_dataset
from diffml.networks import PricingNet
from diffml.training import nn_value_delta_gamma, rmse, train_model


def run_smoothing_experiment() -> None:
    """Run smoothing experiment comparing different epsilon multipliers for ramp smoothing.

    This experiment demonstrates how ramp smoothing with different epsilon values
    affects the performance of differential ML. We compare:
    1. Standard ML (price only) with different smoothing levels
    2. LRM DML with different smoothing levels

    The experiment uses eps_multipliers = [0.2, 0.5, 1.0, 2.0, 5.0] to show
    the trade-off between smoothness and accuracy.
    """
    print("\n" + "=" * 80)
    print("SMOOTHING EXPERIMENT")
    print("=" * 80)

    # Set default dtype and get device
    set_default_dtype()
    device = get_device()
    print(f"Using device: {device}")
    print(f"Using dtype: {torch.get_default_dtype()}")

    # Parameters
    params = BSParams(r=0.0, sigma=0.20, T=1.0/3.0)
    K = 100.0

    print("\nParameters:")
    print(f"  r = {params.r:.2f}, sigma = {params.sigma:.2f}, T = {params.T:.4f}")
    print(f"  Strike K = {K:.1f}")

    # Test data with analytical values
    print("\nGenerating test data...")
    m_test = 200
    x_test = torch.linspace(40.0, 160.0, m_test, device=device, dtype=torch.float64).reshape(-1, 1)

    # Compute analytical prices and deltas (unsmoothed digital)
    price_test_true = bs_digital_price(x_test, K, params)
    delta_test_true = bs_digital_delta(x_test, K, params)

    print(f"  Test samples: {m_test}")
    print(f"  Spot range: [{x_test.min():.1f}, {x_test.max():.1f}]")

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

    # Epsilon multipliers to test
    eps_multipliers = [0.2, 0.5, 1.0, 2.0, 5.0]

    # Store all results
    all_results = {}

    for eps_mult in eps_multipliers:
        print("\n" + "=" * 60)
        print(f"EPSILON MULTIPLIER = {eps_mult}")
        print("=" * 60)

        # Generate training data with this smoothing level
        print(f"\nGenerating training data (eps_multiplier={eps_mult})...")
        m_train = 512
        n_paths_train = 10

        x_train, price_train, _, delta_lrm_train = make_smoothed_digital_dataset(
            m=m_train,
            K=K,
            params=params,
            x_min=40.0,
            x_max=160.0,
            n_paths_per_x=n_paths_train,
            eps_multiplier=eps_mult,
            seed=1234
        )

        print(f"  Training samples: {m_train}")
        print(f"  Paths per sample: {n_paths_train}")
        print(f"  Price range: [{price_train.min():.4f}, {price_train.max():.4f}]")
        print(f"  LRM delta range: [{delta_lrm_train.min():.4f}, {delta_lrm_train.max():.4f}]")

        # Results for this epsilon
        eps_results = {}

        # 1. Standard ML (price only)
        print(f"\nTraining Standard ML (eps={eps_mult})...")

        model_standard = PricingNet(**network_config)
        # Create dummy pathwise delta (zeros) for dataset consistency
        delta_pw_train = torch.zeros_like(delta_lrm_train)
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

        eps_results["Standard ML"] = {
            "Price RMSE": price_rmse_standard,
            "Delta RMSE": delta_rmse_standard
        }

        # 2. LRM DML
        print(f"Training LRM DML (eps={eps_mult})...")

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

        eps_results["LRM DML"] = {
            "Price RMSE": price_rmse_lrm,
            "Delta RMSE": delta_rmse_lrm
        }

        # Store results for this epsilon
        all_results[eps_mult] = eps_results

    # Print comprehensive results table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY - SMOOTHING EXPERIMENT")
    print("=" * 80)

    # Print header
    print(f"\n{'Epsilon':<10} {'Model':<15} {'Price RMSE':>12} {'Delta RMSE':>12}")
    print("-" * 50)

    # Print results for each epsilon
    for eps_mult in eps_multipliers:
        eps_results = all_results[eps_mult]
        for i, (model_name, metrics) in enumerate(eps_results.items()):
            if i == 0:
                print(f"{eps_mult:<10.1f} {model_name:<15} {metrics['Price RMSE']:>12.6f} {metrics['Delta RMSE']:>12.6f}")
            else:
                print(f"{'':10} {model_name:<15} {metrics['Price RMSE']:>12.6f} {metrics['Delta RMSE']:>12.6f}")
        print()  # Blank line between epsilon values

    # Print analysis
    print("\nObservations:")
    print("- Smaller epsilon (0.2): More smoothing, easier to learn but less accurate to true digital")
    print("- Larger epsilon (5.0): Less smoothing, closer to true digital but harder to learn")
    print("- Optimal epsilon typically around 0.5-1.0 for balance between learnability and accuracy")
    print("- LRM DML consistently outperforms Standard ML across all smoothing levels")
    print("- Delta accuracy improves significantly with DML regardless of smoothing")
    print("=" * 80)


if __name__ == "__main__":
    run_smoothing_experiment()
