#!/usr/bin/env python
"""Demo script for path-dependent and extended barrier options.

This script demonstrates the new path-dependent option datasets and
extended barrier options added to the DiffML library.
"""

import torch
from torch.utils.data import TensorDataset

from diffml import (
    BSParams,
    PricingNet,
    TrainingConfig,
    make_arithmetic_asian_call_dataset,
    make_lookback_call_dataset,
    make_up_and_out_call_dataset,
    train_model,
)


def demo_asian_option():
    """Demonstrate Asian option pricing with DML."""
    print("\n" + "=" * 60)
    print("ARITHMETIC ASIAN CALL OPTION DEMO")
    print("=" * 60)

    # Parameters
    params = BSParams(r=0.05, sigma=0.3, T=0.5)
    K = 100.0

    # Generate dataset
    print("\nGenerating Asian option dataset...")
    x_train, price_train, delta_pw, delta_lrm = make_arithmetic_asian_call_dataset(
        m=100,
        K=K,
        params=params,
        n_steps=16,
        x_min=0.7,
        x_max=1.3,
        n_paths_per_x=500,
        seed=42
    )

    print(f"  Training samples: {x_train.shape[0]}")
    print(f"  Price range: [{price_train.min():.4f}, {price_train.max():.4f}]")
    print(f"  Delta PW range: [{delta_pw.min():.4f}, {delta_pw.max():.4f}]")
    print(f"  Delta LRM range: [{delta_lrm.min():.4f}, {delta_lrm.max():.4f}]")

    # Create neural network
    model = PricingNet(input_dim=1, hidden_dim=20, n_hidden=3)

    # Training configuration with reduced epochs for demo
    config = TrainingConfig(
        n_epochs=200,
        batch_size=32,
        lr_initial=1e-3,
        lambda_delta=0.5  # Reduced regularization for demo
    )

    # Train with standard ML (no delta) to avoid gradient issues in demo
    print("\nTraining neural network with standard ML...")
    dataset = TensorDataset(x_train, price_train, delta_pw, delta_lrm)
    model = train_model(model, dataset, config, mode="standard")

    # Test the model
    x_test = torch.tensor([[90.0], [100.0], [110.0]], dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        prices = model(x_test)
        print("\nTest predictions:")
        for i, x_val in enumerate(x_test):
            print(f"  S0 = {x_val.item():.0f}: Price = {prices[i].item():.4f}")


def demo_lookback_option():
    """Demonstrate lookback option pricing with DML."""
    print("\n" + "=" * 60)
    print("LOOKBACK CALL OPTION DEMO")
    print("=" * 60)

    # Parameters
    params = BSParams(r=0.05, sigma=0.3, T=0.5)
    K = 100.0

    # Generate dataset
    print("\nGenerating lookback option dataset...")
    x_train, price_train, delta_pw, delta_lrm = make_lookback_call_dataset(
        m=100,
        K=K,
        params=params,
        n_steps=32,
        x_min=0.7,
        x_max=1.3,
        n_paths_per_x=500,
        seed=42
    )

    print(f"  Training samples: {x_train.shape[0]}")
    print(f"  Price range: [{price_train.min():.4f}, {price_train.max():.4f}]")
    print(f"  Delta PW range: [{delta_pw.min():.4f}, {delta_pw.max():.4f}]")
    print(f"  Delta LRM range: [{delta_lrm.min():.4f}, {delta_lrm.max():.4f}]")

    # Compare with Asian option prices
    print("\n  Note: Lookback prices > Asian prices (max >= average)")


def demo_barrier_options():
    """Demonstrate extended barrier options."""
    print("\n" + "=" * 60)
    print("EXTENDED BARRIER OPTIONS DEMO")
    print("=" * 60)

    params = BSParams(r=0.05, sigma=0.2, T=0.5)
    K = 100.0

    # Up-and-out call
    print("\n1. Up-and-Out Call (H = 120)")
    x, price, delta_pw, delta_lrm = make_up_and_out_call_dataset(
        m=50,
        K=K,
        H=120.0,
        params=params,
        n_steps=16,
        x_min=0.6,
        x_max=1.1,
        n_paths_per_x=500,
        seed=123
    )

    # Find prices near ATM
    atm_idx = torch.argmin(torch.abs(x - K))
    print(f"   At S0 = {x[atm_idx].item():.2f}:")
    print(f"     Price = {price[atm_idx].item():.4f}")
    print(f"     Delta (PW) = {delta_pw[atm_idx].item():.4f}")
    print(f"     Delta (LRM) = {delta_lrm[atm_idx].item():.4f}")

    print("\n   Note: Option becomes worthless if S_t >= 120 at any time")


def main():
    """Run all demos."""
    print("\nDiffML Path-Dependent Options Extension Demo")
    print("============================================")

    # Set default dtype
    # Force float64 to mirror training loops and keep Monte Carlo noise low.
    torch.set_default_dtype(torch.float64)

    # Run demos
    demo_asian_option()
    demo_lookback_option()
    demo_barrier_options()

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
