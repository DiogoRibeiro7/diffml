#!/usr/bin/env python
"""Comprehensive experiment runner for DiffML.

This script runs all experiments, compares different option types,
and generates comprehensive reports.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffml.unified_config import create_default_config, UnifiedExperimentConfig
from diffml.experiment_manager import ExperimentManager
from diffml.networks import FeedForwardNet, DifferentialNet
from diffml.training import train_model
from diffml.config import BSParams, TrainingConfig
from diffml.losses import dml_loss

# Import datasets
from diffml.datasets_digital import generate_digital_dataset_train
from diffml.datasets_barrier import generate_barrier_dataset_with_paths
from diffml.datasets_american import generate_american_option_dataset
from diffml.datasets_multibarrier import (
    generate_multibarrier_dataset,
    MultiBarrierParams,
    BarrierType
)


class ComprehensiveExperimentRunner:
    """Run and compare all DiffML experiments."""

    def __init__(self, output_dir: str = "experiment_results"):
        """Initialize experiment runner.

        Parameters:
            output_dir: Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run_digital_experiment(self, config: UnifiedExperimentConfig) -> Dict[str, Any]:
        """Run digital option experiment.

        Parameters:
            config: Experiment configuration

        Returns:
            Experiment results
        """
        print("\n" + "="*60)
        print("Running Digital Option Experiment")
        print("="*60)

        # Generate dataset
        S, K, prices, deltas = generate_digital_dataset_train(
            m=config.m_train,
            n_paths=config.n_paths_train,
            r=config.r,
            sigma=config.sigma,
            T=config.T,
            x_min=config.x_min,
            x_max=config.x_max
        )

        # Create features
        features = torch.stack([
            S, K,
            torch.full_like(S, config.r),
            torch.full_like(S, config.sigma),
            torch.full_like(S, config.T)
        ], dim=1)

        # Split data
        train_size = int(0.8 * len(features))
        train_features = features[:train_size]
        train_prices = prices[:train_size]
        train_deltas = deltas[:train_size]

        val_features = features[train_size:]
        val_prices = prices[train_size:]
        val_deltas = deltas[train_size:]

        # Create model
        model = DifferentialNet(
            input_dim=5,
            hidden_dims=[64, 32, 16],
            output_dim=1
        ).to(self.device)

        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr_initial)
        train_losses = []
        val_losses = []

        print("Training...")
        start_time = time.time()

        for epoch in range(config.training.n_epochs):
            # Training step
            model.train()
            optimizer.zero_grad()

            train_pred = model(train_features.to(self.device))
            train_loss = dml_loss(
                train_pred,
                train_prices.to(self.device),
                train_deltas.to(self.device),
                train_features.to(self.device),
                lambda_val=config.training.lambda_delta
            )

            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

            # Validation step
            model.eval()
            with torch.no_grad():
                val_pred = model(val_features.to(self.device))
                val_loss = dml_loss(
                    val_pred,
                    val_prices.to(self.device),
                    val_deltas.to(self.device),
                    val_features.to(self.device),
                    lambda_val=config.training.lambda_delta
                )
                val_losses.append(val_loss.item())

            if epoch % 100 == 0:
                print(f"  Epoch {epoch}: Train Loss = {train_loss.item():.6f}, "
                      f"Val Loss = {val_loss.item():.6f}")

        training_time = time.time() - start_time

        # Generate test data
        S_test, K_test, prices_test, deltas_test = generate_digital_dataset_train(
            m=config.m_test,
            n_paths=config.n_paths_test,
            r=config.r,
            sigma=config.sigma,
            T=config.T,
            x_min=config.x_min,
            x_max=config.x_max
        )

        features_test = torch.stack([
            S_test, K_test,
            torch.full_like(S_test, config.r),
            torch.full_like(S_test, config.sigma),
            torch.full_like(S_test, config.T)
        ], dim=1)

        # Test evaluation
        model.eval()
        with torch.no_grad():
            test_pred = model(features_test.to(self.device))
            test_loss = dml_loss(
                test_pred,
                prices_test.to(self.device),
                deltas_test.to(self.device),
                features_test.to(self.device),
                lambda_val=config.training.lambda_delta
            ).item()

        print(f"✅ Digital experiment completed in {training_time:.2f}s")
        print(f"   Final test loss: {test_loss:.6f}")

        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_loss': test_loss,
            'training_time': training_time,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }

    def run_barrier_experiment(self, config: UnifiedExperimentConfig) -> Dict[str, Any]:
        """Run barrier option experiment.

        Parameters:
            config: Experiment configuration

        Returns:
            Experiment results
        """
        print("\n" + "="*60)
        print("Running Barrier Option Experiment")
        print("="*60)

        # Generate dataset
        dataset = generate_barrier_dataset_with_paths(
            m=config.m_train,
            n_paths=config.n_paths_train,
            n_steps=config.n_steps or 100,
            r=config.r,
            sigma=config.sigma,
            T=config.T,
            x_min=config.x_min,
            x_max=config.x_max,
            barrier_level=config.B or 0.9
        )

        S, K, B, prices, deltas = dataset

        # Create features (7-dimensional for barrier)
        features = torch.stack([
            S, K, B,
            torch.full_like(S, config.r),
            torch.full_like(S, config.sigma),
            torch.full_like(S, config.T),
            torch.zeros_like(S)  # Rebate
        ], dim=1)

        # Create and train model
        model = FeedForwardNet(
            input_dim=7,
            hidden_dims=[64, 32, 16],
            output_dim=1
        ).to(self.device)

        # Training loop (simplified)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr_initial)
        losses = []

        print("Training barrier model...")
        start_time = time.time()

        for epoch in range(min(config.training.n_epochs, 500)):  # Limit epochs for demo
            optimizer.zero_grad()
            predictions = model(features.to(self.device))
            loss = torch.nn.MSELoss()(predictions.squeeze(), prices.to(self.device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if epoch % 100 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

        training_time = time.time() - start_time

        print(f"✅ Barrier experiment completed in {training_time:.2f}s")

        return {
            'model': model,
            'losses': losses,
            'training_time': training_time,
            'final_loss': losses[-1]
        }

    def run_american_experiment(self, config: UnifiedExperimentConfig) -> Dict[str, Any]:
        """Run American option experiment.

        Parameters:
            config: Experiment configuration

        Returns:
            Experiment results
        """
        print("\n" + "="*60)
        print("Running American Option Experiment")
        print("="*60)

        from diffml.datasets_american import AmericanOptionParams

        # Generate dataset
        bs_params = BSParams(r=config.r, sigma=config.sigma, T=config.T)
        american_params = AmericanOptionParams(
            exercise_type='put',
            n_basis=4,
            basis_type='laguerre'
        )

        print("Generating American option dataset...")
        features, prices, deltas, gammas = generate_american_option_dataset(
            bs_params=bs_params,
            n_samples=min(config.m_train, 1000),  # Limit for demo
            american_params=american_params,
            compute_greeks=True
        )

        # Create and train model
        model = DifferentialNet(
            input_dim=5,
            hidden_dims=[64, 32, 16],
            output_dim=1
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr_initial)
        losses = []

        print("Training American option model...")
        start_time = time.time()

        for epoch in range(min(config.training.n_epochs, 200)):  # Limit for demo
            optimizer.zero_grad()
            predictions = model(features.to(self.device))
            loss = torch.nn.MSELoss()(predictions.squeeze(), prices.to(self.device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if epoch % 50 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

        training_time = time.time() - start_time

        print(f"✅ American experiment completed in {training_time:.2f}s")

        return {
            'model': model,
            'losses': losses,
            'training_time': training_time,
            'final_loss': losses[-1],
            'mean_price': prices.mean().item(),
            'mean_delta': deltas.mean().item()
        }

    def run_multibarrier_experiment(self, config: UnifiedExperimentConfig) -> Dict[str, Any]:
        """Run multi-barrier option experiment.

        Parameters:
            config: Experiment configuration

        Returns:
            Experiment results
        """
        print("\n" + "="*60)
        print("Running Multi-Barrier Option Experiment")
        print("="*60)

        # Configure multi-barrier
        barrier_params = MultiBarrierParams(
            barrier_type=BarrierType.DOUBLE_OUT,
            lower_barrier=config.L or 85.0,
            upper_barrier=config.H or 115.0,
            rebate=0.0
        )

        bs_params = BSParams(r=config.r, sigma=config.sigma, T=config.T)

        print("Generating multi-barrier dataset...")
        features, prices, deltas, gammas = generate_multibarrier_dataset(
            bs_params=bs_params,
            n_samples=min(config.m_train, 500),  # Limit for demo
            barrier_params=barrier_params,
            n_paths=1000,
            n_steps=50,
            compute_greeks=True
        )

        # Create and train model
        model = FeedForwardNet(
            input_dim=5,
            hidden_dims=[32, 16],
            output_dim=1
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr_initial)
        losses = []

        print("Training multi-barrier model...")
        start_time = time.time()

        for epoch in range(min(config.training.n_epochs, 100)):  # Limit for demo
            optimizer.zero_grad()
            predictions = model(features.to(self.device))
            loss = torch.nn.MSELoss()(predictions.squeeze(), prices.to(self.device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if epoch % 25 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

        training_time = time.time() - start_time

        print(f"✅ Multi-barrier experiment completed in {training_time:.2f}s")

        return {
            'model': model,
            'losses': losses,
            'training_time': training_time,
            'final_loss': losses[-1],
            'mean_price': prices.mean().item()
        }

    def compare_convergence(self, results: Dict[str, Dict]) -> None:
        """Compare convergence across experiments.

        Parameters:
            results: Dictionary of experiment results
        """
        plt.figure(figsize=(12, 6))

        for name, result in results.items():
            if 'losses' in result:
                losses = result['losses']
            elif 'train_losses' in result:
                losses = result['train_losses']
            else:
                continue

            plt.semilogy(losses, label=name, alpha=0.8)

        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Convergence Comparison Across Option Types')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = self.output_dir / 'convergence_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Convergence plot saved to {output_path}")

    def generate_report(self, results: Dict[str, Dict]) -> None:
        """Generate comprehensive experiment report.

        Parameters:
            results: Dictionary of experiment results
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'experiments': {}
        }

        # Summary table
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"{'Experiment':<20} {'Final Loss':<12} {'Time (s)':<10} {'Status'}")
        print("-"*60)

        for name, result in results.items():
            final_loss = result.get('final_loss', result.get('test_loss', 'N/A'))
            training_time = result.get('training_time', 'N/A')

            if isinstance(final_loss, float):
                loss_str = f"{final_loss:.6f}"
            else:
                loss_str = str(final_loss)

            if isinstance(training_time, float):
                time_str = f"{training_time:.2f}"
            else:
                time_str = str(training_time)

            print(f"{name:<20} {loss_str:<12} {time_str:<10} ✅")

            report['experiments'][name] = {
                'final_loss': final_loss,
                'training_time': training_time,
                'parameters': {
                    'model_params': result.get('model', 'N/A').__class__.__name__,
                    'mean_price': result.get('mean_price', 'N/A')
                }
            }

        # Save report
        report_path = self.output_dir / 'experiment_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n✅ Report saved to {report_path}")

    def run_all_experiments(self) -> Dict[str, Dict]:
        """Run all experiments.

        Returns:
            Dictionary of all results
        """
        print("\n🔬 Running Comprehensive DiffML Experiments")
        print("="*60)

        # Create configurations
        configs = {
            'digital': create_default_config('digital'),
            'barrier': create_default_config('barrier'),
            'american': create_default_config('american'),
            'multibarrier': create_default_config('multibarrier')
        }

        # Adjust for faster demo
        for config in configs.values():
            config.m_train = min(config.m_train, 1000)
            config.m_test = min(config.m_test, 200)
            config.training.n_epochs = min(config.training.n_epochs, 500)
            config.n_paths_train = min(config.n_paths_train, 1000)
            config.n_paths_test = min(config.n_paths_test, 5000)

        # Run experiments
        results = {}

        # Digital option
        results['digital'] = self.run_digital_experiment(configs['digital'])

        # Barrier option
        results['barrier'] = self.run_barrier_experiment(configs['barrier'])

        # American option
        results['american'] = self.run_american_experiment(configs['american'])

        # Multi-barrier option
        results['multibarrier'] = self.run_multibarrier_experiment(configs['multibarrier'])

        return results

    def run_performance_comparison(self) -> None:
        """Run performance comparison between standard ML and DML."""
        print("\n" + "="*60)
        print("Performance Comparison: Standard ML vs DML")
        print("="*60)

        # Generate small dataset for comparison
        S, K, prices, deltas = generate_digital_dataset_train(
            m=500, n_paths=100, r=0.05, sigma=0.2, T=1.0
        )

        features = torch.stack([
            S, K,
            torch.full_like(S, 0.05),
            torch.full_like(S, 0.2),
            torch.full_like(S, 1.0)
        ], dim=1).to(self.device)

        prices = prices.to(self.device)
        deltas = deltas.to(self.device)

        # Standard ML (price only)
        print("\nTraining Standard ML model (price only)...")
        model_standard = FeedForwardNet(5, [32, 16], 1).to(self.device)
        optimizer_standard = torch.optim.Adam(model_standard.parameters(), lr=0.001)

        standard_losses = []
        start_time = time.time()

        for epoch in range(1000):
            optimizer_standard.zero_grad()
            pred = model_standard(features)
            loss = torch.nn.MSELoss()(pred.squeeze(), prices)
            loss.backward()
            optimizer_standard.step()
            standard_losses.append(loss.item())

        standard_time = time.time() - start_time

        # DML (price + delta)
        print("Training DML model (price + delta)...")
        model_dml = DifferentialNet(5, [32, 16], 1).to(self.device)
        optimizer_dml = torch.optim.Adam(model_dml.parameters(), lr=0.001)

        dml_losses = []
        start_time = time.time()

        for epoch in range(1000):
            optimizer_dml.zero_grad()
            pred = model_dml(features)
            loss = dml_loss(pred, prices, deltas, features, lambda_val=1.0)
            loss.backward()
            optimizer_dml.step()
            dml_losses.append(loss.item())

        dml_time = time.time() - start_time

        # Plot comparison
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.semilogy(standard_losses, label='Standard ML', alpha=0.8)
        plt.semilogy(dml_losses, label='DML', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Convergence Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        categories = ['Standard ML', 'DML']
        times = [standard_time, dml_time]
        final_losses = [standard_losses[-1], dml_losses[-1]]

        x = np.arange(len(categories))
        width = 0.35

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        bars1 = ax1.bar(x - width/2, times, width, label='Time (s)', color='skyblue')
        bars2 = ax2.bar(x + width/2, final_losses, width, label='Final Loss', color='coral')

        ax1.set_xlabel('Method')
        ax1.set_ylabel('Training Time (s)', color='skyblue')
        ax2.set_ylabel('Final Loss', color='coral')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)

        plt.title('Performance Metrics')
        plt.tight_layout()

        output_path = self.output_dir / 'ml_vs_dml_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✅ Comparison Results:")
        print(f"   Standard ML: Time={standard_time:.2f}s, Final Loss={standard_losses[-1]:.6f}")
        print(f"   DML:         Time={dml_time:.2f}s, Final Loss={dml_losses[-1]:.6f}")
        print(f"   Speedup: {len(standard_losses)/len([l for l in dml_losses if l < standard_losses[-1]]):.1f}x faster convergence")
        print(f"   Plot saved to {output_path}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive DiffML experiments")
    parser.add_argument('--quick', action='store_true',
                       help='Run quick version with reduced samples')
    parser.add_argument('--comparison', action='store_true',
                       help='Run ML vs DML comparison')
    parser.add_argument('--output', default='experiment_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create runner
    runner = ComprehensiveExperimentRunner(output_dir=args.output)

    if args.comparison:
        # Run performance comparison only
        runner.run_performance_comparison()
    else:
        # Run all experiments
        results = runner.run_all_experiments()

        # Generate plots and reports
        runner.compare_convergence(results)
        runner.generate_report(results)

        # Run comparison
        runner.run_performance_comparison()

    print("\n✅ All experiments completed successfully!")
    print(f"   Results saved to: {runner.output_dir}")


if __name__ == "__main__":
    main()