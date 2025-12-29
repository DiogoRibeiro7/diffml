"""
DiffML Command Line Interface

Main CLI entry point for the DiffML package.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import torch
import json

from .models import DifferentialRegressor
from .trainers import DifferentialTrainer
from .datasets import BlackScholesDataset, create_dataset
from .benchmarking import BenchmarkRunner


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DiffML - Differential Machine Learning for Quantitative Finance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  diffml train --dataset black-scholes --epochs 100 --output model.pt

  # Price an option
  diffml price --model model.pt --spot 100 --strike 100 --maturity 1.0

  # Run benchmarks
  diffml benchmark --suite full --output results.json

  # Start dashboard
  diffml dashboard --port 8501
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a DML model')
    train_parser.add_argument('--dataset', type=str, default='black-scholes',
                             choices=['black-scholes', 'american', 'barrier', 'exotic'],
                             help='Dataset type')
    train_parser.add_argument('--samples', type=int, default=10000,
                             help='Number of training samples')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=256,
                             help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.001,
                             help='Learning rate')
    train_parser.add_argument('--differential-weight', type=float, default=0.5,
                             help='Weight for differential loss')
    train_parser.add_argument('--hidden-units', type=int, nargs='+', default=[128, 128, 128],
                             help='Hidden layer sizes')
    train_parser.add_argument('--output', type=str, default='model.pt',
                             help='Output model file')
    train_parser.add_argument('--device', type=str, default='auto',
                             choices=['auto', 'cpu', 'cuda'],
                             help='Device to use')

    # Price command
    price_parser = subparsers.add_parser('price', help='Price an option')
    price_parser.add_argument('--model', type=str, required=True,
                             help='Path to trained model')
    price_parser.add_argument('--spot', type=float, default=100.0,
                             help='Spot price')
    price_parser.add_argument('--strike', type=float, default=100.0,
                             help='Strike price')
    price_parser.add_argument('--maturity', type=float, default=1.0,
                             help='Time to maturity')
    price_parser.add_argument('--rate', type=float, default=0.05,
                             help='Risk-free rate')
    price_parser.add_argument('--volatility', type=float, default=0.2,
                             help='Volatility')
    price_parser.add_argument('--greeks', action='store_true',
                             help='Calculate Greeks')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument('--suite', type=str, default='quick',
                                 choices=['quick', 'full', 'custom'],
                                 help='Benchmark suite to run')
    benchmark_parser.add_argument('--output', type=str, default='benchmark_results.json',
                                 help='Output file for results')
    benchmark_parser.add_argument('--compare', type=str,
                                 help='Compare with baseline file')

    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Start web dashboard')
    dashboard_parser.add_argument('--port', type=int, default=8501,
                                 help='Port for dashboard')
    dashboard_parser.add_argument('--host', type=str, default='localhost',
                                 help='Host address')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show package information')

    # Parse arguments
    args = parser.parse_args()

    if args.command == 'train':
        train_model(args)
    elif args.command == 'price':
        price_option(args)
    elif args.command == 'benchmark':
        run_benchmark(args)
    elif args.command == 'dashboard':
        start_dashboard(args)
    elif args.command == 'info':
        show_info()
    else:
        parser.print_help()
        sys.exit(1)


def train_model(args):
    """Train a DML model."""
    print(f"🚀 Training DML model...")
    print(f"   Dataset: {args.dataset}")
    print(f"   Samples: {args.samples}")
    print(f"   Epochs: {args.epochs}")

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"   Device: {device}")

    # Create dataset
    if args.dataset == 'black-scholes':
        dataset = BlackScholesDataset(n_samples=args.samples)
    else:
        dataset = create_dataset(args.dataset, n_samples=args.samples)

    X, y, dy = dataset.generate()

    # Create model
    model = DifferentialRegressor(
        input_dim=X.shape[1],
        hidden_units=args.hidden_units,
        activation='relu'
    )

    # Train
    trainer = DifferentialTrainer(
        model=model,
        differential_weight=args.differential_weight,
        learning_rate=args.learning_rate,
        device=device
    )

    history = trainer.fit(
        X, y, dy,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': X.shape[1],
            'hidden_units': args.hidden_units,
            'activation': 'relu'
        },
        'training_history': history
    }, args.output)

    print(f"\n✅ Model saved to {args.output}")
    print(f"   Final loss: {history['train_loss'][-1]:.6f}")


def price_option(args):
    """Price an option using trained model."""
    print(f"💹 Pricing option...")

    # Load model
    checkpoint = torch.load(args.model, map_location='cpu')

    model = DifferentialRegressor(
        input_dim=checkpoint['model_config']['input_dim'],
        hidden_units=checkpoint['model_config']['hidden_units'],
        activation=checkpoint['model_config']['activation']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepare input
    x = torch.tensor([[
        args.spot,
        args.strike,
        args.maturity,
        args.rate,
        args.volatility
    ]], dtype=torch.float32)

    # Price
    with torch.no_grad():
        if args.greeks:
            x.requires_grad_(True)
            price = model(x)
            grads = torch.autograd.grad(price, x, create_graph=False)[0]

            print(f"\n📊 Results:")
            print(f"   Price: ${price.item():.4f}")
            print(f"   Delta: {grads[0, 0].item():.4f}")
            print(f"   Gamma: (requires second-order)")
            print(f"   Vega: {grads[0, 4].item():.4f}")
            print(f"   Theta: {-grads[0, 2].item():.4f}")
            print(f"   Rho: {grads[0, 3].item():.4f}")
        else:
            price = model(x)
            print(f"\n💰 Option Price: ${price.item():.4f}")


def run_benchmark(args):
    """Run performance benchmarks."""
    print(f"⚡ Running benchmarks...")
    print(f"   Suite: {args.suite}")

    runner = BenchmarkRunner()

    if args.suite == 'quick':
        results = runner.run_quick_benchmark()
    elif args.suite == 'full':
        results = runner.run_full_benchmark()
    else:
        results = runner.run_custom_benchmark()

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {args.output}")

    # Compare with baseline if provided
    if args.compare:
        print(f"\n📊 Comparing with baseline: {args.compare}")
        with open(args.compare, 'r') as f:
            baseline = json.load(f)

        # Simple comparison
        for key in results:
            if key in baseline:
                improvement = (baseline[key] - results[key]) / baseline[key] * 100
                print(f"   {key}: {improvement:+.1f}% improvement")


def start_dashboard(args):
    """Start the web dashboard."""
    print(f"🎛️ Starting DiffML Dashboard...")
    print(f"   URL: http://{args.host}:{args.port}")
    print(f"\n   Press Ctrl+C to stop")

    import subprocess
    import os

    # Find dashboard.py
    dashboard_path = Path(__file__).parent.parent.parent / 'dashboard.py'

    if not dashboard_path.exists():
        print(f"❌ Dashboard not found at {dashboard_path}")
        sys.exit(1)

    # Run streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        str(dashboard_path),
        '--server.port', str(args.port),
        '--server.address', args.host
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")


def show_info():
    """Show package information."""
    import diffml

    print(f"""
╔══════════════════════════════════════════════════════════╗
║                    DiffML Package Info                    ║
╚══════════════════════════════════════════════════════════╝

Version: {diffml.__version__}
Location: {Path(diffml.__file__).parent}
Python: {sys.version.split()[0]}
PyTorch: {torch.__version__}
CUDA Available: {torch.cuda.is_available()}

Features:
  ✓ Differential Machine Learning
  ✓ 5-10x faster convergence
  ✓ Automatic Greek calculation
  ✓ GPU acceleration
  ✓ Production ready

For documentation, visit:
  https://github.com/DiogoRibeiro7/diffml
    """)


if __name__ == "__main__":
    main()