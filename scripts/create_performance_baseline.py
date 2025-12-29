#!/usr/bin/env python
"""Create performance baselines for DiffML.

This script establishes performance baselines for all components
and saves them for regression testing.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffml.benchmarking import DiffMLBenchmarkSuite, create_benchmark_report


def create_baselines():
    """Create comprehensive performance baselines."""
    print("\n📊 Creating Performance Baselines for DiffML")
    print("="*60)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device.type == 'cuda':
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Run benchmark suite
    suite = DiffMLBenchmarkSuite(device=device)

    print("Running benchmarks (this may take a few minutes)...")
    print("-" * 60)

    # Run all benchmarks
    results = suite.run_all()

    # Save baseline
    baseline_dir = Path('benchmarks')
    baseline_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_path = baseline_dir / f'baseline_{timestamp}.json'

    suite.benchmarker.save_results(str(baseline_path))
    print(f"\n✅ Baseline saved to: {baseline_path}")

    # Create latest symlink
    latest_path = baseline_dir / 'baseline_latest.json'
    if latest_path.exists():
        latest_path.unlink()

    # Copy content instead of symlink (Windows compatibility)
    with open(baseline_path, 'r') as src:
        with open(latest_path, 'w') as dst:
            dst.write(src.read())

    print(f"✅ Latest baseline updated: {latest_path}")

    # Generate report
    report_path = baseline_dir / f'report_{timestamp}.md'
    create_benchmark_report(str(baseline_path), str(report_path))
    print(f"✅ Report generated: {report_path}")

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    categories = {}
    for result in results:
        if result.category not in categories:
            categories[result.category] = []
        categories[result.category].append(result)

    for category, cat_results in categories.items():
        print(f"\n{category.upper()}:")
        for r in cat_results:
            print(f"  {r.name}: {r.mean_time*1000:.2f}ms "
                  f"(±{r.std_time*1000:.2f}ms)")

    return baseline_path


def compare_with_baseline(baseline_path: str = None):
    """Compare current performance with baseline."""
    print("\n🔍 Comparing with Baseline")
    print("="*60)

    if baseline_path is None:
        baseline_path = 'benchmarks/baseline_latest.json'

    if not Path(baseline_path).exists():
        print(f"❌ Baseline not found: {baseline_path}")
        print("   Run with --create flag first to create a baseline")
        return 1

    # Run current benchmarks
    suite = DiffMLBenchmarkSuite()
    suite.run_all()

    # Compare with baseline
    regressions = suite.benchmarker.compare_with_baseline(
        baseline_path,
        threshold_pct=10.0  # 10% regression threshold
    )

    if regressions:
        print("\n⚠️ Performance Regressions Detected:")
        for reg in regressions:
            print(f"  - {reg}")
        return 1
    else:
        print("\n✅ No performance regressions detected")
        return 0


def run_quick_benchmark():
    """Run a quick performance check."""
    print("\n⚡ Quick Performance Check")
    print("="*60)

    import time
    from diffml.simulation import simulate_bs_terminal
    from diffml.networks import FeedForwardNet
    from diffml.datasets_digital import generate_digital_dataset_train

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test 1: Monte Carlo simulation
    print("\n1. Monte Carlo Simulation (10k paths):")
    start = time.time()
    spots = torch.tensor([[100.0]], device=device)
    from diffml.config import BSParams
    params = BSParams(r=0.05, sigma=0.2, T=1.0)
    ST, _ = simulate_bs_terminal(spots, params, n_paths=10000)
    sim_time = time.time() - start
    print(f"   Time: {sim_time*1000:.2f}ms")
    print(f"   Throughput: {10000/sim_time:.0f} paths/sec")

    # Test 2: Neural network inference
    print("\n2. Neural Network Inference (1000 samples):")
    model = FeedForwardNet(5, [64, 32, 16], 1).to(device)
    model.eval()
    input_data = torch.randn(1000, 5, device=device)

    # Warmup
    for _ in range(10):
        _ = model(input_data)

    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_data)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    inf_time = (time.time() - start) / 100
    print(f"   Time per batch: {inf_time*1000:.2f}ms")
    print(f"   Throughput: {1000/inf_time:.0f} samples/sec")

    # Test 3: Dataset generation
    print("\n3. Dataset Generation (1000 samples):")
    start = time.time()
    S, K, prices, deltas = generate_digital_dataset_train(
        m=1000, n_paths=100, r=0.05, sigma=0.2, T=1.0
    )
    data_time = time.time() - start
    print(f"   Time: {data_time:.2f}s")
    print(f"   Throughput: {1000/data_time:.0f} samples/sec")

    print("\n✅ Quick benchmark completed")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Performance baseline management")
    parser.add_argument('--create', action='store_true',
                       help='Create new baseline')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with baseline')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Path to baseline file for comparison')

    args = parser.parse_args()

    if args.create:
        baseline = create_baselines()
        sys.exit(0)
    elif args.compare:
        sys.exit(compare_with_baseline(args.baseline))
    elif args.quick:
        sys.exit(run_quick_benchmark())
    else:
        # Default: create baseline
        create_baselines()
        sys.exit(0)