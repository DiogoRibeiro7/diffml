"""Benchmarking suite for DiffML performance regression testing.

This module provides comprehensive benchmarking tools to track performance
metrics across different components and detect performance regressions.
"""

import time
import json
import statistics
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
import platform
import psutil
import torch
import numpy as np
from contextlib import contextmanager

from .config import get_device
from .simulation import simulate_bs_terminal, simulate_bs_paths
from .networks import FeedForwardNet, DifferentialNet
from .training import train_model
from .losses import dml_loss


@dataclass
class BenchmarkResult:
    """Container for benchmark results.

    Attributes:
        name: Benchmark name
        category: Benchmark category (simulation, training, inference, etc.)
        mean_time: Mean execution time in seconds
        std_time: Standard deviation of execution time
        min_time: Minimum execution time
        max_time: Maximum execution time
        median_time: Median execution time
        p95_time: 95th percentile execution time
        p99_time: 99th percentile execution time
        iterations: Number of iterations run
        memory_peak_mb: Peak memory usage in MB
        gpu_memory_mb: GPU memory usage in MB (if applicable)
        throughput: Operations per second
        metadata: Additional metadata
    """

    name: str
    category: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    median_time: float
    p95_time: float
    p99_time: float
    iterations: int
    memory_peak_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    throughput: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def compare(self, baseline: 'BenchmarkResult') -> Dict[str, float]:
        """Compare with baseline results.

        Parameters:
            baseline: Baseline results to compare against

        Returns:
            Dictionary with percentage changes
        """
        return {
            'mean_change_pct': ((self.mean_time - baseline.mean_time) / baseline.mean_time) * 100,
            'median_change_pct': ((self.median_time - baseline.median_time) / baseline.median_time) * 100,
            'p95_change_pct': ((self.p95_time - baseline.p95_time) / baseline.p95_time) * 100,
            'throughput_change_pct': ((self.throughput - baseline.throughput) / baseline.throughput * 100)
                                     if self.throughput and baseline.throughput else 0
        }


@dataclass
class SystemInfo:
    """System information for benchmark context."""

    python_version: str
    torch_version: str
    cuda_version: Optional[str]
    gpu_name: Optional[str]
    cpu_name: str
    cpu_count: int
    memory_gb: float
    os_name: str
    timestamp: str

    @classmethod
    def collect(cls) -> 'SystemInfo':
        """Collect current system information."""
        cuda_version = None
        gpu_name = None

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)

        return cls(
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            cuda_version=cuda_version,
            gpu_name=gpu_name,
            cpu_name=platform.processor(),
            cpu_count=psutil.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            os_name=platform.platform(),
            timestamp=datetime.now().isoformat()
        )


class Benchmarker:
    """Main benchmarking class."""

    def __init__(
        self,
        warmup_iterations: int = 5,
        benchmark_iterations: int = 100,
        collect_memory: bool = True,
        device: Optional[torch.device] = None
    ):
        """Initialize benchmarker.

        Parameters:
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
            collect_memory: Whether to collect memory statistics
            device: Device to run benchmarks on
        """
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.collect_memory = collect_memory
        self.device = device or get_device()
        self.results: List[BenchmarkResult] = []
        self.system_info = SystemInfo.collect()

    @contextmanager
    def _memory_tracking(self):
        """Context manager for memory tracking."""
        if self.collect_memory:
            # CPU memory
            process = psutil.Process()
            cpu_mem_start = process.memory_info().rss / 1024**2

            # GPU memory
            gpu_mem_start = None
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize(self.device)
                gpu_mem_start = torch.cuda.memory_allocated(self.device) / 1024**2

            yield

            # Collect peak memory
            cpu_mem_peak = process.memory_info().rss / 1024**2 - cpu_mem_start

            gpu_mem_peak = None
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
                gpu_mem_peak = torch.cuda.max_memory_allocated(self.device) / 1024**2

            self._last_memory = (cpu_mem_peak, gpu_mem_peak)
        else:
            yield
            self._last_memory = (None, None)

    def benchmark_function(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict = {},
        name: str = "unnamed",
        category: str = "general"
    ) -> BenchmarkResult:
        """Benchmark a function.

        Parameters:
            func: Function to benchmark
            args: Positional arguments for function
            kwargs: Keyword arguments for function
            name: Name of benchmark
            category: Category of benchmark

        Returns:
            Benchmark results
        """
        # Warmup
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)

        # Synchronize if using GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

        # Benchmark
        times = []
        with self._memory_tracking():
            for _ in range(self.benchmark_iterations):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)

                start_time = time.perf_counter()
                result = func(*args, **kwargs)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)

                end_time = time.perf_counter()
                times.append(end_time - start_time)

        # Calculate statistics
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        p95_time = np.percentile(times, 95)
        p99_time = np.percentile(times, 99)

        # Calculate throughput (operations per second)
        throughput = 1 / mean_time if mean_time > 0 else 0

        result = BenchmarkResult(
            name=name,
            category=category,
            mean_time=mean_time,
            std_time=std_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            p95_time=p95_time,
            p99_time=p99_time,
            iterations=self.benchmark_iterations,
            memory_peak_mb=self._last_memory[0],
            gpu_memory_mb=self._last_memory[1],
            throughput=throughput
        )

        self.results.append(result)
        return result

    def save_results(self, filepath: str):
        """Save benchmark results to JSON file.

        Parameters:
            filepath: Path to save results
        """
        data = {
            'system_info': asdict(self.system_info),
            'results': [r.to_dict() for r in self.results]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_baseline(self, filepath: str) -> Dict[str, BenchmarkResult]:
        """Load baseline results for comparison.

        Parameters:
            filepath: Path to baseline results

        Returns:
            Dictionary mapping benchmark names to results
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        baseline = {}
        for r in data['results']:
            result = BenchmarkResult(**r)
            baseline[result.name] = result

        return baseline

    def compare_with_baseline(self, baseline_path: str, threshold_pct: float = 10.0) -> List[str]:
        """Compare current results with baseline.

        Parameters:
            baseline_path: Path to baseline results
            threshold_pct: Regression threshold percentage

        Returns:
            List of detected regressions
        """
        baseline = self.load_baseline(baseline_path)
        regressions = []

        for result in self.results:
            if result.name in baseline:
                comparison = result.compare(baseline[result.name])
                if comparison['mean_change_pct'] > threshold_pct:
                    regressions.append(
                        f"{result.name}: {comparison['mean_change_pct']:.1f}% slower"
                    )

        return regressions


class DiffMLBenchmarkSuite:
    """Comprehensive benchmark suite for DiffML components."""

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize benchmark suite.

        Parameters:
            device: Device to run benchmarks on
        """
        self.device = device or get_device()
        self.benchmarker = Benchmarker(device=self.device)

    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks.

        Returns:
            List of all benchmark results
        """
        self.benchmark_simulation()
        self.benchmark_networks()
        self.benchmark_training()
        self.benchmark_datasets()
        return self.benchmarker.results

    def benchmark_simulation(self):
        """Benchmark Monte Carlo simulation performance."""
        # Terminal value simulation
        self.benchmarker.benchmark_function(
            simulate_bs_terminal,
            args=(100.0, 0.05, 0.2, 1.0, 100, 10000),
            name="simulate_bs_terminal_10k",
            category="simulation"
        )

        self.benchmarker.benchmark_function(
            simulate_bs_terminal,
            args=(100.0, 0.05, 0.2, 1.0, 100, 100000),
            name="simulate_bs_terminal_100k",
            category="simulation"
        )

        # Path simulation
        self.benchmarker.benchmark_function(
            simulate_bs_paths,
            args=(100.0, 0.05, 0.2, 1.0, 252, 1000),
            name="simulate_bs_paths_1k",
            category="simulation"
        )

        self.benchmarker.benchmark_function(
            simulate_bs_paths,
            args=(100.0, 0.05, 0.2, 1.0, 252, 10000),
            name="simulate_bs_paths_10k",
            category="simulation"
        )

    def benchmark_networks(self):
        """Benchmark neural network inference."""
        # Small network
        small_net = FeedForwardNet(5, [32, 16], 1).to(self.device)
        input_small = torch.randn(100, 5, device=self.device)

        self.benchmarker.benchmark_function(
            small_net,
            args=(input_small,),
            name="feedforward_small_inference",
            category="inference"
        )

        # Medium network
        medium_net = FeedForwardNet(5, [64, 32, 16], 1).to(self.device)
        input_medium = torch.randn(256, 5, device=self.device)

        self.benchmarker.benchmark_function(
            medium_net,
            args=(input_medium,),
            name="feedforward_medium_inference",
            category="inference"
        )

        # Large network
        large_net = FeedForwardNet(20, [128, 64, 32, 16], 1).to(self.device)
        input_large = torch.randn(512, 20, device=self.device)

        self.benchmarker.benchmark_function(
            large_net,
            args=(input_large,),
            name="feedforward_large_inference",
            category="inference"
        )

        # Differential network
        diff_net = DifferentialNet(5, [50, 50], 1).to(self.device)

        self.benchmarker.benchmark_function(
            diff_net,
            args=(input_small,),
            name="differential_net_inference",
            category="inference"
        )

    def benchmark_training(self):
        """Benchmark training operations."""
        # Setup model and data
        model = FeedForwardNet(5, [32, 16], 1).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())

        features = torch.randn(1000, 5, device=self.device)
        prices = torch.randn(1000, device=self.device)
        deltas = torch.randn(1000, device=self.device)

        def train_step():
            optimizer.zero_grad()
            predictions = model(features)
            loss = dml_loss(predictions, prices, deltas, features, lambda_val=1.0)
            loss.backward()
            optimizer.step()
            return loss.item()

        self.benchmarker.benchmark_function(
            train_step,
            name="training_step_dml",
            category="training"
        )

        # Benchmark with different batch sizes
        for batch_size in [32, 64, 128, 256]:
            batch_features = features[:batch_size]
            batch_prices = prices[:batch_size]
            batch_deltas = deltas[:batch_size]

            def train_batch():
                optimizer.zero_grad()
                predictions = model(batch_features)
                loss = dml_loss(predictions, batch_prices, batch_deltas,
                              batch_features, lambda_val=1.0)
                loss.backward()
                optimizer.step()
                return loss.item()

            self.benchmarker.benchmark_function(
                train_batch,
                name=f"training_batch_{batch_size}",
                category="training"
            )

    def benchmark_datasets(self):
        """Benchmark dataset generation."""
        from .datasets_digital import generate_digital_dataset_train
        from .datasets_barrier import generate_barrier_dataset_with_paths
        from .datasets_basket import generate_basket_dataset

        # Digital dataset
        self.benchmarker.benchmark_function(
            generate_digital_dataset_train,
            kwargs={'m': 1000, 'n_paths': 1000, 'r': 0.05, 'sigma': 0.2, 'T': 1.0},
            name="generate_digital_1k",
            category="dataset"
        )

        # Barrier dataset
        self.benchmarker.benchmark_function(
            generate_barrier_dataset_with_paths,
            kwargs={'m': 1000, 'n_paths': 1000, 'n_steps': 100,
                   'r': 0.05, 'sigma': 0.2, 'T': 1.0},
            name="generate_barrier_1k",
            category="dataset"
        )

        # Basket dataset
        self.benchmarker.benchmark_function(
            generate_basket_dataset,
            kwargs={'d': 20, 'm': 1000, 'n_paths': 1000,
                   'r': 0.05, 'sigma': 0.2, 'T': 0.25},
            name="generate_basket_1k",
            category="dataset"
        )


def run_regression_tests(baseline_path: Optional[str] = None, save_path: str = "benchmark_results.json"):
    """Run performance regression tests.

    Parameters:
        baseline_path: Path to baseline results (if None, just save current)
        save_path: Path to save current results

    Returns:
        List of detected regressions (empty if no baseline)
    """
    suite = DiffMLBenchmarkSuite()
    suite.run_all()

    # Save results
    suite.benchmarker.save_results(save_path)

    # Check for regressions if baseline provided
    regressions = []
    if baseline_path and Path(baseline_path).exists():
        regressions = suite.benchmarker.compare_with_baseline(baseline_path)

        if regressions:
            print("⚠️ Performance regressions detected:")
            for reg in regressions:
                print(f"  - {reg}")
        else:
            print("✅ No performance regressions detected")

    return regressions


def create_benchmark_report(results_path: str, output_path: str = "benchmark_report.md"):
    """Create a markdown report from benchmark results.

    Parameters:
        results_path: Path to benchmark results JSON
        output_path: Path to save markdown report
    """
    with open(results_path, 'r') as f:
        data = json.load(f)

    report = ["# DiffML Performance Benchmark Report\n"]
    report.append(f"Generated: {data['system_info']['timestamp']}\n")

    # System information
    report.append("## System Information\n")
    info = data['system_info']
    report.append(f"- Python: {info['python_version']}")
    report.append(f"- PyTorch: {info['torch_version']}")
    if info['cuda_version']:
        report.append(f"- CUDA: {info['cuda_version']}")
        report.append(f"- GPU: {info['gpu_name']}")
    report.append(f"- CPU: {info['cpu_name']} ({info['cpu_count']} cores)")
    report.append(f"- Memory: {info['memory_gb']:.1f} GB")
    report.append(f"- OS: {info['os_name']}\n")

    # Results by category
    categories = {}
    for result in data['results']:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)

    for category, results in categories.items():
        report.append(f"\n## {category.capitalize()} Benchmarks\n")
        report.append("| Benchmark | Mean (ms) | Std (ms) | P95 (ms) | Throughput (ops/s) |")
        report.append("|-----------|-----------|----------|----------|-------------------|")

        for r in results:
            mean_ms = r['mean_time'] * 1000
            std_ms = r['std_time'] * 1000
            p95_ms = r['p95_time'] * 1000
            throughput = r.get('throughput', 0)

            report.append(f"| {r['name']} | {mean_ms:.2f} | {std_ms:.2f} | "
                         f"{p95_ms:.2f} | {throughput:.1f} |")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Benchmark report saved to {output_path}")


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DiffML Benchmarking Suite")
    parser.add_argument("--baseline", help="Path to baseline results for regression testing")
    parser.add_argument("--output", default="benchmark_results.json",
                       help="Path to save results")
    parser.add_argument("--report", action="store_true",
                       help="Generate markdown report")
    parser.add_argument("--category", choices=["all", "simulation", "inference",
                                               "training", "dataset"],
                       default="all", help="Benchmark category to run")

    args = parser.parse_args()

    if args.report and Path(args.output).exists():
        create_benchmark_report(args.output)
    else:
        # Run benchmarks
        if args.category == "all":
            regressions = run_regression_tests(args.baseline, args.output)
        else:
            suite = DiffMLBenchmarkSuite()
            if args.category == "simulation":
                suite.benchmark_simulation()
            elif args.category == "inference":
                suite.benchmark_networks()
            elif args.category == "training":
                suite.benchmark_training()
            elif args.category == "dataset":
                suite.benchmark_datasets()

            suite.benchmarker.save_results(args.output)
            print(f"Results saved to {args.output}")

            if args.baseline:
                regressions = suite.benchmarker.compare_with_baseline(args.baseline)

        if args.report:
            create_benchmark_report(args.output)