#!/usr/bin/env python
"""CLI for running benchmark and sensitivity analyses on the digital experiment."""

from __future__ import annotations

import argparse

from diffml_article_replication.benchmark import (
    format_benchmark_table,
    run_digital_benchmark,
    run_lambda_delta_sweep,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run benchmarking and lambda_delta sweeps on the digital experiment.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Random seeds to benchmark over (default: 0 1).",
    )
    parser.add_argument(
        "--lambda-delta-values",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.5, 1.0],
        help="lambda_delta values to sweep (default: 0.0 0.1 0.5 1.0).",
    )
    return parser.parse_args()


def main() -> None:
    """Execute benchmark and lambda sweeps."""
    args = parse_args()

    model_modes: list[str] = ["standard", "delta_pathwise", "delta_lrm", "baseline_poly", "baseline_mlp"]
    print("Running digital benchmark...")
    benchmark_results = run_digital_benchmark(seeds=args.seeds, model_modes=model_modes)
    print(format_benchmark_table(benchmark_results))

    print("\nRunning lambda_delta sensitivity sweep...")
    sweep_results = run_lambda_delta_sweep(lambda_values=args.lambda_delta_values)
    for lambda_value, metrics in sweep_results.items():
        print(
            f"lambda_delta={lambda_value:.3f} -> "
            f"price_rmse={metrics['price_rmse']:.4f}, "
            f"delta_rmse={metrics['delta_rmse']:.4f}"
        )


if __name__ == "__main__":
    main()
