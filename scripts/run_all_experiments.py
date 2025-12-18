#!/usr/bin/env python
"""Main script to run all experiments from the DiffML paper.

This script executes all experiments in sequence and saves results
to the output directory.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add parent directory to path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffml.config import get_device, set_default_dtype
from diffml.experiments_barrier import run_barrier_experiment
from diffml.experiments_basket import run_basket_digital_experiment
from diffml.experiments_digital import run_digital_experiment
from diffml.experiments_gamma import run_gamma_experiment
from diffml.experiments_smoothing import run_smoothing_experiment


def create_output_directory(base_dir: str = "output") -> Path:
    """Create output directory structure.

    Parameters
    ----------
    base_dir : str
        Base output directory name.

    Returns
    -------
    Path
        Path to output directory.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each experiment
    for exp_name in ["digital", "barrier", "basket", "smoothing", "gamma"]:
        (output_dir / exp_name).mkdir(exist_ok=True)

    return output_dir


def run_all_experiments(
    output_dir: Path,
    experiments_to_run: List[str] = None,
    seed: int = 42,
) -> Dict[str, any]:
    """Run all experiments.

    Parameters
    ----------
    output_dir : Path
        Output directory for results.
    experiments_to_run : List[str]
        List of experiments to run. If None, runs all.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, any]
        Results from all experiments.
    """
    if experiments_to_run is None:
        experiments_to_run = ["digital", "barrier", "basket", "smoothing", "gamma"]

    # Set default dtype for financial computations
    set_default_dtype()

    # Check device
    device = get_device()
    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print(f"Experiments to run: {experiments_to_run}")
    print("-" * 50)

    results = {}

    # Run experiments
    if "digital" in experiments_to_run:
        print("\n" + "=" * 50)
        print("Running Digital Option Experiment")
        print("=" * 50)
        try:
            run_digital_experiment()
            results["digital"] = "Completed"
            print("✓ Digital option experiment completed")
        except Exception as e:
            print(f"✗ Digital option experiment failed: {e}")
            results["digital"] = f"Failed: {e}"

    if "barrier" in experiments_to_run:
        print("\n" + "=" * 50)
        print("Running Barrier Option Experiment")
        print("=" * 50)
        try:
            run_barrier_experiment()
            results["barrier"] = "Completed"
            print("✓ Barrier option experiment completed")
        except Exception as e:
            print(f"✗ Barrier option experiment failed: {e}")
            results["barrier"] = f"Failed: {e}"

    if "basket" in experiments_to_run:
        print("\n" + "=" * 50)
        print("Running Basket Digital Option Experiment")
        print("=" * 50)
        try:
            run_basket_digital_experiment()
            results["basket"] = "Completed"
            print("✓ Basket digital option experiment completed")
        except Exception as e:
            print(f"✗ Basket digital option experiment failed: {e}")
            results["basket"] = f"Failed: {e}"

    if "smoothing" in experiments_to_run:
        print("\n" + "=" * 50)
        print("Running Smoothing Experiment")
        print("=" * 50)
        try:
            run_smoothing_experiment()
            results["smoothing"] = "Completed"
            print("✓ Smoothing experiment completed")
        except Exception as e:
            print(f"✗ Smoothing experiment failed: {e}")
            results["smoothing"] = f"Failed: {e}"

    if "gamma" in experiments_to_run:
        print("\n" + "=" * 50)
        print("Running Gamma Portfolio Experiment")
        print("=" * 50)
        try:
            run_gamma_experiment()
            results["gamma"] = "Completed"
            print("✓ Gamma portfolio experiment completed")
        except Exception as e:
            print(f"✗ Gamma portfolio experiment failed: {e}")
            results["gamma"] = f"Failed: {e}"

    # Save summary
    print("\n" + "=" * 50)
    print("Experiments Complete!")
    print("=" * 50)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "seed": seed,
        "experiments_run": experiments_to_run,
        "output_directory": str(output_dir),
    }

    with open(output_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"Summary saved to: {output_dir / 'experiment_summary.json'}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run DiffML paper experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Base output directory for results",
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["digital", "barrier", "basket", "smoothing", "gamma", "all"],
        default=["all"],
        help="Which experiments to run",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for computation",
    )

    args = parser.parse_args()

    # Parse experiments to run
    if "all" in args.experiments:
        experiments_to_run = None
    else:
        experiments_to_run = args.experiments

    # Create output directory
    output_dir = create_output_directory(args.output_dir)

    # Run experiments
    try:
        results = run_all_experiments(
            output_dir=output_dir,
            experiments_to_run=experiments_to_run,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        print("\nExperiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running experiments: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()