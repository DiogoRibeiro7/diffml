#!/usr/bin/env python
"""Command-line interface for running experiments with configuration files.

This script provides a CLI for running DiffML experiments using TOML
configuration files.

Usage:
    python scripts/run_experiment.py --config configs/digital_default.toml
    python scripts/run_experiment.py --list
    python scripts/run_experiment.py --validate configs/digital_default.toml
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diffml.run_experiment import (
    run_experiment_from_config,
    list_available_experiments,
    validate_config,
)


def main():
    """Main entry point for the experiment runner CLI."""
    parser = argparse.ArgumentParser(
        description="Run DiffML experiments from configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run an experiment:
    %(prog)s --config configs/digital_default.toml

  List available experiments:
    %(prog)s --list

  Validate a configuration:
    %(prog)s --validate configs/digital_default.toml

Available configurations:
  - configs/digital_default.toml    : Digital option experiment
  - configs/barrier_default.toml    : Barrier option experiment
  - configs/basket_digital_default.toml : Basket option experiment
  - configs/asian_default.toml      : Asian option experiment
  - configs/smoothing_default.toml  : Smoothing experiment
        """,
    )

    # Create mutually exclusive group for different actions
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        help="Path to TOML configuration file",
    )

    group.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments",
    )

    group.add_argument(
        "--validate",
        type=str,
        metavar="PATH",
        help="Validate a configuration file without running",
    )

    # Additional options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Override random seed from configuration",
    )

    args = parser.parse_args()

    # Handle different actions
    try:
        if args.list:
            list_available_experiments()

        elif args.validate:
            config_path = Path(args.validate)
            if not config_path.exists():
                print(f"ERROR: Configuration file not found: {args.validate}")
                sys.exit(1)

            if validate_config(args.validate):
                sys.exit(0)
            else:
                sys.exit(1)

        elif args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                print(f"ERROR: Configuration file not found: {args.config}")
                sys.exit(1)

            # If seed override is provided, we would need to modify the config
            # For now, just run with the config as-is
            if args.seed is not None:
                print(f"Note: Seed override not yet implemented. Using seed from config.")

            run_experiment_from_config(args.config)

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(1)

    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"\nERROR: {e}")
            print("\nRun with --verbose for full traceback.")
        sys.exit(1)


if __name__ == "__main__":
    main()