#!/usr/bin/env python
"""Simple test runner script for the DiffML test suite.

Usage:
    python run_tests.py              # Run all tests except slow ones
    python run_tests.py --all         # Run all tests including slow ones
    python run_tests.py --coverage    # Run with coverage report
    python run_tests.py --verbose     # Run with verbose output
"""

import argparse
import subprocess
from pathlib import Path


def main():
    """Run the test suite with specified options."""
    parser = argparse.ArgumentParser(
        description="Run DiffML test suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests including slow ones",
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage report",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run tests with verbose output",
    )

    parser.add_argument(
        "--specific",
        type=str,
        help="Run specific test file or test function",
    )

    parser.add_argument(
        "--markers", "-m",
        type=str,
        help="Run tests with specific markers (e.g., 'not slow')",
    )

    args = parser.parse_args()

    # Build pytest command
    cmd = ["pytest"]

    # Ensure package can be imported
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    # Add test directory
    test_dir = Path(__file__).parent / "tests"
    if args.specific:
        cmd.append(f"{test_dir}/{args.specific}")
    else:
        cmd.append(str(test_dir))

    # Add verbosity
    if args.verbose:
        cmd.append("-v")

    # Add markers
    if args.markers:
        cmd.extend(["-m", args.markers])
    elif not args.all:
        # By default, skip slow tests
        cmd.extend(["-m", "not slow"])

    # Add coverage
    if args.coverage:
        cmd.extend([
            "--cov=diffml",
            "--cov-report=term-missing",
            "--cov-report=html"
        ])

    # Add color output
    cmd.append("--color=yes")

    # Show test summary
    cmd.append("-ra")

    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)

    # Run pytest
    result = subprocess.run(cmd)

    # Return exit code
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
