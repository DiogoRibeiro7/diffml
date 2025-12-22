"""CLI for running path-dependent experiments."""

from __future__ import annotations

import argparse

from diffml_article_replication.experiments_path_dependent import (
    run_arithmetic_asian_experiment,
    run_lookback_call_experiment,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run path-dependent option experiments (Asian / Lookback)."
    )
    parser.add_argument(
        "--experiment",
        choices=("asian", "lookback", "both"),
        default="both",
        help="Which experiment to run (default: both).",
    )
    return parser


def main() -> None:
    """Entry-point for running Asian and lookback experiments."""

    parser = _build_parser()
    args = parser.parse_args()

    if args.experiment in {"asian", "both"}:
        run_arithmetic_asian_experiment()
    if args.experiment in {"lookback", "both"}:
        run_lookback_call_experiment()


if __name__ == "__main__":
    main()
