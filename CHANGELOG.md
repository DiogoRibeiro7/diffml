# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
<!-- New features -->

### Changed
<!-- Changes in existing functionality -->

### Fixed
<!-- Bug fixes -->

### Deprecated
<!-- Soon-to-be removed features -->

### Removed
<!-- Removed features -->

### Security
<!-- Security vulnerability fixes -->

## [0.1.0] - 2024-12-18

### Added
- Initial implementation of differential machine learning experiments from "Differential ML with a Difference" paper
  - Digital option pricing experiment with pathwise and LRM delta methods
  - Barrier option (down-and-out call) pricing experiment
  - Basket digital option experiment with Bachelier model (20-dimensional)
  - Smoothed digital option experiment with ramp smoothing analysis
  - Gamma-regularized portfolio hedging experiment (butterfly spread)
- PyTorch-based neural network architecture (`PricingNet`) with Softplus activations
- Differential ML loss function supporting price, delta, and gamma regularization
- Black-Scholes analytical formulas for benchmarking
  - Digital option price and delta calculations
  - Vanilla call option price and gamma calculations
- Monte Carlo simulation framework
  - Single-step and two-step Black-Scholes path simulation
  - Pathwise and likelihood ratio method (LRM) sensitivity estimation
- Dataset generation utilities for all experiment types
  - Support for both pathwise and LRM delta labels
  - Configurable Monte Carlo paths and spot price grids
- Generic training loop with multiple modes
  - Standard ML (price only)
  - Delta pathwise regularization
  - Delta LRM regularization
  - Gamma PW-LR regularization
- Comprehensive test suite using pytest
  - Unit tests for Black-Scholes analytics
  - Tests for simulation and dataset generation
  - Training loop and neural network tests
  - Smoke tests for all experiments
- CI/CD infrastructure
  - GitHub Actions workflow for continuous integration
  - Automated testing with Python 3.10, 3.11, and 3.12
  - Code quality checks with ruff and mypy
  - Release workflow for PyPI publishing
  - Dependabot configuration for dependency updates
- Project documentation
  - Comprehensive README with installation and usage instructions
  - CONTRIBUTING guide with development workflow
  - CODE_OF_CONDUCT for community guidelines
  - SECURITY policy for vulnerability reporting
  - RELEASE guide for version management
  - MIT LICENSE
- Development tooling
  - Poetry for dependency management
  - Configuration for ruff (linting) and mypy (type checking)
  - Pre-configured pytest settings
  - Pull request and issue templates

### Technical Details
- Double precision (`torch.float64`) by default for numerical accuracy
- Cosine annealing learning rate scheduler
- Adam optimizer with configurable parameters
- Support for both CPU and CUDA computation
- Reproducible experiments with fixed random seeds

### Dependencies
- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib, Pandas
- Development tools: pytest, mypy, ruff

---

[Unreleased]: https://github.com/diogoribeiro7/diffml/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/diogoribeiro7/diffml/releases/tag/v0.1.0
