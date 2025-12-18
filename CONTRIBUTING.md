# Contributing to diffml

Thank you for your interest in contributing to `diffml`! This project implements a faithful reproduction of experiments from the paper "Differential ML with a Difference" by Paul Glasserman and Siddharth Hemant Karmarkar using PyTorch and modern Python tooling.

We welcome contributions that improve code quality, enhance reproducibility, extend functionality, or fix issues. This guide will help you get started.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Poetry (for dependency management)
- Git

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/diffml.git
   cd diffml
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   poetry install --with dev
   ```
   This installs both the project dependencies and development tools (pytest, mypy, ruff).

3. **Verify your setup**
   ```bash
   # Run tests
   poetry run pytest

   # Check code style
   poetry run ruff check .

   # Run type checking
   poetry run mypy src tests

   # Run a quick experiment to verify everything works
   poetry run python scripts/run_all_experiments.py --experiments digital
   ```

## 🌿 Branching and Commit Messages

### Branch Naming

Create small, focused branches with descriptive names:

- **Features:** `feature/add-american-option-pricing`
- **Fixes:** `fix/correct-gamma-calculation`
- **Documentation:** `docs/improve-basket-option-docs`
- **Refactoring:** `refactor/optimize-monte-carlo-loop`

### Commit Messages

We encourage conventional commit messages for clarity and automation:

- `feat: add support for American option pricing`
- `fix: correct LRM delta weight in barrier dataset`
- `docs: update README with CI badge`
- `test: add edge cases for zero volatility`
- `refactor: extract common MC logic to utils`
- `perf: vectorize pathwise delta computation`
- `chore: update dependencies`

### Keeping Your Branch Updated

Before submitting a PR, rebase your branch on the latest `main`:

```bash
git fetch upstream
git rebase upstream/main
```

## 🔄 Pull Requests

### Before Submitting

Please ensure your PR meets these criteria:

- [ ] **Tests pass** - CI should be green (`poetry run pytest`)
- [ ] **Code style clean** - No ruff warnings (`poetry run ruff check .`)
- [ ] **Type checking passes** - mypy is happy (`poetry run mypy src tests`)
- [ ] **Documentation added** - Docstrings for new functions, especially those with mathematical operations
- [ ] **Tests added** - New functionality should have corresponding tests
- [ ] **Experiments verified** - If modifying experiment code, verify outputs are sensible

### PR Description

In your pull request:

1. **Link the issue** - Reference any related issue (e.g., `Fixes #123` or `Relates to #45`)
2. **Describe changes** - Brief summary of what and why
3. **Show impact** - If applicable, include before/after experiment outputs
4. **Note breaking changes** - Highlight any API or behavior changes

Example PR description:
```markdown
## Summary
Fixes incorrect gamma calculation in portfolio hedging experiment.

## Changes
- Corrected second-order derivative computation in `nn_value_delta_gamma`
- Added test cases for gamma near ATM strikes
- Updated experiment output to show gamma convergence

Fixes #42
```

## 💻 Coding Style

### General Guidelines

- **Python 3.10+ required** - Use modern Python features where appropriate
- **Type hints mandatory** - All new functions and methods must have type annotations
- **Small functions preferred** - Keep functions focused and testable
- **Explicit over implicit** - Avoid magic globals; pass parameters explicitly
- **Double precision default** - Use `torch.float64` for numerical experiments unless there's a specific reason not to

### Mathematical Code

When implementing mathematical concepts:

1. **Follow paper notation** - Variable names should match the paper where it improves clarity
2. **Comment the math** - Add comments with mathematical formulas for complex operations
3. **Reference equations** - Reference specific equations from the paper

Example:
```python
def bs_digital_delta(x: Tensor, K: float, params: BSParams) -> Tensor:
    """Compute Black-Scholes delta for digital call option.

    Delta = dV/dS = phi(d2) / (S * sigma * sqrt(T))
    where d2 = (log(S/K) + (r - sigma^2/2)*T) / (sigma * sqrt(T))

    Reference: Equation (3.2) in the paper
    """
    # ... implementation ...
```

### Code Organization

- Place new experiments in `src/diffml/experiments_*.py`
- Dataset generators go in `src/diffml/datasets_*.py`
- Utility functions in appropriate module (e.g., `simulation.py`, `bs_analytics.py`)
- Always add corresponding tests in `tests/`

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please use our GitHub issue templates and include:

- **Environment details**
  - Python version: `python --version`
  - PyTorch version: `python -c "import torch; print(torch.__version__)"`
  - OS and hardware (especially if GPU-related)

- **Reproduction steps**
  - Exact command line used
  - Any modifications to default configurations
  - Minimal code example if applicable

- **Expected vs actual behavior**
  - What you expected to happen
  - What actually happened
  - Error messages or stack traces

### Experiment Mismatches

If you find discrepancies with the paper's results:

1. **Verify your setup**
   - Using default hyperparameters?
   - Correct random seeds?
   - Sufficient Monte Carlo paths?

2. **Document the mismatch**
   - Paper's reported values
   - Your observed values
   - Configuration used

3. **Consider numerical factors**
   - Monte Carlo variance
   - Different random number generators
   - Floating-point precision differences

## 🧪 Testing

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_bs_analytics.py

# Run with coverage
poetry run pytest --cov=diffml

# Run only fast tests (skip slow ones)
poetry run pytest -m "not slow"
```

### Writing Tests

- Test both happy paths and edge cases
- Use small data sizes for speed
- Test shapes, value ranges, and numerical properties
- Use fixed seeds for reproducibility

Example test:
```python
def test_digital_price_limits(bs_params: BSParams) -> None:
    """Test digital price at extreme spot values."""
    K = 100.0

    # Deep ITM should approach discount factor
    s_high = torch.tensor([[1000.0]], dtype=torch.float64)
    price_high = bs_digital_price(s_high, K, bs_params)
    discount = torch.exp(-bs_params.r * bs_params.T)
    assert torch.allclose(price_high, discount, rtol=1e-2)
```

## 📚 Documentation

### Docstrings

Use NumPy-style docstrings for all public functions:

```python
def make_digital_dataset(
    m: int,
    K: float,
    params: BSParams,
    x_min: float = 80.0,
    x_max: float = 120.0,
    n_paths_per_x: int = 100,
    seed: Optional[int] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate dataset for digital option pricing.

    Parameters
    ----------
    m : int
        Number of spot price points
    K : float
        Strike price
    params : BSParams
        Black-Scholes parameters

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor]
        Spot prices, option prices, pathwise deltas, LRM deltas
    """
```

### README Updates

If adding new features or experiments, update the README to:
- List the new functionality
- Provide usage examples
- Update any relevant tables or figures

## 📝 License

This project is licensed under the MIT License. By contributing, you agree that your contributions will be licensed under the same license. See the [LICENSE](LICENSE) file for details.

## 🙏 Thank You!

Your contributions help make this project better for everyone interested in differential machine learning. Whether you're fixing a typo, adding a test, or implementing a new experiment, every contribution is valued!

If you have questions or need help getting started, feel free to:
- Open a discussion on GitHub
- Check existing issues and PRs for similar topics
- Reach out to maintainers

Happy contributing! 🚀