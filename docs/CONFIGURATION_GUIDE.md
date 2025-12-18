# Configuration Guide

This guide explains how to use the configuration-based experiment system in DiffML.

## Overview

The DiffML library now supports configuration-based experiments using TOML files. This allows you to:
- Run experiments with different parameters without modifying code
- Easily reproduce experiments with saved configurations
- Manage multiple experiment variants
- Share configurations with collaborators

## Configuration Structure

Configuration files use the TOML format with the following sections:

### `[experiment]` Section
Basic experiment metadata:
```toml
[experiment]
name = "digital"  # Registered experiment name
seed = 42         # Random seed for reproducibility
```

### `[training]` Section
Training hyperparameters:
```toml
[training]
n_epochs = 2000      # Number of training epochs
batch_size = 256     # Batch size
lr_initial = 1e-3    # Initial learning rate
lr_min = 1e-6        # Minimum learning rate for scheduler
lambda_delta = 1.0   # Weight for delta regularization
lambda_gamma = 0.0   # Weight for gamma regularization
```

### `[dataset]` Section
Dataset generation parameters:
```toml
[dataset]
m_train = 1024       # Number of training samples
m_test = 256         # Number of test samples
n_paths_train = 10000   # MC paths for training
n_paths_test = 50000    # MC paths for testing
x_min = 0.5          # Min spot as fraction of K
x_max = 1.5          # Max spot as fraction of K
```

### `[payoff]` Section
Option payoff parameters:
```toml
[payoff]
K = 100.0            # Strike price
B = 0.85             # Barrier level (for barrier options)
H = 120.0            # Upper barrier (for up-and-out)
L = 80.0             # Lower barrier (for double barrier)
d = 20               # Dimension (for basket options)
n_steps = 16         # Time steps (for path-dependent options)
eps_multipliers = [0.5, 1.0, 2.0]  # For smoothing experiments
```

### `[black_scholes]` Section
Market parameters:
```toml
[black_scholes]
r = 0.05             # Risk-free rate
sigma = 0.2          # Volatility
T = 0.5              # Time to maturity
```

## Using the Configuration System

### 1. Command Line Interface

```bash
# List available experiments
python scripts/run_experiment.py --list

# Run with configuration
python scripts/run_experiment.py --config configs/digital_default.toml

# Validate configuration
python scripts/run_experiment.py --validate configs/my_config.toml

# Run with verbose output
python scripts/run_experiment.py --config configs/digital_default.toml --verbose
```

### 2. Programmatic Usage

```python
from diffml.config_experiments import load_experiment_config
from diffml.run_experiment import run_experiment_from_config

# Load and run
config_path = "configs/digital_default.toml"
run_experiment_from_config(config_path)

# Or load config and modify
config = load_experiment_config(config_path)
config.seed = 123  # Override seed
config.training.n_epochs = 500  # Fewer epochs

# Run with modified config
from diffml.experiments_registry import get_experiment
experiment_func = get_experiment(config.name)
experiment_func(config)
```

### 3. Creating Custom Configurations

Create a new TOML file with your parameters:

```toml
# my_custom_config.toml
[experiment]
name = "digital"
seed = 999

[training]
n_epochs = 3000
batch_size = 128
lr_initial = 5e-4
lambda_delta = 2.0  # Stronger delta regularization

[dataset]
m_train = 2048      # More training data
m_test = 512
n_paths_train = 20000
n_paths_test = 100000

[payoff]
K = 110.0           # Different strike

[black_scholes]
r = 0.03
sigma = 0.25        # Higher volatility
T = 1.0             # Longer maturity
```

## Experiment Registry

### Available Experiments

The following experiments are registered and can be used in configurations:

1. **`digital`** - Digital option with discontinuous payoff
2. **`barrier`** - Down-and-out barrier option
3. **`basket`** - High-dimensional basket option
4. **`asian`** - Arithmetic Asian option
5. **`smoothing`** - Smoothing technique comparison

### Registering New Experiments

To register a new experiment:

```python
from diffml.experiments_registry import register_experiment
from diffml.config_experiments import ExperimentConfig

@register_experiment("my_experiment")
def run_my_experiment(config: ExperimentConfig) -> None:
    """My custom experiment."""
    print(f"Running {config.name} with seed {config.seed}")
    # Experiment implementation...
```

## Configuration Examples

### Digital Option with Strong Regularization

```toml
[experiment]
name = "digital"
seed = 42

[training]
n_epochs = 3000
lambda_delta = 5.0  # Very strong delta regularization

[dataset]
m_train = 2048
n_paths_train = 50000  # More MC paths

[payoff]
K = 100.0

[black_scholes]
r = 0.0
sigma = 0.3  # Higher volatility
T = 0.25
```

### Quick Test Configuration

```toml
[experiment]
name = "digital"
seed = 123

[training]
n_epochs = 100  # Quick training
batch_size = 32

[dataset]
m_train = 100   # Small dataset
m_test = 50
n_paths_train = 100
n_paths_test = 500

[payoff]
K = 1.0

[black_scholes]
r = 0.0
sigma = 0.2
T = 0.25
```

### High-Dimensional Basket Option

```toml
[experiment]
name = "basket"
seed = 42

[training]
n_epochs = 3000
batch_size = 512

[dataset]
m_train = 4096  # More samples for high dimension
m_test = 1024

[payoff]
K = 100.0
d = 50  # 50-dimensional basket

[black_scholes]
r = 0.05
sigma = 0.15
T = 1.0
```

## Tips and Best Practices

1. **Start from defaults**: Copy an existing configuration and modify it
2. **Use descriptive names**: Name configs by their purpose (e.g., `digital_high_vol.toml`)
3. **Document changes**: Add comments explaining non-standard parameters
4. **Version control**: Track configuration files in git
5. **Validate first**: Use `--validate` before long-running experiments
6. **Small tests**: Create small configs for testing changes quickly

## Troubleshooting

### Common Issues

1. **Experiment not found**: Ensure the experiment name matches a registered experiment
2. **Missing parameters**: Check that all required sections are present
3. **Invalid types**: TOML is strict about types (use `1.0` not `1` for floats)
4. **Import errors**: Ensure the package is installed (`poetry install`)

### Debug Mode

Run with `--verbose` to see detailed error messages:
```bash
python scripts/run_experiment.py --config configs/my_config.toml --verbose
```

## API Reference

### `ExperimentConfig`
Main configuration dataclass with all experiment parameters.

### `load_experiment_config(path: str) -> ExperimentConfig`
Load configuration from TOML file.

### `run_experiment_from_config(config_path: str) -> None`
Load and run experiment from configuration file.

### `@register_experiment(name: str)`
Decorator to register an experiment function.

### `get_experiment(name: str) -> ExperimentFunc`
Retrieve registered experiment by name.

## Future Enhancements

Planned improvements to the configuration system:
- YAML support alongside TOML
- Configuration inheritance/composition
- Parameter sweeps and grid search
- Automatic hyperparameter optimization
- Configuration validation schemas
- Web-based configuration editor