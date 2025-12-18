# New Features: Path-Dependent and Extended Barrier Options

This document describes the new extensions to the DiffML library for path-dependent options and additional barrier option variants.

## Overview

The DiffML library has been extended with support for:
1. **Time-discretized path simulation** for Monte Carlo pricing
2. **Path-dependent options** (Asian and lookback options)
3. **Extended barrier options** (up-and-out, double barrier, down-and-in)

All new features integrate seamlessly with the existing DML framework for training neural networks with sensitivity regularization.

## 1. Enhanced Simulation Capabilities

### New Functions in `simulation.py`

#### `build_time_grid(T, n_steps)`
Creates a uniform time grid from 0 to T with n_steps + 1 points.

```python
from diffml import build_time_grid

grid = build_time_grid(T=1.0, n_steps=252)  # Daily steps for 1 year
```

#### `simulate_bs_paths(spots, params, n_steps, n_paths, seed)`
Simulates full Black-Scholes paths with discrete time steps.

```python
from diffml import simulate_bs_paths, BSParams

params = BSParams(r=0.05, sigma=0.2, T=1.0)
spots = torch.tensor([[100.0]])
paths, xi = simulate_bs_paths(spots, params, n_steps=252, n_paths=1000)
# paths shape: (1, 1000, 253) - includes time 0
```

## 2. Path-Dependent Options

### Arithmetic Asian Call Options

Asian options have payoffs based on the average stock price over the option's life.

```python
from diffml import make_arithmetic_asian_call_dataset

x, price, delta_pw, delta_lrm = make_arithmetic_asian_call_dataset(
    m=100,          # Number of spot prices
    K=100.0,        # Strike price
    params=params,  # BSParams object
    n_steps=16,     # Time discretization steps
    n_paths_per_x=500,  # MC paths per spot
    seed=42
)
```

**Key Features:**
- Arithmetic average includes time 0
- Pathwise delta: Uses chain rule with dA/dS0 = avg(paths)/S0
- LRM delta: Based on sum of all path increments

### Lookback Call Options

Lookback options have payoffs based on the maximum (or minimum) stock price.

```python
from diffml import make_lookback_call_dataset

x, price, delta_pw, delta_lrm = make_lookback_call_dataset(
    m=100,
    K=100.0,
    params=params,
    n_steps=32,     # More steps for better max approximation
    n_paths_per_x=500
)
```

**Key Features:**
- Fixed-strike lookback: Payoff = max(M - K, 0) where M = max_t S_t
- Pathwise delta: Approximation using dM/dS0 ≈ M/S0
- Generally more expensive than Asian options

## 3. Extended Barrier Options

### Up-and-Out Call

Knocks out (becomes worthless) if stock price exceeds upper barrier H.

```python
from diffml import make_up_and_out_call_dataset

x, price, delta_pw, delta_lrm = make_up_and_out_call_dataset(
    m=50,
    K=100.0,
    H=120.0,  # Upper barrier
    params=params,
    n_steps=16
)
```

### Double Barrier Call

Knocks out if stock price exits the range [L, H].

```python
from diffml import make_double_barrier_call_dataset

x, price, delta_pw, delta_lrm = make_double_barrier_call_dataset(
    m=50,
    K=100.0,
    L=80.0,   # Lower barrier
    H=120.0,  # Upper barrier
    params=params
)
```

### Down-and-In Call

Activates (knocks in) only if stock price touches lower barrier B.

```python
from diffml import make_down_and_in_call_dataset

x, price, delta_pw, delta_lrm = make_down_and_in_call_dataset(
    m=50,
    K=100.0,
    B=85.0,   # Lower knock-in barrier
    params=params
)
```

## Integration with DML Training

All new datasets are fully compatible with the existing DML training framework:

```python
from diffml import PricingNet, TrainingConfig, train_model
from torch.utils.data import TensorDataset

# Generate Asian option dataset
x, price, delta_pw, delta_lrm = make_arithmetic_asian_call_dataset(...)

# Create neural network
model = PricingNet(input_dim=1, hidden_dim=20, n_hidden=4)

# Configure training
config = TrainingConfig(
    n_epochs=1000,
    lambda_delta=1.0  # Weight for delta regularization
)

# Train with DML
dataset = TensorDataset(x, price, delta_pw, delta_lrm)
model = train_model(model, dataset, config, mode="delta_pathwise")
```

## Mathematical Details

### Pathwise Sensitivities

For path-dependent options, the pathwise derivative is computed using the chain rule:

- **Asian**: dV/dS0 = E[1_{A > K} × (average of paths)/S0]
- **Lookback**: dV/dS0 ≈ E[1_{M > K} × M/S0]
- **Barrier**: Includes knock-out/in indicator in expectation

### Likelihood Ratio Method (LRM)

The LRM score function for full paths is:
```
score = Σᵢ ξᵢ / (S₀ × σ × √T)
```
where ξᵢ are the standard normal increments.

## Testing

Comprehensive tests are provided in `tests/test_datasets_path_dependent.py`:

```bash
python -m pytest tests/test_datasets_path_dependent.py -v
```

Tests include:
- Shape and finiteness checks
- Price monotonicity and bounds
- Delta sign consistency
- Comparison between option types

## Example Usage

A complete demonstration is available in `examples/demo_path_dependent.py`:

```bash
python examples/demo_path_dependent.py
```

## Performance Considerations

1. **Memory**: Full path simulation requires O(m × n_paths × n_steps) memory
2. **Computation**: Asian/lookback options are more expensive than vanilla options
3. **Accuracy**: More time steps improve discrete monitoring approximation
4. **MC Variance**: Use more paths for stable sensitivity estimates

## References

- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*
- Broadie, M., & Glasserman, P. (1997). Pricing American-style securities using simulation
- Hull, J. C. (2018). *Options, Futures, and Other Derivatives*

## Future Extensions

Potential areas for further development:
- Continuous barrier monitoring approximations
- American option pricing via regression methods
- Variance reduction techniques (control variates, importance sampling)
- Additional exotic options (cliquet, Himalayan, etc.)