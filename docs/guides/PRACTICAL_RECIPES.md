# DiffML Practical Recipes

A collection of ready-to-use code recipes for common tasks in differential machine learning for finance.

## Table of Contents

1. [Quick Start Recipes](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
4. [Option Pricing](#option-pricing)
5. [Risk Management](#risk-management)
6. [Performance Optimization](#performance-optimization)
7. [Visualization](#visualization)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)

---

## 1. Quick Start Recipes {#quick-start}

### Recipe 1.1: Your First DML Model in 5 Minutes

```python
import torch
from diffml import DifferentialRegressor, BlackScholesDataset, DifferentialTrainer

# Generate data
dataset = BlackScholesDataset(n_samples=10000, option_type='call')
X, y, dy = dataset.generate()

# Create model
model = DifferentialRegressor(
    input_dim=5,
    hidden_units=[64, 64, 64],
    activation='relu'
)

# Train
trainer = DifferentialTrainer(model, differential_weight=0.5)
trainer.fit(X, y, dy, epochs=50)

# Price an option
test_input = torch.tensor([[100, 100, 1.0, 0.05, 0.2]])  # S, K, T, r, σ
price = model(test_input)
print(f"Option price: ${price.item():.2f}")
```

### Recipe 1.2: Quick Greeks Calculation

```python
def calculate_all_greeks(model, spot, strike, maturity, rate, vol):
    """Calculate all Greeks for an option."""
    x = torch.tensor([[spot, strike, maturity, rate, vol]],
                     requires_grad=True)

    # Price
    price = model(x)

    # First-order Greeks
    grads = torch.autograd.grad(price, x, create_graph=True)[0]
    delta = grads[0, 0]  # ∂V/∂S
    vega = grads[0, 4]   # ∂V/∂σ
    theta = -grads[0, 2]  # -∂V/∂T
    rho = grads[0, 3]    # ∂V/∂r

    # Gamma (second-order)
    gamma = torch.autograd.grad(delta, x, retain_graph=True)[0][0, 0]

    return {
        'price': price.item(),
        'delta': delta.item(),
        'gamma': gamma.item(),
        'vega': vega.item(),
        'theta': theta.item(),
        'rho': rho.item()
    }

# Example usage
greeks = calculate_all_greeks(model, 100, 100, 1.0, 0.05, 0.2)
for name, value in greeks.items():
    print(f"{name}: {value:.4f}")
```

---

## 2. Data Preparation {#data-preparation}

### Recipe 2.1: Generate Diverse Training Data

```python
def generate_diverse_dataset(n_samples=50000):
    """Generate training data covering various market conditions."""

    # Market regimes
    normal_market = n_samples // 3
    volatile_market = n_samples // 3
    stressed_market = n_samples - 2 * (n_samples // 3)

    datasets = []

    # Normal market (σ = 10-30%)
    data1 = BlackScholesDataset(
        n_samples=normal_market,
        spot_range=(80, 120),
        vol_range=(0.10, 0.30),
        maturity_range=(0.1, 2.0)
    )
    datasets.append(data1.generate())

    # Volatile market (σ = 30-60%)
    data2 = BlackScholesDataset(
        n_samples=volatile_market,
        spot_range=(60, 140),
        vol_range=(0.30, 0.60),
        maturity_range=(0.05, 1.0)
    )
    datasets.append(data2.generate())

    # Stressed market (σ = 60-100%)
    data3 = BlackScholesDataset(
        n_samples=stressed_market,
        spot_range=(50, 150),
        vol_range=(0.60, 1.00),
        maturity_range=(0.01, 0.5)
    )
    datasets.append(data3.generate())

    # Combine datasets
    X = torch.cat([d[0] for d in datasets])
    y = torch.cat([d[1] for d in datasets])
    dy = torch.cat([d[2] for d in datasets])

    # Shuffle
    indices = torch.randperm(len(X))
    return X[indices], y[indices], dy[indices]
```

### Recipe 2.2: Add Market Data Augmentation

```python
def augment_market_data(X, y, dy, noise_level=0.01):
    """Add realistic noise to training data."""

    # Price noise (bid-ask spread simulation)
    price_noise = torch.randn_like(y) * noise_level * y
    y_augmented = y + price_noise

    # Greek noise (numerical error simulation)
    greek_noise = torch.randn_like(dy) * noise_level
    dy_augmented = dy + greek_noise

    # Feature perturbation (market microstructure)
    feature_noise = torch.randn_like(X) * noise_level
    X_augmented = X + feature_noise

    # Ensure positive values where needed
    X_augmented[:, 0] = torch.abs(X_augmented[:, 0])  # Spot > 0
    X_augmented[:, 1] = torch.abs(X_augmented[:, 1])  # Strike > 0
    X_augmented[:, 2] = torch.abs(X_augmented[:, 2])  # Maturity > 0
    X_augmented[:, 4] = torch.abs(X_augmented[:, 4])  # Vol > 0

    return X_augmented, y_augmented, dy_augmented
```

### Recipe 2.3: Create Time Series Features

```python
def create_time_series_features(spot_history, window=20):
    """Create features from historical spot prices."""

    features = []

    # Returns
    returns = torch.diff(torch.log(spot_history))

    # Moving averages
    ma_5 = torch.nn.functional.avg_pool1d(
        spot_history.unsqueeze(0).unsqueeze(0),
        kernel_size=5, stride=1
    ).squeeze()

    ma_20 = torch.nn.functional.avg_pool1d(
        spot_history.unsqueeze(0).unsqueeze(0),
        kernel_size=20, stride=1
    ).squeeze()

    # Realized volatility
    realized_vol = torch.std(returns[-window:]) * torch.sqrt(torch.tensor(252.0))

    # Technical indicators
    rsi = compute_rsi(spot_history, window=14)

    features = torch.tensor([
        spot_history[-1],           # Current spot
        ma_5[-1] / spot_history[-1], # MA ratio
        ma_20[-1] / spot_history[-1],
        realized_vol,
        rsi
    ])

    return features

def compute_rsi(prices, window=14):
    """Compute Relative Strength Index."""
    deltas = torch.diff(prices)
    gains = torch.where(deltas > 0, deltas, torch.tensor(0.0))
    losses = torch.where(deltas < 0, -deltas, torch.tensor(0.0))

    avg_gain = torch.mean(gains[-window:])
    avg_loss = torch.mean(losses[-window:])

    if avg_loss == 0:
        return torch.tensor(100.0)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi
```

---

## 3. Model Training {#model-training}

### Recipe 3.1: Advanced Training with Validation

```python
def train_with_validation(model, X, y, dy, val_split=0.2, patience=10):
    """Train model with validation and early stopping."""

    # Split data
    n_val = int(len(X) * val_split)
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]
    dy_train, dy_val = dy[:-n_val], dy[-n_val:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1000):
        # Training
        model.train()
        X_train.requires_grad_(True)

        y_pred = model(X_train)
        dy_pred = torch.autograd.grad(
            y_pred.sum(), X_train, create_graph=True
        )[0]

        loss_value = torch.mean((y_pred - y_train)**2)
        loss_greek = torch.mean((dy_pred - dy_train)**2)
        loss = 0.5 * loss_value + 0.5 * loss_greek

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = torch.mean((val_pred - y_val)**2)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={loss:.6f}, Val Loss={val_loss:.6f}")

    return model
```

### Recipe 3.2: Curriculum Learning

```python
def curriculum_training(model, datasets, epochs_per_stage=50):
    """Train model progressively on harder examples."""

    stages = [
        ("ATM options", lambda x: torch.abs(x[:, 0] - x[:, 1]) < 5),
        ("Near ATM", lambda x: torch.abs(x[:, 0] - x[:, 1]) < 20),
        ("All options", lambda x: torch.ones(len(x), dtype=torch.bool))
    ]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for stage_name, condition in stages:
        print(f"\nTraining stage: {stage_name}")

        # Filter data for this stage
        mask = condition(datasets[0])
        X_stage = datasets[0][mask]
        y_stage = datasets[1][mask]
        dy_stage = datasets[2][mask]

        # Train on this stage
        for epoch in range(epochs_per_stage):
            X_stage.requires_grad_(True)

            y_pred = model(X_stage)
            dy_pred = torch.autograd.grad(
                y_pred.sum(), X_stage, create_graph=True
            )[0]

            loss = torch.mean((y_pred - y_stage)**2) + \
                   torch.mean((dy_pred - dy_stage)**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss={loss:.6f}")

    return model
```

### Recipe 3.3: Ensemble Training

```python
def train_ensemble(n_models=5, X, y, dy):
    """Train ensemble of models for uncertainty quantification."""

    models = []

    for i in range(n_models):
        print(f"Training model {i+1}/{n_models}")

        # Different architecture for diversity
        hidden_units = [64 + i*16] * 3

        model = DifferentialRegressor(
            input_dim=X.shape[1],
            hidden_units=hidden_units,
            activation='relu' if i % 2 == 0 else 'tanh',
            dropout_rate=0.1 * (i + 1) / n_models
        )

        # Different data subset (bagging)
        indices = torch.randint(0, len(X), (len(X),))
        X_boot = X[indices]
        y_boot = y[indices]
        dy_boot = dy[indices]

        # Train
        trainer = DifferentialTrainer(
            model,
            differential_weight=0.3 + 0.1 * i  # Vary weight
        )
        trainer.fit(X_boot, y_boot, dy_boot, epochs=50, verbose=0)

        models.append(model)

    return models

def ensemble_predict(models, X, return_std=True):
    """Make predictions with uncertainty estimates."""

    predictions = []

    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(X)
            predictions.append(pred)

    predictions = torch.stack(predictions)
    mean_pred = torch.mean(predictions, dim=0)

    if return_std:
        std_pred = torch.std(predictions, dim=0)
        return mean_pred, std_pred

    return mean_pred
```

---

## 4. Option Pricing {#option-pricing}

### Recipe 4.1: Price Multiple Options Efficiently

```python
def batch_price_options(model, option_specs):
    """Price multiple options in batch for efficiency."""

    # Convert specs to tensor
    features = []
    for spec in option_specs:
        features.append([
            spec['spot'],
            spec['strike'],
            spec['maturity'],
            spec['rate'],
            spec['volatility']
        ])

    X = torch.tensor(features)

    # Batch inference
    model.eval()
    with torch.no_grad():
        prices = model(X)

    # Add to results
    results = []
    for i, spec in enumerate(option_specs):
        result = spec.copy()
        result['price'] = prices[i].item()
        results.append(result)

    return results

# Example usage
options = [
    {'spot': 100, 'strike': 95, 'maturity': 0.25, 'rate': 0.05, 'volatility': 0.2},
    {'spot': 100, 'strike': 100, 'maturity': 0.25, 'rate': 0.05, 'volatility': 0.2},
    {'spot': 100, 'strike': 105, 'maturity': 0.25, 'rate': 0.05, 'volatility': 0.2},
]

results = batch_price_options(model, options)
```

### Recipe 4.2: Implied Volatility Calculation

```python
def calculate_implied_vol(model, spot, strike, maturity, rate,
                         market_price, option_type='call'):
    """Calculate implied volatility using Newton's method."""

    # Initial guess
    vol = torch.tensor([0.2], requires_grad=True)

    for _ in range(20):  # Max iterations
        # Prepare input
        x = torch.tensor([[spot, strike, maturity, rate, vol.item()]])
        x.requires_grad_(True)

        # Model price and vega
        model_price = model(x)
        vega = torch.autograd.grad(model_price, x)[0][0, 4]

        # Price difference
        price_diff = model_price - market_price

        # Newton update
        if torch.abs(price_diff) < 1e-4:
            break

        vol_update = price_diff / vega
        vol = vol - vol_update

        # Ensure positive volatility
        vol = torch.maximum(vol, torch.tensor([0.001]))

    return vol.item()

# Example
implied_vol = calculate_implied_vol(
    model, spot=100, strike=100, maturity=0.25,
    rate=0.05, market_price=5.0
)
print(f"Implied volatility: {implied_vol:.2%}")
```

### Recipe 4.3: American Option Pricing

```python
def price_american_option(model, spot, strike, maturity, rate, vol,
                         n_steps=50, n_paths=10000):
    """Price American option using LSM with DML."""

    from diffml.datasets_american import AmericanOptionDataset

    # Generate paths
    dataset = AmericanOptionDataset(
        n_samples=1,
        n_steps=n_steps,
        n_paths=n_paths
    )

    # Override with specific parameters
    dataset.params.spot = spot
    dataset.params.strike = strike
    dataset.params.maturity = maturity
    dataset.params.rate = rate
    dataset.params.vol = vol

    # Generate paths and exercise decisions
    paths = dataset.generate_paths()

    # Use model for continuation values
    dt = maturity / n_steps
    exercise_values = []

    for t_idx in range(n_steps):
        current_spot = paths[:, t_idx]
        time_remaining = (n_steps - t_idx) * dt

        # Prepare features
        features = torch.stack([
            current_spot,
            torch.full_like(current_spot, strike),
            torch.full_like(current_spot, time_remaining),
            torch.full_like(current_spot, rate),
            torch.full_like(current_spot, vol)
        ], dim=1)

        # Continuation value from model
        with torch.no_grad():
            continuation = model(features).squeeze()

        # Exercise value
        exercise = torch.maximum(strike - current_spot,
                                torch.zeros_like(current_spot))

        # Optimal exercise decision
        should_exercise = exercise > continuation
        exercise_values.append(should_exercise)

    # Backward induction for price
    # ... (full LSM implementation)

    return american_price
```

---

## 5. Risk Management {#risk-management}

### Recipe 5.1: Portfolio Greeks Aggregation

```python
def calculate_portfolio_greeks(model, portfolio):
    """Calculate aggregate Greeks for a portfolio."""

    total_delta = 0
    total_gamma = 0
    total_vega = 0
    total_value = 0

    for position in portfolio:
        x = torch.tensor([[
            position['spot'],
            position['strike'],
            position['maturity'],
            position['rate'],
            position['volatility']
        ]], requires_grad=True)

        # Price and first-order Greeks
        price = model(x)
        grads = torch.autograd.grad(price, x, create_graph=True)[0]

        delta = grads[0, 0]
        vega = grads[0, 4]

        # Gamma
        gamma = torch.autograd.grad(delta, x)[0][0, 0]

        # Aggregate with position size
        quantity = position['quantity']
        total_value += quantity * price.item()
        total_delta += quantity * delta.item()
        total_gamma += quantity * gamma.item()
        total_vega += quantity * vega.item()

    return {
        'value': total_value,
        'delta': total_delta,
        'gamma': total_gamma,
        'vega': total_vega,
        'delta_$': total_delta * position['spot'],  # Dollar delta
        'gamma_$': total_gamma * position['spot']**2 / 100  # 1% gamma
    }
```

### Recipe 5.2: VaR Calculation

```python
def calculate_var(model, portfolio, confidence=0.95, n_scenarios=10000):
    """Calculate Value at Risk using Monte Carlo."""

    portfolio_values = []

    for _ in range(n_scenarios):
        # Generate market scenario
        spot_shock = torch.randn(1) * 0.02  # 2% daily vol
        vol_shock = torch.randn(1) * 0.05   # 5% vol of vol

        scenario_value = 0

        for position in portfolio:
            # Shocked parameters
            shocked_spot = position['spot'] * (1 + spot_shock)
            shocked_vol = position['volatility'] * (1 + vol_shock)

            x = torch.tensor([[
                shocked_spot,
                position['strike'],
                position['maturity'] - 1/252,  # One day forward
                position['rate'],
                shocked_vol
            ]])

            with torch.no_grad():
                price = model(x)
                scenario_value += position['quantity'] * price.item()

        portfolio_values.append(scenario_value)

    # Calculate VaR
    portfolio_values = torch.tensor(portfolio_values)
    current_value = calculate_portfolio_greeks(model, portfolio)['value']

    pnl = portfolio_values - current_value
    var = -torch.quantile(pnl, 1 - confidence)
    cvar = -torch.mean(pnl[pnl < -var])

    return {
        'var_95': var.item(),
        'cvar_95': cvar.item(),
        'current_value': current_value
    }
```

### Recipe 5.3: Dynamic Hedging

```python
def dynamic_hedge(model, option_position, hedge_freq='daily'):
    """Implement dynamic delta hedging strategy."""

    hedge_history = []

    # Simulation parameters
    n_days = int(option_position['maturity'] * 252)
    dt = 1/252

    current_spot = option_position['spot']
    current_hedge = 0

    for day in range(n_days):
        # Calculate current delta
        x = torch.tensor([[
            current_spot,
            option_position['strike'],
            option_position['maturity'] - day * dt,
            option_position['rate'],
            option_position['volatility']
        ]], requires_grad=True)

        price = model(x)
        delta = torch.autograd.grad(price, x)[0][0, 0]

        # Rebalance hedge
        target_hedge = -delta.item()  # Short delta amount of stock
        trade = target_hedge - current_hedge

        # Simulate next day's spot (GBM)
        z = torch.randn(1)
        current_spot = current_spot * torch.exp(
            (option_position['rate'] - 0.5 * option_position['volatility']**2) * dt +
            option_position['volatility'] * torch.sqrt(dt) * z
        )

        # Record
        hedge_history.append({
            'day': day,
            'spot': current_spot.item(),
            'delta': delta.item(),
            'hedge_position': target_hedge,
            'trade': trade,
            'option_value': price.item()
        })

        current_hedge = target_hedge

    return pd.DataFrame(hedge_history)
```

---

## 6. Performance Optimization {#performance-optimization}

### Recipe 6.1: GPU Acceleration

```python
def optimize_for_gpu(model, batch_size=1024):
    """Optimize model for GPU inference."""

    if not torch.cuda.is_available():
        print("GPU not available")
        return model

    # Move model to GPU
    model = model.cuda()
    model.eval()

    # Compile with TorchScript for faster execution
    example_input = torch.randn(batch_size, 5).cuda()
    traced_model = torch.jit.trace(model, example_input)

    # Warm up GPU
    for _ in range(10):
        _ = traced_model(example_input)

    # Benchmark
    import time

    start = time.time()
    for _ in range(100):
        _ = traced_model(example_input)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    throughput = (100 * batch_size) / elapsed
    print(f"GPU throughput: {throughput:.0f} options/second")

    return traced_model
```

### Recipe 6.2: Mixed Precision Training

```python
def train_mixed_precision(model, X, y, dy):
    """Train with automatic mixed precision for speed."""

    from torch.cuda.amp import autocast, GradScaler

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = GradScaler()

    X, y, dy = X.cuda(), y.cuda(), dy.cuda()

    for epoch in range(100):
        X.requires_grad_(True)

        # Mixed precision forward pass
        with autocast():
            y_pred = model(X)

            # Manual gradient computation in float32
            dy_pred = torch.autograd.grad(
                y_pred.sum(), X,
                create_graph=True
            )[0]

            loss = torch.mean((y_pred - y)**2) + \
                   torch.mean((dy_pred - dy)**2)

        # Scaled backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss:.6f}")
```

### Recipe 6.3: Caching and Memoization

```python
class CachedPricer:
    """Cache frequently requested option prices."""

    def __init__(self, model, cache_size=10000):
        self.model = model
        self.cache = {}
        self.cache_size = cache_size

    def price(self, spot, strike, maturity, rate, vol):
        """Price with caching."""

        # Create cache key
        key = (
            round(spot, 2),
            round(strike, 2),
            round(maturity, 4),
            round(rate, 4),
            round(vol, 4)
        )

        # Check cache
        if key in self.cache:
            return self.cache[key]

        # Compute price
        x = torch.tensor([[spot, strike, maturity, rate, vol]])
        with torch.no_grad():
            price = self.model(x).item()

        # Update cache (LRU style)
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))

        self.cache[key] = price

        return price

    def cache_stats(self):
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'capacity': self.cache_size,
            'utilization': len(self.cache) / self.cache_size
        }
```

---

## 7. Visualization {#visualization}

### Recipe 7.1: Plot Greeks Surface

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_greeks_surface(model, strike=100, maturity=1.0, rate=0.05):
    """3D visualization of Greeks across spot and volatility."""

    spots = torch.linspace(80, 120, 50)
    vols = torch.linspace(0.1, 0.5, 50)

    S, V = torch.meshgrid(spots, vols, indexing='ij')

    # Calculate Greeks for grid
    delta_surface = torch.zeros_like(S)
    gamma_surface = torch.zeros_like(S)

    for i in range(len(spots)):
        for j in range(len(vols)):
            x = torch.tensor([[
                spots[i], strike, maturity, rate, vols[j]
            ]], requires_grad=True)

            price = model(x)
            grads = torch.autograd.grad(price, x, create_graph=True)[0]

            delta_surface[i, j] = grads[0, 0]

            gamma = torch.autograd.grad(grads[0, 0], x)[0][0, 0]
            gamma_surface[i, j] = gamma

    # Plot
    fig = plt.figure(figsize=(15, 6))

    # Delta surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(S.numpy(), V.numpy(), delta_surface.detach().numpy(),
                     cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Spot Price')
    ax1.set_ylabel('Volatility')
    ax1.set_zlabel('Delta')
    ax1.set_title('Delta Surface')

    # Gamma surface
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(S.numpy(), V.numpy(), gamma_surface.detach().numpy(),
                     cmap='plasma', alpha=0.8)
    ax2.set_xlabel('Spot Price')
    ax2.set_ylabel('Volatility')
    ax2.set_zlabel('Gamma')
    ax2.set_title('Gamma Surface')

    plt.tight_layout()
    plt.show()
```

### Recipe 7.2: Training Progress Dashboard

```python
def create_training_dashboard(history):
    """Create interactive training progress dashboard."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Loss', 'Value Loss', 'Greek Loss', 'Learning Rate')
    )

    epochs = list(range(len(history['total_loss'])))

    # Total loss
    fig.add_trace(
        go.Scatter(x=epochs, y=history['total_loss'], name='Total Loss'),
        row=1, col=1
    )

    # Value loss
    fig.add_trace(
        go.Scatter(x=epochs, y=history['value_loss'], name='Value Loss'),
        row=1, col=2
    )

    # Greek loss
    fig.add_trace(
        go.Scatter(x=epochs, y=history['greek_loss'], name='Greek Loss'),
        row=2, col=1
    )

    # Learning rate
    fig.add_trace(
        go.Scatter(x=epochs, y=history['lr'], name='Learning Rate'),
        row=2, col=2
    )

    fig.update_layout(height=600, showlegend=False,
                     title_text="Training Progress Dashboard")
    fig.show()
```

---

## 8. Production Deployment {#production-deployment}

### Recipe 8.1: Model Serving with FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

class OptionRequest(BaseModel):
    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float

class OptionResponse(BaseModel):
    price: float
    delta: float
    gamma: float
    vega: float

app = FastAPI(title="DiffML Option Pricer")

# Load model once at startup
model = torch.jit.load("model_production.pt")
model.eval()

@app.post("/price", response_model=OptionResponse)
async def price_option(request: OptionRequest):
    """Price option and calculate Greeks."""

    try:
        x = torch.tensor([[
            request.spot,
            request.strike,
            request.maturity,
            request.rate,
            request.volatility
        ]], requires_grad=True)

        # Calculate price and Greeks
        price = model(x)
        grads = torch.autograd.grad(price, x, create_graph=True)[0]

        delta = grads[0, 0]
        vega = grads[0, 4]
        gamma = torch.autograd.grad(delta, x)[0][0, 0]

        return OptionResponse(
            price=price.item(),
            delta=delta.item(),
            gamma=gamma.item(),
            vega=vega.item()
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Recipe 8.2: Model Versioning

```python
def save_model_with_metadata(model, version, metrics, path="models/"):
    """Save model with version and performance metadata."""

    import json
    from datetime import datetime

    # Create version directory
    version_path = f"{path}/v{version}/"
    os.makedirs(version_path, exist_ok=True)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': {
            'input_dim': model.input_dim,
            'hidden_units': model.hidden_units,
            'activation': model.activation
        }
    }, f"{version_path}/model.pt")

    # Save metadata
    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'framework_version': torch.__version__,
        'python_version': sys.version
    }

    with open(f"{version_path}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Model v{version} saved to {version_path}")

def load_model_version(version, path="models/"):
    """Load specific model version."""

    version_path = f"{path}/v{version}/"

    # Load metadata
    with open(f"{version_path}/metadata.json", 'r') as f:
        metadata = json.load(f)

    # Load model
    checkpoint = torch.load(f"{version_path}/model.pt")

    model = DifferentialRegressor(
        input_dim=checkpoint['architecture']['input_dim'],
        hidden_units=checkpoint['architecture']['hidden_units'],
        activation=checkpoint['architecture']['activation']
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, metadata
```

---

## 9. Troubleshooting {#troubleshooting}

### Recipe 9.1: Debug NaN/Inf Issues

```python
def debug_training(model, X, y, dy):
    """Debug training issues with detailed logging."""

    # Check input data
    print("Data Statistics:")
    print(f"  X: min={X.min():.4f}, max={X.max():.4f}, nan={torch.isnan(X).any()}")
    print(f"  y: min={y.min():.4f}, max={y.max():.4f}, nan={torch.isnan(y).any()}")
    print(f"  dy: min={dy.min():.4f}, max={dy.max():.4f}, nan={torch.isnan(dy).any()}")

    # Add hooks to monitor gradients
    def hook_fn(module, grad_input, grad_output):
        if any(torch.isnan(g).any() for g in grad_output if g is not None):
            print(f"NaN detected in {module.__class__.__name__}")
            return tuple(torch.zeros_like(g) if g is not None else None
                        for g in grad_output)

    # Register hooks
    for module in model.modules():
        module.register_backward_hook(hook_fn)

    # Test forward pass
    X_test = X[:10].clone().requires_grad_(True)
    y_test = y[:10].clone()
    dy_test = dy[:10].clone()

    try:
        # Forward
        y_pred = model(X_test)
        print(f"Forward pass OK: output shape={y_pred.shape}")

        # Gradient computation
        dy_pred = torch.autograd.grad(y_pred.sum(), X_test, create_graph=True)[0]
        print(f"Gradient computation OK: grad shape={dy_pred.shape}")

        # Loss
        loss = torch.mean((y_pred - y_test)**2) + torch.mean((dy_pred - dy_test)**2)
        print(f"Loss computation OK: loss={loss:.6f}")

        # Backward
        loss.backward()
        print("Backward pass OK")

        # Check parameter gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 100:
                    print(f"  Large gradient in {name}: {grad_norm:.2f}")
                if torch.isnan(param.grad).any():
                    print(f"  NaN gradient in {name}")

    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()
```

### Recipe 9.2: Fix Convergence Issues

```python
def diagnose_convergence(model, X, y, dy, n_epochs=100):
    """Diagnose convergence problems."""

    # Try different hyperparameters
    experiments = [
        {'lr': 0.001, 'weight': 0.5, 'optimizer': 'Adam'},
        {'lr': 0.0001, 'weight': 0.5, 'optimizer': 'Adam'},
        {'lr': 0.001, 'weight': 0.3, 'optimizer': 'Adam'},
        {'lr': 0.001, 'weight': 0.7, 'optimizer': 'Adam'},
        {'lr': 0.01, 'weight': 0.5, 'optimizer': 'SGD'},
    ]

    results = []

    for exp in experiments:
        model_copy = copy.deepcopy(model)

        if exp['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model_copy.parameters(), lr=exp['lr'])
        else:
            optimizer = torch.optim.SGD(model_copy.parameters(), lr=exp['lr'])

        losses = []

        for epoch in range(n_epochs):
            X.requires_grad_(True)

            y_pred = model_copy(X)
            dy_pred = torch.autograd.grad(y_pred.sum(), X, create_graph=True)[0]

            loss_val = torch.mean((y_pred - y)**2)
            loss_greek = torch.mean((dy_pred - dy)**2)
            loss = (1 - exp['weight']) * loss_val + exp['weight'] * loss_greek

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        results.append({
            'config': exp,
            'final_loss': losses[-1],
            'converged': losses[-1] < losses[0] * 0.1
        })

    # Report results
    print("Convergence Diagnosis:")
    for r in results:
        print(f"  Config: {r['config']}")
        print(f"    Final loss: {r['final_loss']:.6f}")
        print(f"    Converged: {r['converged']}")

    # Recommend best configuration
    best = min(results, key=lambda x: x['final_loss'])
    print(f"\nBest configuration: {best['config']}")
```

---

## Summary

This cookbook provides practical, ready-to-use recipes for implementing differential machine learning in quantitative finance. Each recipe is designed to be:

- **Self-contained**: Copy and use directly
- **Customizable**: Adapt to your specific needs
- **Production-ready**: Include error handling and best practices
- **Well-documented**: Clear comments and explanations

Start with the quick start recipes, then explore specific areas based on your needs. Remember to always validate your models against known benchmarks before deploying to production!

---

*For more recipes and contributions, visit the [DiffML GitHub repository](https://github.com/DiogoRibeiro7/diffml)*