# Performance Profiling and Optimization Guide

## Overview

This guide provides comprehensive information on profiling and optimizing DiffML for maximum performance in production environments. DiffML leverages PyTorch's automatic differentiation capabilities combined with Monte Carlo simulation for pricing and hedging financial derivatives.

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Performance Profiling](#performance-profiling)
3. [GPU Optimization Strategies](#gpu-optimization-strategies)
4. [Memory Management](#memory-management)
5. [Batch Processing](#batch-processing)
6. [Monte Carlo Optimization](#monte-carlo-optimization)
7. [Neural Network Optimization](#neural-network-optimization)
8. [Benchmarking](#benchmarking)
9. [Production Deployment](#production-deployment)

## Hardware Requirements

### Minimum Requirements
- **CPU**: 4+ cores (Intel i5 or AMD Ryzen 5 equivalent)
- **RAM**: 8GB
- **GPU**: Optional (NVIDIA GTX 1060 or better for GPU acceleration)
- **Storage**: 2GB free space

### Recommended Requirements
- **CPU**: 8+ cores (Intel i7/i9 or AMD Ryzen 7/9)
- **RAM**: 16GB+
- **GPU**: NVIDIA RTX 3070 or better (8GB+ VRAM)
- **Storage**: SSD with 10GB+ free space

### Optimal Production Setup
- **CPU**: 16+ cores (Intel Xeon or AMD EPYC)
- **RAM**: 32GB+
- **GPU**: NVIDIA A100 or H100 (40GB+ VRAM)
- **Storage**: NVMe SSD

## Performance Profiling

### Using PyTorch Profiler

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from diffml.training import train_model

# Profile training loop
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("model_training"):
        train_model(model, train_loader, config)

# Print profiler results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export to Chrome tracing
prof.export_chrome_trace("trace.json")
```

### Using cProfile for CPU Profiling

```python
import cProfile
import pstats
from pstats import SortKey

# Profile experiment execution
profiler = cProfile.Profile()
profiler.enable()

# Run your experiment
from diffml.experiments_digital import run_digital_experiment
results = run_digital_experiment(config)

profiler.disable()

# Analyze results
stats = pstats.Stats(profiler)
stats.sort_stats(SortKey.CUMULATIVE)
stats.print_stats(20)  # Top 20 functions
```

### Memory Profiling with memory_profiler

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your code here
    pass

# Run with: python -m memory_profiler your_script.py
```

## GPU Optimization Strategies

### 1. Enable Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch_idx, (features, prices, deltas) in enumerate(train_loader):
        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            predictions = model(features)
            loss = dml_loss(predictions, prices, deltas, features, lambda_val)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 2. Optimize Data Loading

```python
# Use multiple workers and pin memory
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=2  # Prefetch batches
)
```

### 3. CUDA Stream Management

```python
import torch.cuda

# Create CUDA streams for parallel execution
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    # Monte Carlo simulation
    paths = simulate_bs_paths(S0, r, sigma, T, n_steps, n_paths)

with torch.cuda.stream(stream2):
    # Neural network forward pass
    predictions = model(features)

# Synchronize streams
torch.cuda.synchronize()
```

### 4. Tensor Core Utilization

```python
# Ensure dimensions are multiples of 8 for Tensor Core usage
def optimize_for_tensor_cores(tensor, target_multiple=8):
    """Pad tensor dimensions to multiples of 8 for Tensor Core optimization."""
    shape = list(tensor.shape)
    for i, dim in enumerate(shape):
        if dim % target_multiple != 0:
            shape[i] = ((dim // target_multiple) + 1) * target_multiple

    padded = torch.zeros(shape, device=tensor.device, dtype=tensor.dtype)
    slices = [slice(0, s) for s in tensor.shape]
    padded[slices] = tensor
    return padded
```

## Memory Management

### 1. Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedResNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            # Checkpoint intermediate layers to save memory
            x = checkpoint(layer, x)
        return x
```

### 2. Memory-Efficient Monte Carlo

```python
def simulate_paths_chunked(S0, r, sigma, T, n_steps, n_paths, chunk_size=10000):
    """Simulate paths in chunks to manage memory usage."""
    all_paths = []

    for i in range(0, n_paths, chunk_size):
        chunk_paths = min(chunk_size, n_paths - i)

        # Simulate chunk
        chunk = simulate_bs_terminal(
            S0, r, sigma, T, n_steps, chunk_paths
        )

        # Move to CPU to free GPU memory
        all_paths.append(chunk.cpu())

        # Clear GPU cache
        torch.cuda.empty_cache()

    return torch.cat(all_paths, dim=0)
```

### 3. Gradient Accumulation

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (features, prices, deltas) in enumerate(train_loader):
    predictions = model(features)
    loss = dml_loss(predictions, prices, deltas, features, lambda_val)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Batch Processing

### 1. Dynamic Batching

```python
def create_dynamic_batches(dataset, min_batch=32, max_batch=512):
    """Create batches with size based on available GPU memory."""
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated()
    available = gpu_mem - allocated

    # Estimate batch size based on available memory
    estimated_batch = min(max_batch, max(min_batch,
                          int(available / (1024**3) * 64)))  # 64 samples per GB

    return DataLoader(dataset, batch_size=estimated_batch, shuffle=True)
```

### 2. Vectorized Operations

```python
def vectorized_black_scholes(S, K, r, sigma, T):
    """Vectorized Black-Scholes for batch processing."""
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)

    # Batch normal CDF computation
    N_d1 = torch.distributions.Normal(0, 1).cdf(d1)
    N_d2 = torch.distributions.Normal(0, 1).cdf(d2)

    call_price = S * N_d1 - K * torch.exp(-r * T) * N_d2
    return call_price
```

## Monte Carlo Optimization

### 1. Quasi-Random Sequences

```python
from scipy.stats import qmc

def sobol_monte_carlo(n_paths, n_dims):
    """Use Sobol sequences for better convergence."""
    sampler = qmc.Sobol(d=n_dims, scramble=True)
    samples = sampler.random(n_paths)

    # Transform to normal distribution
    normal_samples = torch.tensor(
        qmc.scale(samples, -3, 3),  # Scale to reasonable range
        dtype=torch.float32
    )
    return torch.distributions.Normal(0, 1).icdf(
        torch.sigmoid(normal_samples)
    )
```

### 2. Antithetic Variates

```python
def simulate_with_antithetic(S0, r, sigma, T, n_steps, n_paths):
    """Use antithetic variates for variance reduction."""
    # Generate half the paths
    half_paths = n_paths // 2

    # Standard Brownian motion
    dW = torch.randn(half_paths, n_steps) * torch.sqrt(T / n_steps)

    # Antithetic paths (negated Brownian motion)
    dW_anti = -dW

    # Combine both sets
    all_dW = torch.cat([dW, dW_anti], dim=0)

    # Simulate paths
    return simulate_paths_from_brownian(S0, r, sigma, T, all_dW)
```

### 3. Control Variates

```python
def control_variate_pricing(payoffs, control_payoffs, control_price):
    """Apply control variate technique for variance reduction."""
    # Estimate correlation
    cov = torch.cov(torch.stack([payoffs, control_payoffs]))
    var_control = torch.var(control_payoffs)

    # Optimal coefficient
    c = -cov[0, 1] / var_control

    # Adjusted payoffs
    adjusted_payoffs = payoffs + c * (control_payoffs - control_price)

    return torch.mean(adjusted_payoffs)
```

## Neural Network Optimization

### 1. Architecture Optimization

```python
class OptimizedPricingNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        # Use efficient activation functions
        self.activation = nn.GELU()  # More efficient than ReLU for some cases

        # Batch normalization for faster convergence
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(0.1)  # Regularization
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Initialize weights efficiently
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)
```

### 2. Compilation and JIT

```python
# PyTorch 2.0+ compilation
import torch._dynamo as dynamo

# Compile model for faster execution
compiled_model = torch.compile(
    model,
    mode="reduce-overhead",  # Or "max-autotune" for best performance
    backend="inductor"
)

# JIT compilation for production
scripted_model = torch.jit.script(model)
scripted_model.save("optimized_model.pt")
```

### 3. Pruning and Quantization

```python
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic

# Structured pruning
def prune_model(model, amount=0.3):
    """Prune model weights for faster inference."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

# Dynamic quantization for inference
quantized_model = quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)
```

## Benchmarking

### 1. Performance Benchmarking Suite

```python
import time
import numpy as np
from typing import Dict, List
import json

class PerformanceBenchmark:
    """Comprehensive benchmarking suite for DiffML."""

    def __init__(self):
        self.results = {}

    def benchmark_function(self, func, *args, n_runs=100, warmup=10, **kwargs):
        """Benchmark a function with warmup runs."""
        # Warmup runs
        for _ in range(warmup):
            func(*args, **kwargs)

        # Timed runs
        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            start = time.perf_counter()
            result = func(*args, **kwargs)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()

            times.append(end - start)

        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }

    def benchmark_model_inference(self, model, input_tensor, batch_sizes=[1, 16, 64, 256]):
        """Benchmark model inference at different batch sizes."""
        results = {}

        for batch_size in batch_sizes:
            # Create input batch
            batch_input = input_tensor.repeat(batch_size, 1)

            # Benchmark
            stats = self.benchmark_function(
                model,
                batch_input,
                n_runs=100
            )

            # Calculate throughput
            stats['throughput'] = batch_size / stats['mean']
            results[f'batch_{batch_size}'] = stats

        return results

    def benchmark_monte_carlo(self, n_paths_list=[1000, 10000, 100000]):
        """Benchmark Monte Carlo simulation at different scales."""
        from diffml.simulation import simulate_bs_terminal

        results = {}
        for n_paths in n_paths_list:
            stats = self.benchmark_function(
                simulate_bs_terminal,
                S0=100.0,
                r=0.05,
                sigma=0.2,
                T=1.0,
                n_steps=100,
                n_paths=n_paths,
                n_runs=50
            )
            results[f'paths_{n_paths}'] = stats

        return results

    def save_results(self, filename="benchmark_results.json"):
        """Save benchmark results to file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

    def generate_report(self):
        """Generate a performance report."""
        report = ["# Performance Benchmark Report\n"]

        for test_name, test_results in self.results.items():
            report.append(f"\n## {test_name}\n")

            if isinstance(test_results, dict):
                for metric, value in test_results.items():
                    if isinstance(value, dict):
                        report.append(f"\n### {metric}:")
                        for k, v in value.items():
                            report.append(f"  - {k}: {v:.6f}")
                    else:
                        report.append(f"  - {metric}: {value:.6f}")

        return "\n".join(report)
```

### 2. Memory Benchmarking

```python
def benchmark_memory_usage(func, *args, **kwargs):
    """Benchmark memory usage of a function."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Initial memory
    start_memory = torch.cuda.memory_allocated()

    # Run function
    result = func(*args, **kwargs)

    torch.cuda.synchronize()

    # Peak memory
    peak_memory = torch.cuda.max_memory_allocated()
    end_memory = torch.cuda.memory_allocated()

    return {
        'peak_memory_mb': peak_memory / 1024**2,
        'allocated_memory_mb': (end_memory - start_memory) / 1024**2,
        'result': result
    }
```

## Production Deployment

### 1. Model Optimization for Deployment

```python
def prepare_model_for_production(model, example_input):
    """Prepare model for production deployment."""
    model.eval()

    # Trace the model
    traced_model = torch.jit.trace(model, example_input)

    # Optimize for inference
    traced_model = torch.jit.optimize_for_inference(traced_model)

    # Freeze the model
    traced_model = torch.jit.freeze(traced_model)

    return traced_model
```

### 2. Batching Service

```python
from collections import deque
from threading import Lock
import asyncio

class BatchingService:
    """Efficient batching service for production inference."""

    def __init__(self, model, max_batch_size=32, max_latency_ms=50):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
        self.queue = deque()
        self.lock = Lock()

    async def predict(self, input_tensor):
        """Add request to batch queue."""
        future = asyncio.Future()

        with self.lock:
            self.queue.append((input_tensor, future))

        # Process batch if full or timeout
        if len(self.queue) >= self.max_batch_size:
            await self._process_batch()
        else:
            asyncio.create_task(self._timeout_process())

        return await future

    async def _timeout_process(self):
        """Process batch after timeout."""
        await asyncio.sleep(self.max_latency_ms / 1000.0)
        await self._process_batch()

    async def _process_batch(self):
        """Process accumulated batch."""
        with self.lock:
            if not self.queue:
                return

            batch_items = list(self.queue)
            self.queue.clear()

        # Create batch
        inputs = torch.cat([item[0] for item in batch_items])

        # Run inference
        with torch.no_grad():
            outputs = self.model(inputs)

        # Distribute results
        for i, (_, future) in enumerate(batch_items):
            future.set_result(outputs[i])
```

### 3. Multi-GPU Deployment

```python
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

def setup_multi_gpu(model, distributed=False):
    """Setup model for multi-GPU deployment."""
    if not torch.cuda.is_available():
        return model

    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        return model
    elif num_gpus == 1:
        return model.cuda()
    else:
        if distributed:
            # Distributed Data Parallel (more efficient)
            torch.distributed.init_process_group(backend='nccl')
            model = model.cuda()
            model = DistributedDataParallel(model)
        else:
            # Data Parallel (simpler but less efficient)
            model = nn.DataParallel(model)
            model = model.cuda()

    return model
```

### 4. Monitoring and Profiling in Production

```python
import logging
from prometheus_client import Counter, Histogram, Gauge
import psutil

# Metrics
inference_counter = Counter('model_inference_total', 'Total number of inferences')
inference_duration = Histogram('model_inference_duration_seconds', 'Inference duration')
gpu_memory_usage = Gauge('gpu_memory_usage_mb', 'GPU memory usage in MB')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')

def monitored_inference(model, input_tensor):
    """Inference with monitoring."""
    inference_counter.inc()

    with inference_duration.time():
        output = model(input_tensor)

    # Update resource metrics
    if torch.cuda.is_available():
        gpu_memory_usage.set(torch.cuda.memory_allocated() / 1024**2)

    cpu_usage.set(psutil.cpu_percent())

    return output
```

## Best Practices

1. **Profile First**: Always profile before optimizing. Focus on bottlenecks.

2. **GPU Utilization**: Keep GPU utilization above 90% for efficient training.

3. **Batch Size Tuning**: Find optimal batch size through experimentation:
   ```python
   for batch_size in [32, 64, 128, 256, 512]:
       # Measure throughput and memory usage
       pass
   ```

4. **Memory Management**: Clear cache regularly in long-running experiments:
   ```python
   torch.cuda.empty_cache()
   ```

5. **Reproducibility**: Set seeds for reproducible benchmarks:
   ```python
   torch.manual_seed(42)
   torch.cuda.manual_seed_all(42)
   np.random.seed(42)
   ```

6. **Monitoring**: Use tensorboard or wandb for real-time monitoring:
   ```python
   from torch.utils.tensorboard import SummaryWriter
   writer = SummaryWriter('runs/experiment')
   writer.add_scalar('Loss/train', loss, epoch)
   ```

## Troubleshooting

### Common Performance Issues

1. **Low GPU Utilization**
   - Increase batch size
   - Use data prefetching
   - Optimize data loading pipeline

2. **Out of Memory Errors**
   - Reduce batch size
   - Use gradient accumulation
   - Enable gradient checkpointing
   - Use mixed precision training

3. **Slow Convergence**
   - Tune learning rate
   - Use learning rate scheduling
   - Check data normalization
   - Verify loss function implementation

4. **CPU Bottleneck**
   - Increase num_workers in DataLoader
   - Use GPU for data preprocessing
   - Profile CPU operations

## References

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)