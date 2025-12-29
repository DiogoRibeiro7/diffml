"""GPU batch processing optimization strategies for DiffML.

This module provides optimized GPU implementations for batch processing,
mixed precision training, and efficient memory management for large-scale
Monte Carlo simulations and neural network training.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Tuple, Optional, List, Callable, Dict, Any
from dataclasses import dataclass
import math
import warnings
from contextlib import contextmanager

from .config import get_device


@dataclass
class GPUConfig:
    """Configuration for GPU optimization settings.

    Attributes:
        enable_mixed_precision: Use automatic mixed precision (AMP)
        enable_cudnn_benchmark: Enable cuDNN autotuner for convolutions
        enable_tf32: Enable TensorFloat-32 for Ampere GPUs
        memory_fraction: Fraction of GPU memory to allocate
        enable_gradient_checkpointing: Trade compute for memory
        batch_size_finder: Automatically find optimal batch size
        pin_memory: Pin memory for faster CPU-GPU transfer
        num_workers: Number of data loading workers
        prefetch_factor: Number of batches to prefetch per worker
    """

    enable_mixed_precision: bool = True
    enable_cudnn_benchmark: bool = True
    enable_tf32: bool = True
    memory_fraction: float = 0.9
    enable_gradient_checkpointing: bool = False
    batch_size_finder: bool = False
    pin_memory: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2


class GPUOptimizer:
    """Main class for GPU optimization strategies."""

    def __init__(self, config: Optional[GPUConfig] = None):
        """Initialize GPU optimizer.

        Parameters:
            config: GPU configuration settings
        """
        self.config = config or GPUConfig()
        self.device = get_device()
        self.scaler = GradScaler() if self.config.enable_mixed_precision else None

        # Configure GPU settings
        self._configure_gpu()

    def _configure_gpu(self):
        """Configure global GPU settings."""
        if not torch.cuda.is_available():
            return

        # Enable cuDNN benchmark mode
        torch.backends.cudnn.benchmark = self.config.enable_cudnn_benchmark

        # Enable TF32 on Ampere GPUs
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = self.config.enable_tf32
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = self.config.enable_tf32

        # Set memory fraction
        if self.config.memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)

    @contextmanager
    def mixed_precision_context(self):
        """Context manager for mixed precision operations.

        Yields:
            Context for mixed precision computation
        """
        if self.config.enable_mixed_precision and self.device.type == 'cuda':
            with autocast():
                yield
        else:
            yield

    def optimize_batch_size(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        min_batch: int = 1,
        max_batch: int = 1024,
        memory_margin: float = 0.9
    ) -> int:
        """Find optimal batch size for given model and input.

        Uses binary search to find the largest batch size that fits in memory.

        Parameters:
            model: Neural network model
            input_shape: Shape of single input sample
            min_batch: Minimum batch size to try
            max_batch: Maximum batch size to try
            memory_margin: Safety margin for memory usage

        Returns:
            Optimal batch size
        """
        if self.device.type != 'cuda':
            return min(32, max_batch)  # Default for CPU

        model = model.to(self.device)
        model.train()

        def can_fit_batch(batch_size: int) -> bool:
            """Test if batch size fits in memory."""
            try:
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Create dummy batch
                dummy_input = torch.randn(batch_size, *input_shape, device=self.device)

                # Forward pass
                with self.mixed_precision_context():
                    output = model(dummy_input)
                    if hasattr(output, 'backward'):
                        # Simulate backward pass
                        loss = output.mean()
                        loss.backward()

                # Check memory usage
                memory_used = torch.cuda.max_memory_allocated()
                memory_available = torch.cuda.get_device_properties(0).total_memory
                return memory_used < memory_available * memory_margin

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    return False
                raise

        # Binary search for optimal batch size
        left, right = min_batch, max_batch
        optimal = min_batch

        while left <= right:
            mid = (left + right) // 2
            # Round to nearest power of 2 for efficiency
            mid = 2 ** round(math.log2(mid))

            if can_fit_batch(mid):
                optimal = mid
                left = mid + 1
            else:
                right = mid - 1

        print(f"Optimal batch size found: {optimal}")
        return optimal


class BatchedMonteCarloSimulator:
    """Optimized batched Monte Carlo simulation for GPU.

    This class provides memory-efficient Monte Carlo simulation
    with automatic batching and GPU optimization.
    """

    def __init__(self, batch_size: int = 10000, device: Optional[torch.device] = None):
        """Initialize batched simulator.

        Parameters:
            batch_size: Number of paths to simulate per batch
            device: Device to run simulations on
        """
        self.batch_size = batch_size
        self.device = device or get_device()

    @torch.jit.script
    def _simulate_gbm_batch(
        S0: torch.Tensor,
        drift: torch.Tensor,
        diffusion: torch.Tensor,
        dt: torch.Tensor,
        random_normals: torch.Tensor
    ) -> torch.Tensor:
        """JIT-compiled GBM simulation for performance.

        Parameters:
            S0: Initial price
            drift: Drift term (r - 0.5 * sigma^2) * dt
            diffusion: Diffusion term sigma * sqrt(dt)
            dt: Time step
            random_normals: Random normal samples

        Returns:
            Simulated paths
        """
        n_paths, n_steps = random_normals.shape

        # Pre-allocate paths
        paths = torch.zeros(n_paths, n_steps + 1, device=S0.device)
        paths[:, 0] = S0

        # Vectorized simulation
        log_returns = drift + diffusion * random_normals
        cum_returns = torch.cumsum(log_returns, dim=1)
        paths[:, 1:] = S0 * torch.exp(cum_returns)

        return paths

    def simulate_paths_batched(
        self,
        S0: float,
        r: float,
        sigma: float,
        T: float,
        n_steps: int,
        n_paths: int,
        antithetic: bool = False,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """Simulate paths in memory-efficient batches.

        Parameters:
            S0: Initial stock price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            n_steps: Number of time steps
            n_paths: Total number of paths
            antithetic: Use antithetic variates
            seed: Random seed

        Returns:
            Simulated paths
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Calculate drift and diffusion
        dt = T / n_steps
        drift = (r - 0.5 * sigma ** 2) * dt
        diffusion = sigma * math.sqrt(dt)

        # Convert to tensors
        S0_t = torch.tensor(S0, device=self.device, dtype=torch.float32)
        drift_t = torch.tensor(drift, device=self.device, dtype=torch.float32)
        diffusion_t = torch.tensor(diffusion, device=self.device, dtype=torch.float32)
        dt_t = torch.tensor(dt, device=self.device, dtype=torch.float32)

        all_paths = []

        # Process in batches
        for i in range(0, n_paths, self.batch_size):
            batch_paths = min(self.batch_size, n_paths - i)

            if antithetic and batch_paths > 1:
                # Generate half paths with antithetic
                half_batch = batch_paths // 2
                randn = torch.randn(half_batch, n_steps, device=self.device)
                randn_anti = -randn
                random_normals = torch.cat([randn, randn_anti], dim=0)
                if batch_paths % 2 == 1:
                    # Add one more path if odd number
                    extra = torch.randn(1, n_steps, device=self.device)
                    random_normals = torch.cat([random_normals, extra], dim=0)
            else:
                random_normals = torch.randn(batch_paths, n_steps, device=self.device)

            # Simulate batch
            batch = self._simulate_gbm_batch(
                S0_t, drift_t, diffusion_t, dt_t, random_normals
            )

            # Move to CPU to save GPU memory
            all_paths.append(batch.cpu())

            # Clear GPU cache
            if i % (self.batch_size * 10) == 0:
                torch.cuda.empty_cache()

        # Concatenate all batches
        return torch.cat(all_paths, dim=0).to(self.device)


class OptimizedDataLoader:
    """Optimized data loader for GPU training."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        gpu_config: Optional[GPUConfig] = None
    ):
        """Initialize optimized data loader.

        Parameters:
            dataset: PyTorch dataset
            batch_size: Batch size
            gpu_config: GPU configuration
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.config = gpu_config or GPUConfig()

    def get_loader(self, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """Create optimized DataLoader.

        Parameters:
            shuffle: Whether to shuffle data

        Returns:
            Optimized DataLoader
        """
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.num_workers > 0
        )


class GradientAccumulator:
    """Helper for gradient accumulation to simulate larger batches."""

    def __init__(self, accumulation_steps: int = 4):
        """Initialize gradient accumulator.

        Parameters:
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def should_step(self) -> bool:
        """Check if optimizer should step.

        Returns:
            True if should step optimizer
        """
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation.

        Parameters:
            loss: Original loss

        Returns:
            Scaled loss
        """
        return loss / self.accumulation_steps


def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """Optimize model for inference performance.

    Applies various optimizations including:
    - JIT compilation
    - Graph optimization
    - Kernel fusion

    Parameters:
        model: Model to optimize

    Returns:
        Optimized model
    """
    model.eval()
    device = next(model.parameters()).device

    # Try TorchScript compilation
    try:
        # Create example input
        example_input = torch.randn(1, model.input_dim, device=device)

        # Trace model
        traced_model = torch.jit.trace(model, example_input)

        # Optimize graph
        traced_model = torch.jit.optimize_for_inference(traced_model)

        # Freeze model
        traced_model = torch.jit.freeze(traced_model)

        return traced_model
    except Exception as e:
        warnings.warn(f"JIT compilation failed: {e}. Returning original model.")
        return model


class MultiGPUTrainer:
    """Trainer for multi-GPU distributed training."""

    def __init__(self, model: nn.Module, world_size: Optional[int] = None):
        """Initialize multi-GPU trainer.

        Parameters:
            model: Model to train
            world_size: Number of GPUs to use
        """
        self.model = model
        self.world_size = world_size or torch.cuda.device_count()

        if self.world_size > 1:
            self._setup_distributed()

    def _setup_distributed(self):
        """Setup distributed training."""
        if torch.cuda.device_count() < 2:
            warnings.warn("Less than 2 GPUs available, using single GPU")
            self.model = self.model.cuda()
            return

        # Use DataParallel for simplicity (DistributedDataParallel is better for production)
        self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler] = None
    ) -> float:
        """Execute single training step.

        Parameters:
            inputs: Input batch
            targets: Target batch
            loss_fn: Loss function
            optimizer: Optimizer
            scaler: GradScaler for mixed precision

        Returns:
            Loss value
        """
        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        return loss.item()


def benchmark_gpu_operations(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    n_iterations: int = 100
) -> Dict[str, float]:
    """Benchmark GPU operations for profiling.

    Parameters:
        model: Model to benchmark
        input_shape: Input shape for model
        n_iterations: Number of iterations to run

    Returns:
        Dictionary with benchmark results
    """
    device = get_device()
    model = model.to(device)

    # Warmup
    for _ in range(10):
        dummy_input = torch.randn(32, *input_shape, device=device)
        _ = model(dummy_input)

    # Benchmark forward pass
    torch.cuda.synchronize()
    forward_times = []

    for _ in range(n_iterations):
        dummy_input = torch.randn(32, *input_shape, device=device)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = model(dummy_input)
        end.record()

        torch.cuda.synchronize()
        forward_times.append(start.elapsed_time(end))

    # Benchmark forward + backward
    model.train()
    backward_times = []

    for _ in range(n_iterations):
        dummy_input = torch.randn(32, *input_shape, device=device)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = model(dummy_input)
        loss = output.mean()
        loss.backward()
        end.record()

        torch.cuda.synchronize()
        backward_times.append(start.elapsed_time(end))

    results = {
        'forward_mean_ms': sum(forward_times) / len(forward_times),
        'forward_std_ms': torch.tensor(forward_times).std().item(),
        'forward_backward_mean_ms': sum(backward_times) / len(backward_times),
        'forward_backward_std_ms': torch.tensor(backward_times).std().item(),
        'throughput_samples_per_sec': 32 * 1000 / (sum(forward_times) / len(forward_times))
    }

    return results


# Example usage and testing
if __name__ == "__main__":
    print("Testing GPU Optimization Strategies")
    print("=" * 50)

    # Test GPU configuration
    config = GPUConfig(
        enable_mixed_precision=True,
        batch_size_finder=True
    )

    optimizer = GPUOptimizer(config)
    print(f"Device: {optimizer.device}")
    print(f"Mixed Precision: {config.enable_mixed_precision}")

    # Test batched Monte Carlo
    print("\nTesting Batched Monte Carlo Simulation...")
    simulator = BatchedMonteCarloSimulator(batch_size=5000)
    paths = simulator.simulate_paths_batched(
        S0=100.0,
        r=0.05,
        sigma=0.2,
        T=1.0,
        n_steps=252,
        n_paths=10000,
        antithetic=True
    )
    print(f"Simulated paths shape: {paths.shape}")
    print(f"Mean final price: {paths[:, -1].mean():.2f}")

    # Test model optimization
    print("\nTesting Model Optimization...")
    from .networks import FeedForwardNet

    model = FeedForwardNet(
        input_dim=5,
        hidden_dims=[64, 32, 16],
        output_dim=1
    )

    # Find optimal batch size
    if torch.cuda.is_available():
        optimal_batch = optimizer.optimize_batch_size(
            model,
            input_shape=(5,),
            min_batch=16,
            max_batch=512
        )
        print(f"Optimal batch size: {optimal_batch}")

        # Benchmark model
        print("\nBenchmarking GPU Operations...")
        results = benchmark_gpu_operations(model, (5,), n_iterations=50)
        for key, value in results.items():
            print(f"  {key}: {value:.2f}")

    print("\n✓ GPU optimization tests completed!")