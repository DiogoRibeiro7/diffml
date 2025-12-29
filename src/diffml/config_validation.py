"""Configuration validation using Pydantic for TOML experiment configs.

This module provides Pydantic models for validating experiment configurations
loaded from TOML files, ensuring type safety and catching configuration errors
early in the experiment pipeline.
"""

from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
import torch


class DeviceType(str, Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class OptimizerType(str, Enum):
    """Supported optimizer types."""
    ADAM = "Adam"
    ADAMW = "AdamW"
    SGD = "SGD"
    RMSPROP = "RMSprop"
    LBFGS = "LBFGS"


class LossType(str, Enum):
    """Supported loss function types."""
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    DML = "dml"
    ADAPTIVE_DML = "adaptive_dml"


class NetworkArchitecture(str, Enum):
    """Supported network architectures."""
    FEEDFORWARD = "feedforward"
    RESNET = "resnet"
    DIFFERENTIAL = "differential"
    CUSTOM = "custom"


class ExperimentType(str, Enum):
    """Types of experiments."""
    DIGITAL = "digital"
    BARRIER = "barrier"
    BASKET = "basket"
    SMOOTHING = "smoothing"
    GAMMA = "gamma"
    PATH_DEPENDENT = "path_dependent"
    AMERICAN = "american"
    MULTIBARRIER = "multibarrier"


class BSParamsSchema(BaseModel):
    """Black-Scholes parameters with validation."""

    S0: float = Field(gt=0, description="Initial stock price")
    K: float = Field(gt=0, description="Strike price")
    r: float = Field(ge=0, le=0.5, description="Risk-free rate")
    sigma: float = Field(gt=0, le=2.0, description="Volatility")
    T: float = Field(gt=0, le=10.0, description="Time to maturity")

    @validator('sigma')
    def validate_volatility(cls, v):
        """Ensure volatility is reasonable."""
        if v > 1.0:
            print(f"Warning: High volatility {v:.2f} may lead to numerical instability")
        return v

    class Config:
        """Pydantic config."""
        extra = "forbid"  # Disallow extra fields


class NetworkConfigSchema(BaseModel):
    """Neural network configuration with validation."""

    architecture: NetworkArchitecture = NetworkArchitecture.FEEDFORWARD
    input_dim: int = Field(gt=0, le=1000)
    hidden_dims: List[int] = Field(min_items=1, max_items=10)
    output_dim: int = Field(gt=0, le=100)
    activation: Literal["relu", "tanh", "sigmoid", "gelu", "selu"] = "relu"
    dropout_rate: float = Field(ge=0, le=0.9, default=0.0)
    batch_norm: bool = False
    weight_init: Literal["xavier", "kaiming", "normal", "uniform"] = "xavier"

    @validator('hidden_dims')
    def validate_hidden_dims(cls, v):
        """Ensure hidden dimensions are reasonable."""
        for dim in v:
            if dim <= 0:
                raise ValueError(f"Hidden dimension must be positive, got {dim}")
            if dim > 10000:
                raise ValueError(f"Hidden dimension {dim} is too large")
        return v

    @validator('dropout_rate')
    def validate_dropout(cls, v):
        """Warn about high dropout rates."""
        if v > 0.5:
            print(f"Warning: High dropout rate {v:.2f} may hurt performance")
        return v

    class Config:
        """Pydantic config."""
        extra = "forbid"
        use_enum_values = True


class TrainingConfigSchema(BaseModel):
    """Training configuration with validation."""

    # Basic training parameters
    n_epochs: int = Field(gt=0, le=10000)
    batch_size: int = Field(gt=0, le=10000)
    learning_rate: float = Field(gt=0, le=1.0)
    optimizer: OptimizerType = OptimizerType.ADAM
    loss_function: LossType = LossType.DML

    # DML specific
    lambda_val: float = Field(ge=0, le=100, default=1.0)
    use_antithetic: bool = False
    use_sobol: bool = False

    # Learning rate scheduling
    lr_scheduler: Optional[Literal["step", "exponential", "cosine", "plateau"]] = None
    lr_decay_factor: float = Field(gt=0, le=1.0, default=0.1)
    lr_decay_steps: int = Field(gt=0, default=100)

    # Early stopping
    early_stopping: bool = False
    patience: int = Field(gt=0, default=10)
    min_delta: float = Field(ge=0, default=1e-4)

    # Gradient clipping
    gradient_clip_norm: Optional[float] = Field(ge=0, default=None)
    gradient_clip_value: Optional[float] = Field(ge=0, default=None)

    # Device configuration
    device: DeviceType = DeviceType.AUTO
    mixed_precision: bool = False
    compile_model: bool = False  # PyTorch 2.0+ compilation

    # Checkpointing
    checkpoint_dir: Optional[str] = None
    checkpoint_frequency: int = Field(gt=0, default=10)
    keep_best_only: bool = True

    # Logging
    log_frequency: int = Field(gt=0, default=10)
    tensorboard: bool = False
    wandb_project: Optional[str] = None

    @validator('learning_rate')
    def validate_learning_rate(cls, v, values):
        """Validate learning rate based on optimizer."""
        optimizer = values.get('optimizer', OptimizerType.ADAM)
        if optimizer == OptimizerType.SGD and v > 0.1:
            print(f"Warning: High learning rate {v} for SGD may cause instability")
        return v

    @validator('batch_size')
    def validate_batch_size(cls, v):
        """Ensure batch size is power of 2 for GPU efficiency."""
        if v & (v - 1) != 0:  # Not a power of 2
            # Find nearest power of 2
            import math
            nearest = 2 ** round(math.log2(v))
            print(f"Info: Batch size {v} is not a power of 2. Consider {nearest} for better GPU efficiency")
        return v

    @root_validator
    def validate_gradient_clipping(cls, values):
        """Ensure only one type of gradient clipping is used."""
        if values.get('gradient_clip_norm') and values.get('gradient_clip_value'):
            raise ValueError("Cannot use both norm and value gradient clipping")
        return values

    class Config:
        """Pydantic config."""
        extra = "forbid"
        use_enum_values = True


class MonteCarloConfigSchema(BaseModel):
    """Monte Carlo simulation configuration."""

    n_paths: int = Field(gt=0, le=1000000)
    n_steps: int = Field(gt=0, le=10000)
    seed: Optional[int] = Field(ge=0, default=None)
    use_quasi_random: bool = False
    quasi_random_type: Literal["sobol", "halton", "latin_hypercube"] = "sobol"
    antithetic_variates: bool = False
    moment_matching: bool = False
    importance_sampling: bool = False

    @validator('n_paths')
    def validate_paths(cls, v, values):
        """Validate number of paths."""
        if v > 100000:
            print(f"Warning: Large number of paths ({v}) may require significant memory")
        if values.get('use_quasi_random') and v & (v - 1) != 0:
            import math
            nearest = 2 ** math.ceil(math.log2(v))
            print(f"Info: For quasi-random sequences, consider {nearest} paths (power of 2)")
        return v

    @root_validator
    def validate_variance_reduction(cls, values):
        """Validate variance reduction techniques."""
        techniques = sum([
            values.get('antithetic_variates', False),
            values.get('moment_matching', False),
            values.get('importance_sampling', False)
        ])
        if techniques > 2:
            print("Warning: Using multiple variance reduction techniques may not improve results")
        return values

    class Config:
        """Pydantic config."""
        extra = "forbid"


class DatasetConfigSchema(BaseModel):
    """Dataset generation configuration."""

    train_size: int = Field(gt=0)
    val_size: int = Field(gt=0)
    test_size: int = Field(gt=0)
    normalize_features: bool = True
    normalize_targets: bool = False
    augmentation_factor: int = Field(ge=1, default=1)
    noise_level: float = Field(ge=0, le=0.1, default=0.0)
    cache_dataset: bool = True
    num_workers: int = Field(ge=0, default=4)
    pin_memory: bool = True

    @validator('train_size')
    def validate_train_size(cls, v):
        """Ensure training set is large enough."""
        if v < 100:
            print(f"Warning: Small training size ({v}) may lead to overfitting")
        return v

    @root_validator
    def validate_dataset_sizes(cls, values):
        """Validate relative dataset sizes."""
        train = values.get('train_size', 0)
        val = values.get('val_size', 0)
        test = values.get('test_size', 0)

        if val > train:
            print("Warning: Validation set larger than training set")
        if test > train:
            print("Warning: Test set larger than training set")

        total = train + val + test
        if total > 1000000:
            print(f"Warning: Large total dataset size ({total}) may require significant memory")

        return values

    class Config:
        """Pydantic config."""
        extra = "forbid"


class ExperimentConfigSchema(BaseModel):
    """Complete experiment configuration."""

    name: str = Field(min_length=1, max_length=100)
    description: Optional[str] = None
    experiment_type: ExperimentType
    version: str = Field(regex=r"^\d+\.\d+\.\d+$", default="1.0.0")

    # Sub-configurations
    bs_params: BSParamsSchema
    network_config: NetworkConfigSchema
    training_config: TrainingConfigSchema
    monte_carlo_config: MonteCarloConfigSchema
    dataset_config: DatasetConfigSchema

    # Experiment specific parameters
    custom_params: Optional[Dict[str, Any]] = None

    # Reproducibility
    seed: int = Field(ge=0, default=42)
    deterministic: bool = True

    # Output configuration
    output_dir: str = Field(default="./results")
    save_model: bool = True
    save_predictions: bool = True
    save_plots: bool = True

    @validator('name')
    def validate_name(cls, v):
        """Ensure experiment name is valid."""
        import re
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Experiment name must contain only alphanumeric characters, hyphens, and underscores")
        return v

    @root_validator
    def validate_experiment_consistency(cls, values):
        """Validate consistency across configuration."""
        exp_type = values.get('experiment_type')
        network = values.get('network_config')

        # Validate input dimensions match experiment type
        if exp_type and network:
            expected_dims = {
                ExperimentType.DIGITAL: 5,  # S0, K, r, sigma, T
                ExperimentType.BARRIER: 7,  # S0, K, r, sigma, T, barrier, rebate
                ExperimentType.BASKET: 20,  # Multiple assets
                ExperimentType.AMERICAN: 5,
                ExperimentType.MULTIBARRIER: 7,
            }

            if exp_type in expected_dims:
                expected = expected_dims[exp_type]
                actual = network.input_dim
                if actual != expected and values.get('custom_params', {}).get('ignore_dim_check') != True:
                    print(f"Warning: Input dimension {actual} may not match {exp_type.value} experiment (expected {expected})")

        return values

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility with existing code."""
        return self.dict()

    class Config:
        """Pydantic config."""
        extra = "forbid"
        use_enum_values = True
        validate_assignment = True


class MultiExperimentConfigSchema(BaseModel):
    """Configuration for running multiple experiments."""

    experiments: List[ExperimentConfigSchema]
    parallel_execution: bool = False
    max_parallel: int = Field(ge=1, le=10, default=2)
    dependency_graph: Optional[Dict[str, List[str]]] = None
    global_seed: int = Field(ge=0, default=42)
    summary_report: bool = True

    @validator('experiments')
    def validate_unique_names(cls, v):
        """Ensure all experiment names are unique."""
        names = [exp.name for exp in v]
        if len(names) != len(set(names)):
            raise ValueError("All experiment names must be unique")
        return v

    @root_validator
    def validate_dependencies(cls, values):
        """Validate dependency graph."""
        experiments = values.get('experiments', [])
        dep_graph = values.get('dependency_graph', {})

        if dep_graph:
            exp_names = {exp.name for exp in experiments}
            for exp, deps in dep_graph.items():
                if exp not in exp_names:
                    raise ValueError(f"Unknown experiment in dependency graph: {exp}")
                for dep in deps:
                    if dep not in exp_names:
                        raise ValueError(f"Unknown dependency {dep} for experiment {exp}")

                # Check for circular dependencies
                visited = set()
                stack = set()

                def has_cycle(node):
                    if node in stack:
                        return True
                    if node in visited:
                        return False

                    visited.add(node)
                    stack.add(node)

                    for neighbor in dep_graph.get(node, []):
                        if has_cycle(neighbor):
                            return True

                    stack.remove(node)
                    return False

                if has_cycle(exp):
                    raise ValueError(f"Circular dependency detected involving {exp}")

        return values

    class Config:
        """Pydantic config."""
        extra = "forbid"


def validate_toml_config(config_dict: Dict[str, Any], schema_class=ExperimentConfigSchema) -> BaseModel:
    """Validate a TOML configuration dictionary against a schema.

    Parameters:
        config_dict: Dictionary loaded from TOML file
        schema_class: Pydantic model class to validate against

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If configuration is invalid
    """
    return schema_class(**config_dict)


def load_and_validate_config(config_path: str) -> ExperimentConfigSchema:
    """Load and validate a TOML configuration file.

    Parameters:
        config_path: Path to TOML configuration file

    Returns:
        Validated experiment configuration

    Raises:
        ValidationError: If configuration is invalid
        FileNotFoundError: If config file doesn't exist
    """
    import toml

    with open(config_path, 'r') as f:
        config_dict = toml.load(f)

    return validate_toml_config(config_dict)


def generate_config_template(
    experiment_type: ExperimentType,
    output_path: str = "config_template.toml"
) -> None:
    """Generate a template configuration file for a given experiment type.

    Parameters:
        experiment_type: Type of experiment
        output_path: Where to save the template
    """
    import toml

    # Create default configuration
    config = ExperimentConfigSchema(
        name=f"{experiment_type.value}_experiment",
        description=f"Template configuration for {experiment_type.value} experiment",
        experiment_type=experiment_type,
        bs_params=BSParamsSchema(
            S0=100.0,
            K=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0
        ),
        network_config=NetworkConfigSchema(
            architecture=NetworkArchitecture.FEEDFORWARD,
            input_dim=5,
            hidden_dims=[64, 32, 16],
            output_dim=1,
            activation="relu",
            dropout_rate=0.1
        ),
        training_config=TrainingConfigSchema(
            n_epochs=100,
            batch_size=256,
            learning_rate=0.001,
            optimizer=OptimizerType.ADAM,
            loss_function=LossType.DML,
            lambda_val=1.0,
            early_stopping=True,
            patience=10
        ),
        monte_carlo_config=MonteCarloConfigSchema(
            n_paths=10000,
            n_steps=100,
            seed=42,
            antithetic_variates=True
        ),
        dataset_config=DatasetConfigSchema(
            train_size=10000,
            val_size=2000,
            test_size=2000,
            normalize_features=True
        )
    )

    # Convert to dict and save as TOML
    config_dict = config.dict()
    with open(output_path, 'w') as f:
        toml.dump(config_dict, f)

    print(f"Template configuration saved to {output_path}")


def validate_config_file(config_path: str) -> bool:
    """Validate a configuration file and report any errors.

    Parameters:
        config_path: Path to configuration file

    Returns:
        True if valid, False otherwise
    """
    try:
        config = load_and_validate_config(config_path)
        print(f"✓ Configuration '{config.name}' is valid")
        print(f"  Experiment type: {config.experiment_type}")
        print(f"  Network: {config.network_config.architecture}")
        print(f"  Training: {config.training_config.n_epochs} epochs")
        print(f"  Monte Carlo: {config.monte_carlo_config.n_paths} paths")
        return True

    except Exception as e:
        print(f"✗ Configuration validation failed:")
        print(f"  {str(e)}")
        return False


# CLI interface for validation
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="TOML Configuration Validator")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a config file")
    validate_parser.add_argument("config", help="Path to TOML config file")

    # Generate template command
    template_parser = subparsers.add_parser("template", help="Generate a template config")
    template_parser.add_argument(
        "experiment_type",
        choices=[e.value for e in ExperimentType],
        help="Type of experiment"
    )
    template_parser.add_argument(
        "-o", "--output",
        default="config_template.toml",
        help="Output path for template"
    )

    args = parser.parse_args()

    if args.command == "validate":
        success = validate_config_file(args.config)
        sys.exit(0 if success else 1)

    elif args.command == "template":
        exp_type = ExperimentType(args.experiment_type)
        generate_config_template(exp_type, args.output)

    else:
        parser.print_help()