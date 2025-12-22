"""Configuration structures for diffml article replication experiments."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if False:  # pragma: no cover - typing imports
    import tomllib

import sys

if sys.version_info >= (3, 11):
    import tomllib  # type: ignore
else:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Python < 3.11 requires the 'tomli' package") from exc

from diffml.config import TrainingConfig


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration values required to reproduce a paper experiment."""

    name: str
    seed: int
    training: TrainingConfig
    m_train: int
    m_test: int
    n_paths_train: int
    n_paths_test: int
    K: float | None = None
    B: float | None = None
    H: float | None = None
    L: float | None = None
    d: int | None = None
    n_steps: int | None = None
    eps_multipliers: list[float] | None = None
    x_min: float = 0.5
    x_max: float = 1.5
    r: float = 0.05
    sigma: float = 0.2
    T: float = 0.25
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate key numeric parameters."""
        if self.m_train <= 0:
            raise ValueError("m_train must be positive")
        if self.m_test <= 0:
            raise ValueError("m_test must be positive")
        if self.n_paths_train <= 0:
            raise ValueError("n_paths_train must be positive")
        if self.n_paths_test <= 0:
            raise ValueError("n_paths_test must be positive")
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be strictly less than x_max")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.T <= 0:
            raise ValueError("T must be positive")


def load_experiment_config(path: str) -> ExperimentConfig:
    """Parse a TOML configuration file into :class:`ExperimentConfig`."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with config_path.open("rb") as fh:
        data = tomllib.load(fh)

    experiment_section = data.get("experiment", {})
    training_section = data.get("training", {})
    dataset_section = data.get("dataset", {})
    payoff_section = data.get("payoff", {})
    bs_section = data.get("black_scholes", {})

    for field_name in ("name", "seed"):
        if field_name not in experiment_section:
            raise ValueError(f"Missing required field experiment.{field_name}")

    default_training = TrainingConfig()
    training = TrainingConfig(
        n_epochs=training_section.get("n_epochs", default_training.n_epochs),
        batch_size=training_section.get("batch_size", default_training.batch_size),
        lr_initial=training_section.get("lr_initial", default_training.lr_initial),
        lr_min=training_section.get("lr_min", default_training.lr_min),
        lambda_delta=training_section.get("lambda_delta", default_training.lambda_delta),
        lambda_gamma=training_section.get("lambda_gamma", default_training.lambda_gamma),
        use_mixed_precision=training_section.get(
            "use_mixed_precision", default_training.use_mixed_precision
        ),
    )

    return ExperimentConfig(
        name=experiment_section["name"],
        seed=int(experiment_section["seed"]),
        training=training,
        m_train=int(dataset_section.get("m_train", 1024)),
        m_test=int(dataset_section.get("m_test", 256)),
        n_paths_train=int(dataset_section.get("n_paths_train", 10000)),
        n_paths_test=int(dataset_section.get("n_paths_test", 50000)),
        x_min=float(dataset_section.get("x_min", 0.5)),
        x_max=float(dataset_section.get("x_max", 1.5)),
        K=payoff_section.get("K"),
        B=payoff_section.get("B"),
        H=payoff_section.get("H"),
        L=payoff_section.get("L"),
        d=payoff_section.get("d"),
        n_steps=payoff_section.get("n_steps"),
        eps_multipliers=payoff_section.get("eps_multipliers"),
        r=float(bs_section.get("r", 0.05)),
        sigma=float(bs_section.get("sigma", 0.2)),
        T=float(bs_section.get("T", 0.25)),
        extra_params={
            key: value
            for key, value in data.items()
            if key not in {"experiment", "training", "dataset", "payoff", "black_scholes"}
        },
    )


def save_experiment_config(config: ExperimentConfig, path: str) -> None:
    """Persist an :class:`ExperimentConfig` to disk as TOML."""
    toml_module = importlib.import_module("toml")

    payload: dict[str, Any] = {
        "experiment": {"name": config.name, "seed": config.seed},
        "training": {
            "n_epochs": config.training.n_epochs,
            "batch_size": config.training.batch_size,
            "lr_initial": config.training.lr_initial,
            "lr_min": config.training.lr_min,
            "lambda_delta": config.training.lambda_delta,
            "lambda_gamma": config.training.lambda_gamma,
            "use_mixed_precision": config.training.use_mixed_precision,
        },
        "dataset": {
            "m_train": config.m_train,
            "m_test": config.m_test,
            "n_paths_train": config.n_paths_train,
            "n_paths_test": config.n_paths_test,
            "x_min": config.x_min,
            "x_max": config.x_max,
        },
        "black_scholes": {"r": config.r, "sigma": config.sigma, "T": config.T},
    }

    payoff_section: dict[str, Any] = {}
    for attr in ("K", "B", "H", "L", "d", "n_steps", "eps_multipliers"):
        value = getattr(config, attr)
        if value is not None:
            payoff_section[attr] = value

    if payoff_section:
        payload["payoff"] = payoff_section

    payload.update(config.extra_params)
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fh:
        toml_module.dump(payload, fh)


__all__ = ["ExperimentConfig", "load_experiment_config", "save_experiment_config"]
