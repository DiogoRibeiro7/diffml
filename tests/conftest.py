"""Pytest configuration and fixtures."""

import sys
import warnings
from collections.abc import Generator
from pathlib import Path

import pytest
import torch

# Add project root and src directory to path so ``diffml`` is importable without installation
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment() -> Generator[None, None, None]:
    """Configure the test environment for all tests."""
    # Suppress warnings during tests
    warnings.filterwarnings("ignore", category=UserWarning)

    # Set PyTorch to deterministic mode for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set number of threads for CPU operations
    torch.set_num_threads(1)

    yield

    # Cleanup if needed
    pass


@pytest.fixture
def device() -> torch.device:
    """Fixture for torch device."""
    return torch.device("cpu")


@pytest.fixture
def batch_size() -> int:
    """Standard batch size for tests."""
    return 32


@pytest.fixture
def seed() -> int:
    """Random seed for reproducibility in tests."""
    return 42


@pytest.fixture
def disable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable CUDA for tests that should run on CPU only."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# Custom pytest options
def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options for pytest."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run GPU tests"
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item]
) -> None:
    """Modify test collection based on command line options."""
    # Skip slow tests unless --run-slow is provided
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip GPU tests unless --run-gpu is provided
    if not config.getoption("--run-gpu"):
        skip_gpu = pytest.mark.skip(reason="Need --run-gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
