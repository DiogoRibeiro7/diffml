# Makefile for DiffML project
# Simplifies common development and deployment tasks

.PHONY: help install test lint format clean docker docs benchmark release

# Variables
PYTHON := python3
POETRY := poetry
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := diffml
PYTHON_VERSION := 3.11

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Default target
help: ## Show this help message
	@echo "$(GREEN)DiffML Development Makefile$(NC)"
	@echo "=============================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# Installation targets
install: ## Install dependencies with Poetry
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(POETRY) install --with dev

install-prod: ## Install production dependencies only
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(POETRY) install --without dev

install-pre-commit: ## Install pre-commit hooks
	@echo "$(GREEN)Installing pre-commit hooks...$(NC)"
	pre-commit install
	pre-commit run --all-files

update-deps: ## Update all dependencies
	@echo "$(GREEN)Updating dependencies...$(NC)"
	$(POETRY) update

# Testing targets
test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	$(POETRY) run pytest tests/ -v

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(POETRY) run pytest tests/ -v -k "not integration and not slow"

test-integration: ## Run integration tests
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(POETRY) run pytest tests/test_integration_workflows.py -v

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(POETRY) run pytest tests/ --cov=diffml --cov-report=html --cov-report=term

test-all: ## Run comprehensive test suite
	@echo "$(GREEN)Running comprehensive test suite...$(NC)"
	$(POETRY) run python scripts/run_all_tests.py

# Code quality targets
lint: ## Run linting with ruff and mypy
	@echo "$(GREEN)Running linters...$(NC)"
	$(POETRY) run ruff check src/
	$(POETRY) run mypy src/ --ignore-missing-imports

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	$(POETRY) run black src/ tests/ scripts/
	$(POETRY) run isort src/ tests/ scripts/

check: lint test ## Run linting and tests

security: ## Run security checks with bandit
	@echo "$(GREEN)Running security checks...$(NC)"
	$(POETRY) run bandit -r src/ -ll

# Benchmarking targets
benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running benchmarks...$(NC)"
	$(POETRY) run python scripts/create_performance_baseline.py --create

benchmark-quick: ## Run quick benchmarks
	@echo "$(GREEN)Running quick benchmarks...$(NC)"
	$(POETRY) run python scripts/create_performance_baseline.py --quick

benchmark-compare: ## Compare with baseline benchmarks
	@echo "$(GREEN)Comparing with baseline...$(NC)"
	$(POETRY) run python scripts/create_performance_baseline.py --compare

# Experiment targets
experiments: ## Run all experiments
	@echo "$(GREEN)Running comprehensive experiments...$(NC)"
	$(POETRY) run python scripts/run_comprehensive_experiments.py

experiments-quick: ## Run quick experiments
	@echo "$(GREEN)Running quick experiments...$(NC)"
	$(POETRY) run python scripts/run_comprehensive_experiments.py --quick

# Docker targets
docker-build: ## Build all Docker images
	@echo "$(GREEN)Building Docker images...$(NC)"
	$(DOCKER) build -t $(PROJECT_NAME):cpu-latest --target cpu-runtime .
	$(DOCKER) build -t $(PROJECT_NAME):gpu-latest --target gpu-runtime .
	$(DOCKER) build -t $(PROJECT_NAME):dev-latest --target development .

docker-build-cpu: ## Build CPU Docker image
	@echo "$(GREEN)Building CPU Docker image...$(NC)"
	$(DOCKER) build -t $(PROJECT_NAME):cpu-latest --target cpu-runtime .

docker-build-gpu: ## Build GPU Docker image
	@echo "$(GREEN)Building GPU Docker image...$(NC)"
	$(DOCKER) build -t $(PROJECT_NAME):gpu-latest --target gpu-runtime .

docker-run-cpu: ## Run CPU container
	@echo "$(GREEN)Running CPU container...$(NC)"
	$(DOCKER_COMPOSE) up diffml-cpu

docker-run-gpu: ## Run GPU container
	@echo "$(GREEN)Running GPU container...$(NC)"
	$(DOCKER_COMPOSE) --profile gpu up diffml-gpu

docker-jupyter: ## Start Jupyter Lab in Docker
	@echo "$(GREEN)Starting Jupyter Lab...$(NC)"
	$(DOCKER_COMPOSE) --profile dev up jupyter
	@echo "$(YELLOW)Jupyter Lab available at http://localhost:8888$(NC)"

docker-streamlit: ## Start Streamlit dashboard in Docker
	@echo "$(GREEN)Starting Streamlit dashboard...$(NC)"
	$(DOCKER_COMPOSE) --profile dev up streamlit
	@echo "$(YELLOW)Streamlit available at http://localhost:8501$(NC)"

docker-stop: ## Stop all Docker containers
	@echo "$(GREEN)Stopping Docker containers...$(NC)"
	$(DOCKER_COMPOSE) down

docker-clean: ## Clean Docker images and volumes
	@echo "$(RED)Cleaning Docker resources...$(NC)"
	$(DOCKER_COMPOSE) down -v
	$(DOCKER) system prune -f

# Documentation targets
docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	cd docs && make clean && make html
	@echo "$(YELLOW)Documentation available at docs/_build/html/index.html$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation...$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server

# Development targets
dev-setup: install install-pre-commit ## Complete development setup
	@echo "$(GREEN)Development environment ready!$(NC)"

notebook: ## Start Jupyter notebook
	@echo "$(GREEN)Starting Jupyter notebook...$(NC)"
	$(POETRY) run jupyter lab

dashboard: ## Start Streamlit dashboard
	@echo "$(GREEN)Starting Streamlit dashboard...$(NC)"
	$(POETRY) run streamlit run dashboard.py

# Release targets
version: ## Show current version
	@$(POETRY) version

version-patch: ## Bump patch version
	@echo "$(GREEN)Bumping patch version...$(NC)"
	$(POETRY) version patch

version-minor: ## Bump minor version
	@echo "$(GREEN)Bumping minor version...$(NC)"
	$(POETRY) version minor

version-major: ## Bump major version
	@echo "$(GREEN)Bumping major version...$(NC)"
	$(POETRY) version major

build: ## Build distribution packages
	@echo "$(GREEN)Building packages...$(NC)"
	$(POETRY) build

publish-test: ## Publish to TestPyPI
	@echo "$(YELLOW)Publishing to TestPyPI...$(NC)"
	$(POETRY) publish -r testpypi

publish: ## Publish to PyPI
	@echo "$(RED)Publishing to PyPI...$(NC)"
	$(POETRY) publish

release: check build ## Full release process
	@echo "$(GREEN)Release process complete!$(NC)"
	@echo "$(YELLOW)Don't forget to:$(NC)"
	@echo "  1. Create a git tag: git tag -a v$$($(POETRY) version -s) -m 'Release v$$($(POETRY) version -s)'"
	@echo "  2. Push the tag: git push origin v$$($(POETRY) version -s)"
	@echo "  3. Publish to PyPI: make publish"

# Cleaning targets
clean: ## Clean build artifacts
	@echo "$(RED)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean docker-clean ## Clean everything including Docker
	@echo "$(RED)Cleaning all resources...$(NC)"
	rm -rf .venv/
	rm -rf benchmarks/*.json
	rm -rf experiment_results/*
	rm -rf mlruns/
	rm -rf test_report.json

# Utility targets
shell: ## Start a Python shell with project context
	@echo "$(GREEN)Starting Python shell...$(NC)"
	$(POETRY) run python

validate-config: ## Validate TOML configuration files
	@echo "$(GREEN)Validating configuration files...$(NC)"
	$(POETRY) run python src/diffml/config_validation.py validate configs/*.toml

gpu-check: ## Check GPU availability
	@echo "$(GREEN)Checking GPU availability...$(NC)"
	$(POETRY) run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# CI/CD targets
ci: lint test ## Run CI checks locally
	@echo "$(GREEN)CI checks passed!$(NC)"

pre-commit: format lint test ## Run pre-commit checks
	@echo "$(GREEN)Pre-commit checks passed!$(NC)"

.PHONY: all
all: clean install test lint ## Run everything

# Special targets
.DEFAULT_GOAL := help