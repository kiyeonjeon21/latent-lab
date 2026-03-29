.PHONY: help install install-all lint format test clean serve-mlflow

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install core dependencies
	uv sync

install-all: ## Install all optional dependencies
	uv sync --all-extras

lint: ## Run linter
	uv run ruff check src/ tests/ scripts/
	uv run mypy src/

format: ## Format code
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

test: ## Run tests
	uv run pytest -v

clean: ## Clean build artifacts
	rm -rf dist/ build/ *.egg-info/ __pycache__/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

serve-mlflow: ## Start MLflow tracking server
	uv run mlflow ui --port 5000

notebook: ## Open marimo notebook editor
	uv run marimo edit notebooks/

device-check: ## Check Apple Silicon ML setup
	uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, MPS: {torch.backends.mps.is_available()}'); import mlx.core as mx; print(f'MLX: default device = {mx.default_device()}')"
