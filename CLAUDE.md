# Latent Lab - ML/DL/LLM Experimentation

## Project Overview
ML/DL/LLM experimentation lab optimized for MacBook Air M5 (32GB, 10 CPU, 1TB SSD).

## Tech Stack
- **Python 3.12** with **uv** for package management
- **scikit-learn, XGBoost, LightGBM** for classical ML
- **PyTorch MPS** for DL training (CNN, GAN, Autoencoder) + **timm, diffusers** for pretrained models
- **MLX** (Apple Silicon native, LLM inference/fine-tuning)
- **Hydra + OmegaConf** for experiment configuration
- **MLflow** for experiment tracking
- **Polars + DuckDB** for data processing
- **Ruff** for linting/formatting, **mypy** for type checking, **pytest** for tests

## Project Structure
```
src/latent_lab/     - Main source package
configs/            - Hydra experiment configs
data/               - Data directory (raw/interim/processed/external)
models/             - Saved weights, adapters, exports
notebooks/          - Marimo notebooks (.py files)
scripts/            - CLI scripts and utilities
tests/              - pytest tests
reports/            - Generated analysis and figures
```

## Key Commands
```bash
make install          # Install core deps
make install-all      # Install all optional deps (llm, cv, rl, serving, notebooks, dev)
make test             # Run tests
make lint             # Lint & type check
make format           # Auto-format
make device-check     # Verify Apple Silicon ML setup
make serve-mlflow     # Start MLflow UI
make notebook         # Open marimo editor
```

## Conventions
- Source code lives in `src/latent_lab/` (src layout)
- Experiments are configured via YAML in `configs/experiments/`
- Raw data in `data/raw/` is immutable — never modify
- Notebooks are Marimo (.py) for git-friendliness
- Model weights go in `models/` (gitignored, large files)
- Use `lab` CLI (`uv run lab`) for common workflows
