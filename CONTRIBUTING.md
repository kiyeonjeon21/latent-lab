# Contributing to Latent Lab

Thanks for your interest in contributing! This is a personal ML experimentation lab, but contributions are welcome — especially for completing domain modules and adding tests.

## Getting Started

```bash
# Clone and install
git clone https://github.com/kiyeonjeon21/latent-lab.git
cd latent-lab
uv sync --all-extras
pre-commit install

# Verify setup
make device-check    # Check Apple Silicon ML environment
make test            # Run tests
make lint            # Lint + format check
```

> **Note:** Some experiments require Apple Silicon (MPS GPU, MLX). If you're on Linux/Windows, you can still contribute to ML domain modules (scikit-learn, XGBoost) and infrastructure (tests, CI).

## How to Add a New Experiment

Every experiment follows the same pattern:

### 1. Create the module

```python
# src/latent_lab/{domain}/{task}.py
from omegaconf import DictConfig
from rich.console import Console
from latent_lab.experiments.tracker import log_config, log_metrics, setup_tracking, track_run

console = Console()

def run(cfg: DictConfig) -> None:
    """One-line description of what this experiment does."""
    setup_tracking(f"{cfg.domain}-{cfg.name}")
    with track_run(run_name=cfg.name, tags={"domain": cfg.domain, "task": cfg.task}):
        log_config(cfg)
        # Your experiment logic here
        log_metrics({"accuracy": acc})
```

### 2. Add a config

```yaml
# configs/experiments/{domain}_{task}.yaml
defaults:
  - base
  - _self_

name: {domain}_{task}_description
domain: {domain}
task: {task}

data:
  name: dataset_name

model:
  name: model_name

training:
  seed: 42
```

### 3. Add a smoke test

```python
# tests/test_{domain}_smoke.py
from omegaconf import OmegaConf
from latent_lab.{domain}.{task} import run

def test_{domain}_{task}_smoke():
    cfg = OmegaConf.create({...})  # minimal config
    run(cfg)  # should not raise
```

### 4. Run it

```bash
uv run lab run {domain}_{task}
```

## Code Conventions

- **Interface:** Every task module exposes `run(cfg: DictConfig) -> None`
- **Config naming:** `configs/experiments/{domain}_{task}.yaml` (flat, no subdirectories)
- **Formatting:** Ruff handles this — just run `make lint` before committing
- **Line length:** 100 characters
- **Dependencies:** Add to the appropriate group in `pyproject.toml` (`[project.dependencies]` for core, `[project.optional-dependencies]` for domain-specific)

## What to Work On

Check the [open issues](https://github.com/kiyeonjeon21/latent-lab/issues) — issues labeled `good first issue` are a good starting point. Domain completion issues have checklists showing exactly what's needed.

## Pull Requests

1. Fork the repo and create a branch from `main`
2. Make your changes and add tests if applicable
3. Run `make lint && make test` to verify
4. Open a PR with a short description of what you changed and why

Keep PRs focused — one domain or one issue per PR is ideal.
