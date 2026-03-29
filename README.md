# Latent Lab

ML/DL/LLM experimentation lab optimized for **MacBook Air M5** (32GB unified memory, 10 CPU cores).

## Architecture

```
latent-lab/
├── configs/                    # Hydra experiment configs
│   ├── base.yaml               # Default config schema
│   └── experiments/            # Per-experiment YAML overrides
│       ├── llm_inference.yaml
│       ├── llm_finetune_lora.yaml
│       ├── cv_classification.yaml
│       ├── cv_detection_yolo.yaml
│       ├── nlp_classification.yaml
│       └── rl_cartpole.yaml
├── data/                       # Data (gitignored, use DVC for versioning)
│   ├── raw/                    # Immutable original data
│   ├── interim/                # Intermediate transforms
│   ├── processed/              # Final datasets
│   └── external/               # Third-party data
├── models/                     # Model artifacts (gitignored)
│   ├── weights/                # Downloaded model weights
│   ├── checkpoints/            # Training checkpoints
│   ├── adapters/               # LoRA adapters
│   └── exports/                # Exported models (CoreML, ONNX)
├── notebooks/                  # Marimo notebooks (.py, git-friendly)
│   ├── 01_device_check.py      # Apple Silicon environment check
│   ├── 02_mlx_inference.py     # MLX LLM inference playground
│   └── 03_rag_local.py         # Local RAG with ChromaDB + Ollama
├── scripts/                    # CLI utilities
│   ├── setup_environment.sh    # One-command setup
│   └── download_models.py      # Model downloader with recommendations
├── src/latent_lab/             # Main source package
│   ├── cli.py                  # Typer CLI (device-info, run, serve)
│   ├── config.py               # Experiment config dataclasses
│   ├── data/                   # Data loading (Polars)
│   ├── models/                 # MLX & PyTorch utilities
│   ├── domains/                # Domain-specific runners (LLM, CV, NLP, RL)
│   ├── experiments/            # Tracker (MLflow) & Hydra runner
│   ├── serving/                # FastAPI model server
│   └── utils/                  # Memory monitoring & helpers
├── tests/                      # pytest tests
├── pyproject.toml              # All config in one place
└── Makefile                    # Common commands
```

## Quick Start

```bash
# 1. Setup
./scripts/setup_environment.sh

# or manually:
uv sync                  # core deps
uv sync --all-extras     # everything (llm, cv, rl, serving, notebooks, dev)

# 2. Verify setup
make device-check

# 3. Run experiments
uv run lab device-info                               # check ML frameworks
uv run lab run llm_inference                          # run an experiment
uv run lab serve mlx-community/Llama-3.2-3B-Instruct-4bit  # serve a model

# 4. Open notebooks
make notebook                                         # marimo editor
uv run marimo edit notebooks/01_device_check.py       # specific notebook
```

## Experiment Domains

| Domain | Framework | Key Capabilities |
|--------|-----------|-----------------|
| **ML** | scikit-learn, XGBoost, LightGBM | Classification, regression, clustering, cross-validation |
| **DL** | PyTorch MPS, timm, diffusers | CNN, Autoencoder, GAN, Stable Diffusion inference |
| **LLM** | MLX, mlx-lm | Inference, LoRA/QLoRA fine-tuning, quantization, serving |
| **CV** | PyTorch MPS, YOLO, CoreML | Classification, object detection, CoreML export |
| **NLP** | HuggingFace Transformers | Text classification, NER, embeddings |
| **RL** | Gymnasium, Stable-Baselines3 | PPO/A2C/SAC, classic control, MuJoCo |
| **RAG** | ChromaDB, Ollama, LangChain | Local vector search, embedding, retrieval |

## Model Recommendations for 32GB

| Model | Quantization | Memory | Use Case |
|-------|-------------|--------|----------|
| Llama 3.2 3B | 4-bit | ~2.5 GB | Development, fast iteration |
| Mistral 7B | 4-bit | ~4.5 GB | General purpose |
| Qwen 14B | 4-bit | ~9 GB | High quality, multilingual |
| Qwen 30B MoE | 4-bit | ~17 GB | Maximum quality (tight fit) |

```bash
# List recommended models
uv run python scripts/download_models.py list-models

# Download a model
uv run python scripts/download_models.py download mistral-7b-4bit
```

## Tech Stack

- **Python 3.12** + **uv** (package management)
- **MLX** (Apple Silicon native ML) + **PyTorch MPS** (ecosystem compatibility)
- **Hydra + OmegaConf** (experiment configuration)
- **MLflow** (experiment tracking) | `make serve-mlflow`
- **Polars + DuckDB** (data processing)
- **Marimo** (reactive notebooks, git-friendly)
- **Ruff** (linting/formatting) + **mypy** (type checking) + **pytest** (testing)
