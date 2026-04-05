# Latent Lab

A personal ML experimentation lab for Apple Silicon. One CLI to run everything from classical ML to LLM fine-tuning and RAG pipelines, with Hydra configs for parameter overrides and MLflow for tracking.

```bash
uv run lab run ml_classification                    # RandomForest on Iris
uv run lab run ml_classification -o model.name=xgboost -o data.name=wine
uv run lab run dl_cnn_cifar10 -o training.epochs=20  # CNN on CIFAR-10 (MPS GPU)
uv run lab run llm_inference -o model.pretrained=mlx-community/Llama-3.2-3B-Instruct-4bit
```

## Why

- **No cohesive ML stack for Apple Silicon.** TensorFlow Metal is effectively abandoned, JAX Metal is incomplete. PyTorch MPS + MLX is the practical choice, but there was no unified environment to experiment with both.
- **Too much boilerplate per experiment.** Data loading, device setup, logging, config management — instead of rewriting these every time, every experiment exposes a single `run(cfg)` entry point.
- **Hard to compare results.** Swapping algorithms on the same task requires consistent config and tracking. Hydra + MLflow handles this.

## Setup

```bash
# Install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras
brew install libomp  # Required for XGBoost/LightGBM on macOS

# Verify Apple Silicon ML setup
make device-check
# → PyTorch MPS: True, MLX: Device(gpu, 0)
```

## What You Can Do

| Domain | Examples | Framework |
|--------|----------|-----------|
| **ML** | Classification, Regression, Clustering, Optuna tuning, SHAP explainability | scikit-learn, XGBoost, LightGBM |
| **DL** | CNN, Autoencoder, VAE, GAN, Stable Diffusion, optimization | PyTorch MPS, timm, diffusers |
| **LLM** | Inference, LoRA fine-tuning, distillation, quantization, prompting comparison | MLX, mlx-lm |
| **NLP** | Text classification, NER, tokenizer analysis, embeddings | HuggingFace Transformers |
| **CV** | Image classification, YOLO detection, CoreML/ONNX export | PyTorch, ultralytics, coremltools |
| **RL** | PPO/A2C/SAC on classic control environments | Gymnasium, Stable-Baselines3 |
| **RAG** | Local vector search pipeline, chunking, retrieval evaluation | ChromaDB, Ollama, LangChain |

Every experiment runs with `uv run lab run {config_name}` and supports `-o key=value` overrides.

```bash
# Compare ML algorithms
uv run lab run ml_classification -o model.name=lightgbm -o data.name=digits

# Train a VAE
uv run lab run dl_vae_mnist -o training.epochs=20

# LLM inference with a different model
uv run lab run llm_inference -o model.pretrained=mlx-community/Qwen2.5-7B-Instruct-4bit

# Hyperparameter tuning with Optuna
uv run lab run ml_tuning_optuna -o tuning.n_trials=50

# RAG pipeline (requires ollama)
uv run lab run rag_pipeline
```

## How It Works

```
configs/experiments/ml_classification.yaml    →  src/latent_lab/ml/classification.py:run(cfg)
configs/experiments/llm_finetune_lora.yaml    →  src/latent_lab/llm/finetune.py:run(cfg)
configs/experiments/rag_pipeline.yaml         →  src/latent_lab/rag/pipeline.py:run(cfg)
```

1. The YAML config's `domain` + `task` fields determine which module to load
2. The runner dynamically imports `latent_lab.{domain}.{task}` via `importlib`
3. It calls the module's `run(cfg: DictConfig)` function
4. MLflow automatically logs config, metrics, and artifacts

Example config (`configs/experiments/ml_classification.yaml`):

```yaml
defaults:
  - base
  - _self_

name: ml_classification_iris
domain: ml
task: classification

data:
  name: iris

model:
  name: random_forest
  n_estimators: 100

training:
  seed: 42
```

## Project Structure

```
src/latent_lab/
├── ml/           # classification, regression, clustering, tuning, explainability
├── dl/           # cnn, autoencoder, gan, diffusion, optimization
├── llm/          # inference, finetune, distillation, quantize, evaluation, prompting
├── nlp/          # classification, ner, tokenizer, embeddings
├── cv/           # classification, detection, export
├── rl/           # classic, custom
├── rag/          # pipeline, chunking, retrieval, evaluation
├── experiments/  # Hydra runner + MLflow tracker
├── serving/      # FastAPI model server
├── data/         # Data loading (Polars)
└── utils/        # Memory monitoring, helpers

configs/experiments/   # {domain}_{task}.yaml (flat)
notebooks/             # Marimo notebooks by domain
```

## MLflow Tracking

```bash
make serve-mlflow  # http://localhost:5000
```

All experiments automatically log to MLflow: config params, metrics, system metrics, and artifacts.

## Model Recommendations (32GB)

| Model | Quantization | Memory | Use Case |
|-------|-------------|--------|----------|
| Llama 3.2 3B | 4-bit | ~2.5 GB | Fast iteration |
| Mistral 7B | 4-bit | ~4.5 GB | General purpose |
| Qwen 14B | 4-bit | ~9 GB | High quality, multilingual |

```bash
uv run python scripts/download_models.py list-models
```

## Known Issues

- **ollama >= 0.13**: Metal shader crash on macOS 26 + M5. Use [0.12.4](https://github.com/ollama/ollama/releases/tag/v0.12.4) instead.
- **PyTorch MPS**: No float64 support. Fall back to CPU: `tensor.to("cpu").double()`
- **SD 2.1**: Requires HF auth token. Use `stabilityai/sd-turbo` (no auth needed) or set `HF_TOKEN`.
