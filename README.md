# Latent Lab

ML/DL/LLM experimentation lab optimized for **MacBook Air M5** (32GB unified memory, 10 CPU cores).

## Architecture

```
latent-lab/
├── configs/                    # Hydra experiment configs
│   ├── base.yaml               # Default config schema
│   └── experiments/            # Per-experiment YAML overrides
│       ├── ml_classification.yaml
│       ├── ml_regression.yaml
│       ├── dl_cnn_cifar10.yaml
│       ├── dl_autoencoder.yaml
│       ├── dl_gan_mnist.yaml
│       ├── dl_diffusion_inference.yaml
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
│   ├── domains/                # Domain runners (ML, DL, LLM, CV, NLP, RL)
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
uv sync --all-extras     # install everything (ml, dl, llm, cv, rl, serving, notebooks, dev)
brew install libomp       # required for XGBoost/LightGBM on macOS

# 2. Verify Apple Silicon ML setup
make device-check
# → PyTorch: 2.11.0, MPS: True
# → MLX: default device = Device(gpu, 0)

# 3. Run your first experiment
uv run lab run ml_classification

# 4. View results in MLflow
make serve-mlflow         # → http://localhost:5000
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

## Verified Experiments

All experiments below have been tested and confirmed working on MacBook Air M5 (32GB).

### Classical ML (`domain: ml`)

```bash
uv run lab run ml_classification                              # RandomForest on Iris
uv run lab run ml_classification -o model.name=xgboost        # XGBoost on Iris
uv run lab run ml_classification -o model.name=lightgbm -o data.name=wine  # LightGBM on Wine
uv run lab run ml_regression                                  # XGBoost regression on California Housing
```

| Experiment | Algorithm | Dataset | Result |
|-----------|-----------|---------|--------|
| Classification | RandomForest | Iris | CV 95.0%, Test Acc 90.0% |
| Classification | XGBoost | Iris | CV 95.0%, Test Acc 93.3% |
| Classification | LightGBM | Wine | Test Acc 100%, F1 100% |
| Regression | XGBoost | California Housing | R2 0.836, MAE 0.303 |

Supported algorithms: `random_forest`, `xgboost`, `lightgbm`, `svm`, `logistic_regression`
Built-in datasets: `iris`, `wine`, `digits`, `breast_cancer`, `california` (or path to CSV/Parquet)

### Deep Learning (`domain: dl`)

```bash
uv run lab run dl_cnn_cifar10 -o training.epochs=3            # CNN on CIFAR-10 (quick test)
uv run lab run dl_autoencoder -o training.epochs=10            # Autoencoder on MNIST
uv run lab run dl_gan_mnist -o training.epochs=10              # GAN on MNIST
uv run lab run dl_diffusion_inference                          # Stable Diffusion (needs model download)
```

| Experiment | Model | Dataset | Result | Device | Time |
|-----------|-------|---------|--------|--------|------|
| CNN | Simple CNN (95K params) | CIFAR-10 | 56.4% (3ep), ~80%+ (20ep) | MPS GPU | ~2min/3ep |
| Autoencoder | Conv AE (latent=32) | MNIST | Recon Loss 0.0047 | MPS GPU | ~1min/10ep |
| GAN | DCGAN | MNIST | G:1.39, D:1.01 (stable) | MPS GPU | ~50sec/10ep |
| Diffusion | SD-Turbo | text-to-image | 512x512 image generated, 30 steps | MPS GPU | ~40sec |

CNN supports timm pretrained models: `-o model.pretrained=resnet18` (uses `timm.create_model`)

### LLM (`domain: llm`)

```bash
uv run lab run llm_inference -o model.pretrained=mlx-community/Llama-3.2-3B-Instruct-4bit
uv run lab run llm_inference -o model.pretrained=mlx-community/Qwen2.5-7B-Instruct-4bit
uv run lab serve mlx-community/Llama-3.2-3B-Instruct-4bit    # OpenAI-compatible API
```

| Experiment | Model | Backend | Result |
|-----------|-------|---------|--------|
| Inference | Llama 3.2 3B (4-bit) | MLX GPU | ~6sec load, fast generation |

First run downloads the model (~2.5GB for 3B). Subsequent runs use cache.

### Computer Vision (`domain: cv`)

```bash
uv run lab run cv_classification                              # ResNet18 on MPS
uv run lab run cv_detection_yolo                              # YOLO11n on COCO8 (50 epochs)
```

| Experiment | Model | Dataset | Result | Device |
|-----------|-------|---------|--------|--------|
| Classification | ResNet18 (pretrained) | - | Model loaded on MPS | MPS GPU |
| Detection | YOLO11n (50ep) | COCO8 | **mAP50: 89.8%, mAP50-95: 64.7%** | MPS GPU |

### NLP (`domain: nlp`)

```bash
uv run lab run nlp_classification                             # DistilBERT on SST-2
```

| Experiment | Model | Task | Result | Device |
|-----------|-------|------|--------|--------|
| Classification | DistilBERT (SST-2) | Sentiment | Model loaded on MPS | MPS GPU |

### Reinforcement Learning (`domain: rl`)

```bash
uv run lab run rl_cartpole                                    # PPO on CartPole-v1
uv run lab run rl_cartpole -o model.name=A2C                  # A2C algorithm
```

| Experiment | Algorithm | Environment | Result | Time |
|-----------|-----------|-------------|--------|------|
| CartPole | PPO | CartPole-v1 | Converged (value_loss ~0), 5293 FPS | ~18sec |

## Hydra Config System

All experiments are driven by YAML configs with full override support:

```bash
# Override any parameter at runtime
uv run lab run ml_classification \
  -o model.name=xgboost \
  -o data.name=digits \
  -o model.n_estimators=200 \
  -o training.seed=123

# Override training hyperparameters
uv run lab run dl_cnn_cifar10 \
  -o training.epochs=50 \
  -o training.batch_size=64 \
  -o training.learning_rate=3e-4
```

Config inheritance: `experiments/*.yaml` extends `base.yaml` with `defaults: [base, _self_]`.

## Model Recommendations for 32GB

| Model | Quantization | Memory | Use Case |
|-------|-------------|--------|----------|
| Llama 3.2 3B | 4-bit | ~2.5 GB | Development, fast iteration |
| Mistral 7B | 4-bit | ~4.5 GB | General purpose |
| Qwen 14B | 4-bit | ~9 GB | High quality, multilingual (CJK) |
| Qwen 30B MoE | 4-bit | ~17 GB | Maximum quality (tight on 32GB) |

```bash
uv run python scripts/download_models.py list-models     # see all recommendations
uv run python scripts/download_models.py download mistral-7b-4bit
```

> **Note**: 32GB unified memory is shared with macOS. Expect ~26-28GB usable for models.
> MacBook Air has passive cooling — sustained training workloads may throttle.

## MLflow Experiment Tracking

All experiments automatically log to MLflow:

```bash
make serve-mlflow    # → http://localhost:5000
```

Logged per run: config params, metrics (accuracy, loss, etc.), system metrics (CPU/memory), artifacts.

## Tech Stack

| Layer | Tool | Role |
|-------|------|------|
| Package management | **uv** | Fast Python dependency resolution |
| Classical ML | **scikit-learn, XGBoost, LightGBM** | Classification, regression, clustering |
| Deep Learning | **PyTorch MPS** | CNN, GAN, Autoencoder training on Apple GPU |
| LLM | **MLX, mlx-lm** | Apple Silicon native inference & fine-tuning |
| Data | **Polars, DuckDB** | Fast DataFrame & SQL analytics |
| Config | **Hydra + OmegaConf** | YAML-based experiment configuration |
| Tracking | **MLflow** | Experiment logging & comparison |
| Notebooks | **Marimo** | Reactive, git-friendly (.py) notebooks |
| Serving | **FastAPI, Gradio** | Model APIs & interactive demos |
| Dev tools | **Ruff, mypy, pytest** | Linting, type checking, testing |

## Why Not TensorFlow / JAX?

| Framework | Status on Apple Silicon (2025-2026) | Decision |
|-----------|-------------------------------------|----------|
| **PyTorch MPS** | Functional, actively maintained | Primary DL framework |
| **MLX** | Apple-native, best LLM perf (up to 4x vs M4) | Primary for LLM |
| **TensorFlow** | `tensorflow-metal` stalled at v1.2.0, only supports TF ≤2.18 | Excluded |
| **JAX** | `jax-metal` abandoned, community `jax-mps` very early | Excluded |

## Known Issues

| Issue | Status | Workaround |
|-------|--------|------------|
| **ollama crash on macOS 26 + M5** | ollama 0.18.3 Metal shader compilation fails (`bfloat`/`half` type mismatch in `MetalPerformancePrimitives`). | Use MLX-LM for local LLM inference instead. RAG notebook requires ollama update. |
| **SD 2.1 download fails unauthenticated** | Some HF models require auth token. | Use `stabilityai/sd-turbo` (works without auth) or set `HF_TOKEN`. |
| **PyTorch MPS no float64** | Double precision not supported on MPS GPU. | Offload to CPU: `tensor.to("cpu").double()` |
