# Latent Lab - ML/DL/LLM Experimentation

## Project Overview
ML/DL/LLM experimentation lab optimized for MacBook Air M5 (32GB, 10 CPU, 1TB SSD).

## Tech Stack
- **Python 3.12** with **uv** for package management
- **scikit-learn, XGBoost, LightGBM** for classical ML
- **PyTorch MPS** for DL training (CNN, GAN, VAE, Diffusion) + **timm, diffusers**
- **MLX** (Apple Silicon native, LLM inference/fine-tuning/distillation/quantization)
- **Hydra + OmegaConf** for experiment configuration
- **MLflow** for experiment tracking
- **Polars + DuckDB** for data processing
- **Ruff** for linting/formatting, **mypy** for type checking, **pytest** for tests

## Project Structure
```
src/latent_lab/
├── llm/            - LLM: inference, finetune, distillation, quantize, evaluation, prompting
├── dl/             - DL: cnn, autoencoder (AE+VAE), gan, diffusion, optimization
├── ml/             - ML: classification, regression, clustering, tuning (Optuna), explainability (SHAP)
├── nlp/            - NLP: classification, ner, tokenizer, embeddings
├── cv/             - CV: classification, detection, export (CoreML/ONNX)
├── rl/             - RL: classic (SB3), custom environments
├── rag/            - RAG: pipeline, chunking, retrieval, evaluation
├── domains/        - Legacy domain runners (backward compat)
├── models/         - MLX & PyTorch utilities
├── experiments/    - Hydra runner & MLflow tracker
├── serving/        - FastAPI model server
├── data/           - Data loading (Polars)
└── utils/          - Memory monitoring & helpers

configs/experiments/ - Flat YAML configs: {domain}_{task}.yaml
notebooks/           - Marimo notebooks organized by domain
```

## Experiment Routing
The runner uses dynamic `importlib`: `domain` + `task` → `latent_lab.{domain}.{task}.run(cfg)`

```bash
uv run lab run ml_classification        # → latent_lab.ml.classification.run()
uv run lab run llm_prompting_compare    # → latent_lab.llm.prompting.run()
uv run lab run rag_pipeline             # → latent_lab.rag.pipeline.run()
```

## Key Commands
```bash
make install          # Install core deps
make install-all      # Install all optional deps
make test             # Run tests
make lint             # Lint & type check
make device-check     # Verify Apple Silicon ML setup
make serve-mlflow     # Start MLflow UI
```

## Conventions
- Each domain is a subpackage under `src/latent_lab/` with task-specific modules
- Each task module exposes a `run(cfg: DictConfig)` function
- Configs are flat in `configs/experiments/` with naming: `{domain}_{experiment}.yaml`
- Raw data in `data/raw/` is immutable
- Notebooks organized by domain in `notebooks/{domain}/`
- Model weights go in `models/` (gitignored)
