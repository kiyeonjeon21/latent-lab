"""Configuration utilities for experiment management."""

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class DataConfig:
    name: str = MISSING
    path: str = ""
    split: str = "train"
    max_samples: int | None = None


@dataclass
class ModelConfig:
    name: str = MISSING
    framework: str = "mlx"  # mlx, torch, hf
    pretrained: str = ""
    quantize: int | None = None  # None, 4, 8


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int | None = None
    gradient_accumulation_steps: int = 1
    seed: int = 42


@dataclass
class LoRAConfig:
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    layers: int = 16


@dataclass
class ExperimentConfig:
    name: str = MISSING
    description: str = ""
    domain: str = "general"  # llm, cv, nlp, rl, general
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    tags: list[str] = field(default_factory=list)


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_config(config_path: str | Path) -> DictConfig:
    """Load an experiment config from YAML file."""
    return OmegaConf.load(config_path)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configs (later configs override earlier ones)."""
    return OmegaConf.merge(*configs)
