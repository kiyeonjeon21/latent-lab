"""Tests for configuration loading."""

from pathlib import Path

from latent_lab.config import (
    DATA_DIR,
    ExperimentConfig,
    ModelConfig,
    PROJECT_ROOT,
    TrainingConfig,
)


def test_project_root_exists():
    assert PROJECT_ROOT.exists()


def test_data_dir_path():
    assert DATA_DIR == PROJECT_ROOT / "data"


def test_experiment_config_defaults():
    cfg = ExperimentConfig(name="test")
    assert cfg.name == "test"
    assert cfg.domain == "general"
    assert cfg.training.epochs == 10
    assert cfg.training.seed == 42


def test_model_config():
    cfg = ModelConfig(name="test-model", framework="mlx", pretrained="some/model")
    assert cfg.framework == "mlx"
    assert cfg.quantize is None


def test_training_config():
    cfg = TrainingConfig(epochs=5, learning_rate=3e-4)
    assert cfg.epochs == 5
    assert cfg.learning_rate == 3e-4
    assert cfg.batch_size == 8  # default
