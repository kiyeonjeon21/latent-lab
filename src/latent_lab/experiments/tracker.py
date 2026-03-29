"""Experiment tracking wrapper around MLflow."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow


def setup_tracking(
    experiment_name: str,
    tracking_uri: str = "mlruns",
) -> str:
    """Initialize MLflow tracking. Returns the experiment ID."""
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    return experiment.experiment_id


@contextmanager
def track_run(
    run_name: str,
    tags: dict[str, str] | None = None,
    log_system_metrics: bool = True,
):
    """Context manager for tracking an experiment run."""
    with mlflow.start_run(run_name=run_name, tags=tags, log_system_metrics=log_system_metrics):
        yield mlflow.active_run()


def log_params(params: dict[str, Any]):
    """Log parameters to the active run."""
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None):
    """Log metrics to the active run."""
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str | Path):
    """Log a file artifact to the active run."""
    mlflow.log_artifact(str(path))


def log_config(config) -> None:
    """Log an OmegaConf/dict config as params."""
    from omegaconf import OmegaConf

    if hasattr(config, "_metadata"):  # OmegaConf
        flat = dict(OmegaConf.to_container(config, resolve=True, throw_on_missing=False))
    else:
        flat = dict(config)

    # Flatten nested dicts
    def _flatten(d, prefix=""):
        items = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(_flatten(v, key))
            else:
                items[key] = v
        return items

    mlflow.log_params(_flatten(flat))
