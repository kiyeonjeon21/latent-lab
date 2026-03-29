"""Computer Vision experiment domain."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run_experiment(cfg: DictConfig) -> None:
    """Run a CV experiment."""
    from latent_lab.experiments.tracker import log_config, setup_tracking, track_run

    setup_tracking(f"cv-{cfg.name}")

    with track_run(run_name=cfg.name, tags={"domain": "cv"}):
        log_config(cfg)

        task = cfg.get("task", "classify")

        match task:
            case "classify":
                _run_classification(cfg)
            case "detect":
                _run_detection(cfg)
            case _:
                console.print(f"[red]Unknown CV task: {task}[/red]")


def _run_classification(cfg: DictConfig) -> None:
    """Image classification with PyTorch MPS."""
    import torch
    from torchvision import models

    from latent_lab.models.torch_utils import get_device, seed_everything

    seed_everything(cfg.training.seed)
    device = get_device()
    console.print(f"[cyan]Using device: {device}[/cyan]")

    model_name = cfg.model.get("pretrained", "resnet18")
    model = getattr(models, model_name)(weights="DEFAULT").to(device)
    console.print(f"[green]Loaded {model_name} on {device}[/green]")
    console.print("[yellow]Add your training loop here.[/yellow]")


def _run_detection(cfg: DictConfig) -> None:
    """Object detection with YOLO."""
    console.print("[cyan]Loading YOLO model...[/cyan]")
    from ultralytics import YOLO

    model_path = cfg.model.get("pretrained", "yolo11n.pt")
    model = YOLO(model_path)

    if cfg.data.path:
        results = model.train(
            data=cfg.data.path,
            epochs=cfg.training.epochs,
            batch=cfg.training.batch_size,
            device="mps",
        )
        console.print(f"[green]Training complete. Results: {results}[/green]")
