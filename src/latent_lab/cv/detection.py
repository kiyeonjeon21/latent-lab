"""Object detection with YOLO."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
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
