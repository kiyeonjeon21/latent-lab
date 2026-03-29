"""DL optimization experiments - scheduler comparison, pruning, ONNX export."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Run optimization experiment."""
    subtask = cfg.get("subtask", "scheduler_compare")
    match subtask:
        case "scheduler_compare":
            compare_schedulers(cfg)
        case "export_onnx":
            export_onnx(cfg)
        case _:
            console.print(f"[red]Unknown optimization subtask: {subtask}[/red]")


def compare_schedulers(cfg: DictConfig) -> None:
    """Compare learning rate schedulers on the same model/data."""
    console.print("[yellow]Scheduler comparison: implement with your specific model.[/yellow]")
    console.print("[cyan]Available schedulers: CosineAnnealing, StepLR, OneCycleLR, ReduceOnPlateau[/cyan]")


def export_onnx(cfg: DictConfig) -> None:
    """Export a PyTorch model to ONNX format."""
    import torch

    model_path = cfg.get("model_path", f"models/checkpoints/{cfg.name}.pt")
    output_path = cfg.get("output_path", f"models/exports/{cfg.name}.onnx")

    console.print(f"[cyan]Exporting {model_path} → {output_path}[/cyan]")
    console.print("[yellow]Load your model and call torch.onnx.export() here.[/yellow]")
