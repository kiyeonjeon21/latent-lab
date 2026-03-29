"""Image classification with PyTorch MPS."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
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
