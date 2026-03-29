"""Custom RL environments and experiments."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Run custom RL experiment."""
    console.print("[yellow]Custom RL environment: implement your environment here.[/yellow]")
    console.print("[cyan]Use gymnasium.make() with a registered custom env.[/cyan]")
