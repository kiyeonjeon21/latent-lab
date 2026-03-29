"""Hydra-based experiment runner."""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

console = Console()

# Resolve config path relative to project root, not this file
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "configs" / "experiments")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="base")
def main(cfg: DictConfig) -> None:
    """Run an experiment based on Hydra config."""
    console.print(f"[bold cyan]Experiment:[/bold cyan] {cfg.name}")
    console.print(f"[bold cyan]Domain:[/bold cyan] {cfg.domain}")
    console.print(OmegaConf.to_yaml(cfg))

    # Import and run domain-specific logic
    match cfg.domain:
        case "ml":
            from latent_lab.domains.ml import run_experiment
        case "dl":
            from latent_lab.domains.dl import run_experiment
        case "llm":
            from latent_lab.domains.llm import run_experiment
        case "cv":
            from latent_lab.domains.cv import run_experiment
        case "nlp":
            from latent_lab.domains.nlp import run_experiment
        case "rl":
            from latent_lab.domains.rl import run_experiment
        case _:
            console.print(f"[yellow]No domain runner for '{cfg.domain}', printing config only.[/yellow]")
            return

    run_experiment(cfg)


if __name__ == "__main__":
    main()
