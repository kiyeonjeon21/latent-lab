"""Hydra-based experiment runner with dynamic domain/task routing."""

import importlib
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

console = Console()

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "configs" / "experiments")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="base")
def main(cfg: DictConfig) -> None:
    """Run an experiment based on Hydra config."""
    from latent_lab.experiments.tracker import log_config, setup_tracking, track_run

    domain = cfg.get("domain", "general")
    task = cfg.get("task", "")

    console.print(f"[bold cyan]Experiment:[/bold cyan] {cfg.name}")
    console.print(f"[bold cyan]Domain:[/bold cyan] {domain}")
    console.print(f"[bold cyan]Task:[/bold cyan] {task}")
    console.print(OmegaConf.to_yaml(cfg))

    # Dynamic import: latent_lab.{domain}.{task}
    try:
        module = importlib.import_module(f"latent_lab.{domain}.{task}")
    except ModuleNotFoundError:
        # Fallback to old domains/ structure for backwards compat
        try:
            module = importlib.import_module(f"latent_lab.domains.{domain}")
            setup_tracking(f"{domain}-{cfg.name}")
            with track_run(run_name=cfg.name, tags={"domain": domain}):
                log_config(cfg)
                module.run_experiment(cfg)
            return
        except (ModuleNotFoundError, AttributeError):
            console.print(f"[red]Cannot find module: latent_lab.{domain}.{task}[/red]")
            console.print(f"[yellow]Available domains: llm, dl, ml, nlp, cv, rl, rag[/yellow]")
            return

    if not hasattr(module, "run"):
        console.print(f"[red]Module latent_lab.{domain}.{task} has no run() function[/red]")
        return

    setup_tracking(f"{domain}-{cfg.name}")
    with track_run(run_name=cfg.name, tags={"domain": domain, "task": task}):
        log_config(cfg)
        module.run(cfg)


if __name__ == "__main__":
    main()
