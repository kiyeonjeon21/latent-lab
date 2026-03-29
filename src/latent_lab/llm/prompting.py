"""LLM prompting strategies - system prompts, few-shot, chain-of-thought."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Run prompting experiment."""
    strategy = cfg.get("strategy", "compare")
    match strategy:
        case "compare":
            compare_strategies(cfg)
        case _:
            console.print(f"[red]Unknown strategy: {strategy}[/red]")


def compare_strategies(cfg: DictConfig) -> None:
    """Compare different prompting strategies on the same task."""
    from latent_lab.experiments.tracker import log_metrics
    from latent_lab.models.mlx_utils import generate, load_mlx_model

    model, tokenizer = load_mlx_model(cfg.model.pretrained)
    base_prompt = cfg.get("prompt", "What is 15% of 240?")
    max_tokens = cfg.get("max_tokens", 256)

    strategies = {
        "zero_shot": base_prompt,
        "cot": f"{base_prompt}\nLet's think step by step.",
        "few_shot": cfg.get("few_shot_prompt", f"Q: What is 10% of 100?\nA: 10\n\nQ: {base_prompt}\nA:"),
    }

    for name, prompt in strategies.items():
        console.print(f"\n[bold cyan]Strategy: {name}[/bold cyan]")
        console.print(f"[dim]{prompt}[/dim]")
        response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
        console.print(f"[green]Response:[/green] {response[:300]}")
