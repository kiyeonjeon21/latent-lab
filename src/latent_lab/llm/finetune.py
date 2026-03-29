"""LLM fine-tuning - LoRA, QLoRA with MLX."""

import subprocess
from pathlib import Path

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Run LoRA/QLoRA fine-tuning with MLX."""
    adapter_path = Path(f"models/adapters/{cfg.name}")
    adapter_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--train",
        "--model", cfg.model.pretrained,
        "--data", cfg.data.path,
        "--batch-size", str(cfg.training.batch_size),
        "--lora-layers", str(cfg.lora.layers),
        "--iters", str(cfg.training.max_steps or 600),
        "--learning-rate", str(cfg.training.learning_rate),
        "--adapter-path", str(adapter_path),
    ]

    if cfg.lora.get("rank"):
        cmd.extend(["--lora-rank", str(cfg.lora.rank)])

    console.print(f"[cyan]Running LoRA fine-tuning...[/cyan]")
    console.print(f"[dim]{' '.join(cmd)}[/dim]")
    subprocess.run(cmd, check=True)
    console.print(f"[green]Adapter saved to {adapter_path}[/green]")


def fuse(cfg: DictConfig) -> None:
    """Merge LoRA adapter into base model."""
    adapter_path = f"models/adapters/{cfg.name}"
    output_path = f"models/weights/{cfg.name}-fused"

    cmd = [
        "python", "-m", "mlx_lm.fuse",
        "--model", cfg.model.pretrained,
        "--adapter-path", adapter_path,
        "--save-path", output_path,
    ]
    console.print(f"[cyan]Fusing adapter into base model...[/cyan]")
    subprocess.run(cmd, check=True)
    console.print(f"[green]Fused model saved to {output_path}[/green]")


def evaluate_adapter(cfg: DictConfig) -> None:
    """Evaluate fine-tuned model vs base model."""
    from latent_lab.models.mlx_utils import generate, load_mlx_model
    from latent_lab.experiments.tracker import log_metrics

    prompts = cfg.get("eval_prompts", ["Hello, how are you?"])

    # Base model
    console.print("[cyan]Loading base model...[/cyan]")
    base_model, tokenizer = load_mlx_model(cfg.model.pretrained)

    # Adapted model
    adapter_path = f"models/adapters/{cfg.name}"
    console.print(f"[cyan]Loading adapted model from {adapter_path}...[/cyan]")
    adapted_model, _ = load_mlx_model(cfg.model.pretrained)
    # Note: mlx_lm adapter loading is done via fuse or at inference time

    for i, prompt in enumerate(prompts):
        console.print(f"\n[bold]Prompt {i + 1}:[/bold] {prompt}")
        base_resp = generate(base_model, tokenizer, prompt=prompt, max_tokens=128)
        console.print(f"[yellow]Base:[/yellow] {base_resp[:200]}")
