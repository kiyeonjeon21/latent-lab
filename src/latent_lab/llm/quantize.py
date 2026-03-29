"""LLM quantization - convert models and compare quality."""

from pathlib import Path

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def convert(cfg: DictConfig) -> None:
    """Convert a HuggingFace model to MLX quantized format."""
    from latent_lab.models.mlx_utils import quantize_model

    q_bits = cfg.model.get("quantize", 4)
    output = Path(f"models/weights/{cfg.name}-q{q_bits}")
    console.print(f"[cyan]Quantizing {cfg.model.pretrained} → {q_bits}-bit[/cyan]")
    quantize_model(cfg.model.pretrained, output, q_bits=q_bits)
    console.print(f"[green]Saved to {output}[/green]")


def compare(cfg: DictConfig) -> None:
    """Compare quality across quantization levels."""
    import time

    from latent_lab.experiments.tracker import log_metrics
    from latent_lab.models.mlx_utils import generate, load_mlx_model

    prompt = cfg.get("prompt", "Explain quantum computing in simple terms.")
    bit_levels = cfg.get("bit_levels", [4, 8])
    base_model = cfg.model.pretrained

    for bits in bit_levels:
        model_path = f"models/weights/{cfg.name}-q{bits}"
        if not Path(model_path).exists():
            console.print(f"[yellow]Skipping {bits}-bit (not found at {model_path}). Run convert first.[/yellow]")
            continue

        console.print(f"\n[bold cyan]{bits}-bit model[/bold cyan]")
        model, tokenizer = load_mlx_model(model_path)

        start = time.perf_counter()
        response = generate(model, tokenizer, prompt=prompt, max_tokens=128)
        elapsed = time.perf_counter() - start

        console.print(f"[green]Response ({elapsed:.1f}s):[/green] {response[:200]}")
        log_metrics({f"q{bits}/time_s": elapsed, f"q{bits}/tokens": len(response.split())})


def run(cfg: DictConfig) -> None:
    """Run quantization experiment (convert or compare)."""
    subtask = cfg.get("subtask", "convert")
    match subtask:
        case "convert":
            convert(cfg)
        case "compare":
            compare(cfg)
