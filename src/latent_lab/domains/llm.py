"""LLM experiment domain - inference, fine-tuning, evaluation."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run_experiment(cfg: DictConfig) -> None:
    """Run an LLM experiment."""
    from latent_lab.experiments.tracker import log_config, log_metrics, setup_tracking, track_run

    setup_tracking(f"llm-{cfg.name}")

    with track_run(run_name=cfg.name, tags={"domain": "llm"}):
        log_config(cfg)

        task = cfg.get("task", "inference")

        match task:
            case "inference":
                _run_inference(cfg)
            case "finetune":
                _run_finetune(cfg)
            case "evaluate":
                _run_evaluate(cfg)
            case _:
                console.print(f"[red]Unknown LLM task: {task}[/red]")


def _run_inference(cfg: DictConfig) -> None:
    """Run LLM inference experiment."""
    from latent_lab.models.mlx_utils import generate, load_mlx_model

    console.print(f"[cyan]Loading model: {cfg.model.pretrained}[/cyan]")
    model, tokenizer = load_mlx_model(cfg.model.pretrained)

    prompts = cfg.get("prompts", ["Hello, how are you?"])
    for i, prompt in enumerate(prompts):
        console.print(f"\n[bold]Prompt {i + 1}:[/bold] {prompt}")
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=cfg.get("max_tokens", 256),
            temperature=cfg.get("temperature", 0.7),
        )
        console.print(f"[green]Response:[/green] {response}")


def _run_finetune(cfg: DictConfig) -> None:
    """Run LLM fine-tuning with MLX LoRA."""
    import subprocess

    cmd = [
        "python",
        "-m",
        "mlx_lm.lora",
        "--train",
        "--model",
        cfg.model.pretrained,
        "--data",
        cfg.data.path,
        "--batch-size",
        str(cfg.training.batch_size),
        "--lora-layers",
        str(cfg.lora.layers),
        "--iters",
        str(cfg.training.max_steps or 600),
        "--learning-rate",
        str(cfg.training.learning_rate),
        "--adapter-path",
        f"models/adapters/{cfg.name}",
    ]
    console.print(f"[cyan]Running: {' '.join(cmd)}[/cyan]")
    subprocess.run(cmd, check=True)


def _run_evaluate(cfg: DictConfig) -> None:
    """Placeholder for LLM evaluation."""
    console.print("[yellow]LLM evaluation not yet implemented. Add your eval logic here.[/yellow]")
