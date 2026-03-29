"""LLM evaluation - benchmarks and custom metrics."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Run LLM evaluation."""
    eval_type = cfg.get("eval_type", "custom")
    match eval_type:
        case "custom":
            run_custom_eval(cfg)
        case "perplexity":
            run_perplexity(cfg)
        case _:
            console.print(f"[red]Unknown eval type: {eval_type}[/red]")


def run_custom_eval(cfg: DictConfig) -> None:
    """Evaluate model on custom prompt-expected pairs."""
    from latent_lab.experiments.tracker import log_metrics
    from latent_lab.models.mlx_utils import generate, load_mlx_model

    console.print(f"[cyan]Loading model: {cfg.model.pretrained}[/cyan]")
    model, tokenizer = load_mlx_model(cfg.model.pretrained)

    test_cases = cfg.get("test_cases", [])
    if not test_cases:
        console.print("[yellow]No test_cases in config. Provide list of {prompt, expected}.[/yellow]")
        return

    correct = 0
    for i, tc in enumerate(test_cases):
        response = generate(model, tokenizer, prompt=tc["prompt"], max_tokens=64)
        match = tc.get("expected", "").lower() in response.lower()
        correct += int(match)
        status = "[green]PASS[/green]" if match else "[red]FAIL[/red]"
        console.print(f"  [{i+1}] {status} | Expected: {tc.get('expected', 'N/A')}")

    accuracy = correct / len(test_cases) if test_cases else 0
    log_metrics({"eval_accuracy": accuracy, "eval_total": len(test_cases)})
    console.print(f"\n[bold]Accuracy: {accuracy:.1%} ({correct}/{len(test_cases)})[/bold]")


def run_perplexity(cfg: DictConfig) -> None:
    """Compute perplexity on a text file."""
    import subprocess

    model_path = cfg.model.pretrained
    data_path = cfg.get("eval_data", "data/processed/eval.txt")

    cmd = [
        "python", "-m", "mlx_lm.evaluate",
        "--model", model_path,
        "--data", data_path,
    ]
    console.print(f"[cyan]Computing perplexity...[/cyan]")
    subprocess.run(cmd, check=True)
