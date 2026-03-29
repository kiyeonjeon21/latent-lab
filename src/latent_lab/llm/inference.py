"""LLM inference - model loading, generation, multi-model comparison."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
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


def compare_models(cfg: DictConfig) -> None:
    """Compare outputs across multiple models."""
    from latent_lab.models.mlx_utils import generate, load_mlx_model
    from latent_lab.experiments.tracker import log_metrics
    import time

    models_list = cfg.get("models", [cfg.model.pretrained])
    prompt = cfg.get("prompt", "Explain attention mechanism in transformers briefly.")
    max_tokens = cfg.get("max_tokens", 256)

    for model_id in models_list:
        console.print(f"\n[bold cyan]Model: {model_id}[/bold cyan]")
        model, tokenizer = load_mlx_model(model_id)

        start = time.perf_counter()
        response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
        elapsed = time.perf_counter() - start

        tokens = len(response.split())
        console.print(f"[green]Response ({tokens} tokens, {elapsed:.1f}s):[/green] {response[:200]}...")
        log_metrics({f"{model_id}/tokens": tokens, f"{model_id}/time_s": elapsed})
