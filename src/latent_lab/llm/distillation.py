"""LLM knowledge distillation - teacher-student response generation."""

from pathlib import Path

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def generate_training_data(cfg: DictConfig) -> None:
    """Generate training data from teacher model for student fine-tuning."""
    import json

    from latent_lab.models.mlx_utils import generate, load_mlx_model

    teacher_model_id = cfg.get("teacher_model", cfg.model.pretrained)
    output_path = Path(cfg.get("output_path", f"data/processed/distill_{cfg.name}"))
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Loading teacher: {teacher_model_id}[/cyan]")
    model, tokenizer = load_mlx_model(teacher_model_id)

    prompts = cfg.get("prompts", [])
    if not prompts:
        console.print("[red]No prompts provided in config. Set 'prompts' list.[/red]")
        return

    results = []
    for i, prompt in enumerate(prompts):
        console.print(f"  [{i+1}/{len(prompts)}] Generating response...")
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=cfg.get("max_tokens", 512),
            temperature=cfg.get("temperature", 0.7),
        )
        results.append({"prompt": prompt, "completion": response})

    train_path = output_path / "train.jsonl"
    with open(train_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    console.print(f"[green]Generated {len(results)} examples → {train_path}[/green]")
    console.print(f"[cyan]Next: fine-tune student with data.path={output_path}[/cyan]")


def run(cfg: DictConfig) -> None:
    """Full distillation pipeline: generate data + fine-tune student."""
    generate_training_data(cfg)
