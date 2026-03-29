"""Download and prepare commonly used models for experiments."""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()

# Recommended models for 32GB MacBook Air M5
RECOMMENDED_MODELS = {
    "llm": {
        "mistral-7b-4bit": {
            "hf_id": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "memory_gb": 4.5,
            "description": "General purpose, good quality/speed balance",
        },
        "llama-3.2-3b": {
            "hf_id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "memory_gb": 2.5,
            "description": "Fast, lightweight, good for development",
        },
        "qwen-14b-4bit": {
            "hf_id": "mlx-community/Qwen2.5-14B-Instruct-4bit",
            "memory_gb": 9.0,
            "description": "High quality, multilingual (CJK excellent)",
        },
        "qwen-30b-moe-4bit": {
            "hf_id": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
            "memory_gb": 17.0,
            "description": "Coding specialist, needs ~17GB",
        },
    },
    "embedding": {
        "nomic-embed": {
            "hf_id": "nomic-ai/nomic-embed-text-v1.5",
            "memory_gb": 0.5,
            "description": "768-dim embeddings, great for RAG",
        },
    },
}


@app.command()
def list_models():
    """List recommended models for this hardware."""
    for category, models in RECOMMENDED_MODELS.items():
        table = Table(title=f"{category.upper()} Models")
        table.add_column("Name", style="cyan")
        table.add_column("Memory", style="green")
        table.add_column("HF ID", style="dim")
        table.add_column("Description")

        for name, info in models.items():
            table.add_row(
                name,
                f"{info['memory_gb']:.1f} GB",
                info["hf_id"],
                info["description"],
            )
        console.print(table)
        console.print()


@app.command()
def download(
    model_name: str = typer.Argument(..., help="Model name from list-models"),
    output_dir: str = typer.Option("models/weights", help="Output directory"),
):
    """Download a model from HuggingFace."""
    # Find the model
    hf_id = None
    for _category, models in RECOMMENDED_MODELS.items():
        if model_name in models:
            hf_id = models[model_name]["hf_id"]
            break

    if not hf_id:
        console.print(f"[red]Model '{model_name}' not found. Run 'list-models' to see options.[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Downloading {hf_id}...[/cyan]")

    from huggingface_hub import snapshot_download

    path = snapshot_download(hf_id, local_dir=f"{output_dir}/{model_name}")
    console.print(f"[green]Downloaded to: {path}[/green]")


if __name__ == "__main__":
    app()
