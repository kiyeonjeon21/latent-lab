"""Text classification with HuggingFace Transformers."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Run text classification."""
    from transformers import pipeline
    device = "mps"
    model_name = cfg.model.get("pretrained", "distilbert-base-uncased-finetuned-sst-2-english")
    classifier = pipeline("text-classification", model=model_name, device=device)
    console.print(f"[green]Loaded classifier: {model_name} on {device}[/green]")

    texts = cfg.get("texts", [])
    if texts:
        results = classifier(list(texts))
        for text, result in zip(texts, results):
            console.print(f"  {text[:60]}... → {result['label']} ({result['score']:.3f})")
    else:
        console.print("[yellow]No texts provided. Add 'texts' list to config.[/yellow]")
