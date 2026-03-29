"""Named Entity Recognition."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Run NER pipeline."""
    from transformers import pipeline
    model_name = cfg.model.get("pretrained", "dslim/bert-base-NER")
    ner = pipeline("ner", model=model_name, device="mps", aggregation_strategy="simple")
    console.print(f"[green]Loaded NER: {model_name}[/green]")

    texts = cfg.get("texts", ["Apple was founded by Steve Jobs in Cupertino, California."])
    for text in texts:
        entities = ner(text)
        console.print(f"\n[bold]{text}[/bold]")
        for ent in entities:
            console.print(f"  {ent['entity_group']}: {ent['word']} ({ent['score']:.3f})")
