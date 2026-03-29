"""NLP experiment domain - classification, NER, embeddings."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run_experiment(cfg: DictConfig) -> None:
    """Run an NLP experiment."""
    from latent_lab.experiments.tracker import log_config, setup_tracking, track_run

    setup_tracking(f"nlp-{cfg.name}")

    with track_run(run_name=cfg.name, tags={"domain": "nlp"}):
        log_config(cfg)

        task = cfg.get("task", "classify")

        match task:
            case "classify":
                _run_classification(cfg)
            case "ner":
                _run_ner(cfg)
            case _:
                console.print(f"[red]Unknown NLP task: {task}[/red]")


def _run_classification(cfg: DictConfig) -> None:
    """Text classification with HuggingFace Transformers."""
    from transformers import pipeline

    device = "mps"
    model_name = cfg.model.get("pretrained", "distilbert-base-uncased-finetuned-sst-2-english")
    classifier = pipeline("text-classification", model=model_name, device=device)
    console.print(f"[green]Loaded classifier: {model_name} on {device}[/green]")
    console.print("[yellow]Add your evaluation/training loop here.[/yellow]")


def _run_ner(cfg: DictConfig) -> None:
    """Named Entity Recognition."""
    from transformers import pipeline

    model_name = cfg.model.get("pretrained", "dslim/bert-base-NER")
    ner = pipeline("ner", model=model_name, device="mps")
    console.print(f"[green]Loaded NER model: {model_name}[/green]")
    console.print("[yellow]Add your NER evaluation loop here.[/yellow]")
