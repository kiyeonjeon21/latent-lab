"""RAG evaluation with RAGAS framework."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Evaluate RAG pipeline quality."""
    console.print("[yellow]RAG evaluation: install ragas and implement evaluation metrics.[/yellow]")
    console.print("[cyan]Metrics: faithfulness, answer_relevancy, context_precision, context_recall[/cyan]")
