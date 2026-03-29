"""Retrieval strategies - vector search, hybrid search, reranking."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Run retrieval experiment."""
    console.print("[yellow]Hybrid retrieval (BM25 + vector): implement with rank_bm25 + ChromaDB.[/yellow]")
