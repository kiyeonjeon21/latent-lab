"""Document chunking strategies for RAG."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Compare different chunking strategies."""
    from langchain.text_splitter import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    text = cfg.get("text", "")
    if not text:
        console.print("[yellow]Provide 'text' in config or load from file.[/yellow]")
        return

    strategies = {
        "character": CharacterTextSplitter(chunk_size=200, chunk_overlap=20),
        "recursive": RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20),
    }

    for name, splitter in strategies.items():
        chunks = splitter.split_text(text)
        console.print(f"[bold]{name}:[/bold] {len(chunks)} chunks, avg {sum(len(c) for c in chunks) // len(chunks)} chars")
