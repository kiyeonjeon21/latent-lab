"""Text embeddings and semantic search."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Compute embeddings and run similarity search."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model_name = cfg.get("embedding_model", "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)
    console.print(f"[green]Loaded embedding model: {model_name}[/green]")

    corpus = cfg.get("corpus", ["Machine learning is great.", "Deep learning uses neural networks.", "I love pizza."])
    query = cfg.get("query", "What is AI?")

    corpus_embeddings = model.encode(corpus)
    query_embedding = model.encode([query])

    similarities = np.dot(corpus_embeddings, query_embedding.T).flatten()
    ranked = np.argsort(similarities)[::-1]

    console.print(f"\n[bold]Query:[/bold] {query}")
    for i, idx in enumerate(ranked):
        console.print(f"  {i+1}. [{similarities[idx]:.3f}] {corpus[idx]}")
