"""RAG pipeline - embed, retrieve, generate."""

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Run RAG pipeline."""
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import OllamaEmbeddings, OllamaLLM

    embedding_model = cfg.get("embedding_model", "nomic-embed-text")
    llm_model = cfg.get("llm_model", "llama3.2:1b")

    embeddings = OllamaEmbeddings(model=embedding_model)

    texts = cfg.get("documents", [
        "MLX is Apple's ML framework for unified memory.",
        "PyTorch MPS enables GPU acceleration on Mac.",
        "LoRA reduces fine-tuning memory requirements.",
    ])

    console.print(f"[cyan]Indexing {len(texts)} documents...[/cyan]")
    vectorstore = Chroma.from_texts(texts, embeddings, collection_name=cfg.get("collection", "rag-exp"))

    query = cfg.get("query", "How does MLX work?")
    docs = vectorstore.similarity_search(query, k=cfg.get("top_k", 3))
    context = "\n".join(doc.page_content for doc in docs)

    console.print(f"[cyan]Generating answer with {llm_model}...[/cyan]")
    llm = OllamaLLM(model=llm_model)
    prompt = f"Based on this context, answer briefly.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    answer = llm.invoke(prompt)
    console.print(f"[green]Answer: {answer}[/green]")
