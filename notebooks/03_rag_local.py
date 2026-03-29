"""Local RAG pipeline with ChromaDB + Ollama."""
# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "chromadb", "langchain", "langchain-ollama", "langchain-community"]
# ///

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def header():
    import marimo as mo

    mo.md(
        "# Local RAG Pipeline\n"
        "ChromaDB + Ollama — fully offline, no API keys needed.\n\n"
        "**Prerequisites**: `ollama pull nomic-embed-text && ollama pull llama3.2`"
    )
    return (mo,)


@app.cell
def setup_vectorstore():
    import marimo as mo

    from langchain_community.vectorstores import Chroma
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Example: create a simple collection
    texts = [
        "MLX is Apple's ML framework designed for Apple Silicon unified memory.",
        "PyTorch MPS backend enables GPU acceleration on Mac through Metal.",
        "LoRA (Low-Rank Adaptation) reduces fine-tuning memory by training small adapter matrices.",
        "Quantization reduces model size by using fewer bits per parameter (e.g., 4-bit, 8-bit).",
        "ChromaDB is an open-source embedding database for building AI applications.",
        "The M5 chip has Neural Accelerators in each GPU core for faster ML inference.",
    ]

    vectorstore = Chroma.from_texts(
        texts,
        embeddings,
        collection_name="latent-lab-demo",
    )

    mo.md(f"Indexed **{len(texts)}** documents into ChromaDB.")
    return (vectorstore,)


@app.cell
def query_rag(vectorstore):
    import marimo as mo

    from langchain_ollama import OllamaLLM

    query = mo.ui.text(
        value="How does MLX use Apple Silicon?",
        label="Question",
        full_width=True,
    )
    search_btn = mo.ui.run_button(label="Search & Answer")
    mo.md(f"{query}\n\n{search_btn}")

    if search_btn.value:
        # Retrieve relevant docs
        docs = vectorstore.similarity_search(query.value, k=3)
        context = "\n".join(doc.page_content for doc in docs)

        mo.md(f"### Retrieved Context\n```\n{context}\n```")

        # Generate answer
        llm = OllamaLLM(model="llama3.2")
        prompt = f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query.value}\n\nAnswer:"
        answer = llm.invoke(prompt)

        mo.md(f"### Answer\n{answer}")

    return


if __name__ == "__main__":
    app.run()
