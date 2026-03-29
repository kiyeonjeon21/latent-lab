"""Tokenizer training - BPE, Korean tokenizer."""

from pathlib import Path

from omegaconf import DictConfig
from rich.console import Console

console = Console()


def run(cfg: DictConfig) -> None:
    """Train a BPE tokenizer from corpus."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    corpus_path = cfg.get("corpus_path", "data/raw/corpus.txt")
    vocab_size = cfg.get("vocab_size", 32000)
    output_path = Path(cfg.get("output_path", f"models/weights/tokenizer-{cfg.name}.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

    console.print(f"[cyan]Training BPE tokenizer (vocab={vocab_size}) on {corpus_path}[/cyan]")
    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(str(output_path))
    console.print(f"[green]Tokenizer saved to {output_path}[/green]")
