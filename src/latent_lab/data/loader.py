"""Unified data loading for various ML tasks."""

from pathlib import Path

import polars as pl


def load_csv(path: str | Path, **kwargs) -> pl.DataFrame:
    """Load CSV data with Polars."""
    return pl.read_csv(path, **kwargs)


def load_parquet(path: str | Path, **kwargs) -> pl.DataFrame:
    """Load Parquet data with Polars."""
    return pl.read_parquet(path, **kwargs)


def load_jsonl(path: str | Path) -> pl.DataFrame:
    """Load JSONL data (common format for LLM fine-tuning)."""
    return pl.read_ndjson(path)


def prepare_chat_data(
    df: pl.DataFrame,
    prompt_col: str = "prompt",
    completion_col: str = "completion",
    output_path: str | Path | None = None,
) -> pl.DataFrame:
    """Prepare data for LLM fine-tuning (prompt/completion format)."""
    result = df.select([
        pl.col(prompt_col).alias("prompt"),
        pl.col(completion_col).alias("completion"),
    ])
    if output_path:
        result.write_ndjson(output_path)
    return result
