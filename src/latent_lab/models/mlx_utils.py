"""MLX model utilities for Apple Silicon."""

from pathlib import Path


def load_mlx_model(model_path: str, tokenizer_path: str | None = None):
    """Load an MLX model for inference."""
    from mlx_lm import load

    return load(model_path, tokenizer_config=tokenizer_path)


def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate text with an MLX model."""
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature, top_p=top_p)
    return mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )


def quantize_model(
    hf_model_path: str,
    output_path: str | Path,
    q_bits: int = 4,
) -> Path:
    """Quantize a HuggingFace model to MLX format."""
    import subprocess

    output_path = Path(output_path)
    cmd = [
        "python",
        "-m",
        "mlx_lm.convert",
        "--hf-path",
        hf_model_path,
        "--mlx-path",
        str(output_path),
        "--quantize",
        "--q-bits",
        str(q_bits),
    ]
    subprocess.run(cmd, check=True)
    return output_path
