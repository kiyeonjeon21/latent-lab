"""Apple Silicon ML environment check."""
# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "torch", "mlx"]
# ///

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def header():
    import marimo as mo

    mo.md("# Apple Silicon ML Environment Check")
    return (mo,)


@app.cell
def system_info():
    import platform
    import sys

    import marimo as mo

    info = {
        "Python": sys.version.split()[0],
        "Platform": platform.platform(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
    }
    mo.md(
        "## System Info\n"
        + "\n".join(f"- **{k}**: {v}" for k, v in info.items())
    )
    return info


@app.cell
def pytorch_check():
    import marimo as mo

    try:
        import torch

        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()

        mo.md(
            f"## PyTorch\n"
            f"- Version: **{torch.__version__}**\n"
            f"- MPS built: **{mps_built}**\n"
            f"- MPS available: **{mps_available}**\n"
        )
    except ImportError:
        mo.md("## PyTorch\n- **Not installed**")
    return


@app.cell
def mlx_check():
    import marimo as mo

    try:
        import mlx.core as mx

        mo.md(
            f"## MLX\n"
            f"- Default device: **{mx.default_device()}**\n"
            f"- Available devices: GPU, CPU\n"
        )

        # Quick benchmark
        import time

        a = mx.random.normal((1000, 1000))
        b = mx.random.normal((1000, 1000))
        mx.eval(a)
        mx.eval(b)

        start = time.perf_counter()
        for _ in range(100):
            c = a @ b
            mx.eval(c)
        elapsed = time.perf_counter() - start

        mo.md(f"- **Matmul benchmark**: 100x (1000x1000) in {elapsed:.2f}s")
    except ImportError:
        mo.md("## MLX\n- **Not installed**")
    return


@app.cell
def memory_check():
    import marimo as mo

    from latent_lab.utils.memory import estimate_model_memory, get_memory_info

    mem = get_memory_info()

    models = [
        ("Llama 3.2 3B", 3, 4),
        ("Mistral 7B", 7, 4),
        ("Llama 3.1 8B", 8, 4),
        ("Qwen 14B", 14, 4),
        ("Qwen 30B MoE", 30, 4),
    ]

    rows = []
    for name, params, bits in models:
        est = estimate_model_memory(params, bits)
        fits = "Yes" if est < mem["available_gb"] * 0.85 else "No"
        rows.append(f"| {name} | {params}B | {bits}-bit | {est:.1f} GB | {fits} |")

    mo.md(
        f"## Memory\n"
        f"- Total: **{mem['total_gb']:.1f} GB**\n"
        f"- Available: **{mem['available_gb']:.1f} GB**\n"
        f"- Used: **{mem['used_gb']:.1f} GB** ({mem['percent']}%)\n\n"
        f"### Model Fit Estimates\n"
        f"| Model | Params | Quant | Est. Memory | Fits? |\n"
        f"|-------|--------|-------|-------------|-------|\n"
        + "\n".join(rows)
    )
    return


if __name__ == "__main__":
    app.run()
