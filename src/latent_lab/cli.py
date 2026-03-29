"""CLI entrypoint for latent-lab experiments."""

import typer

app = typer.Typer(name="lab", help="Latent Lab - ML/DL/LLM experimentation CLI")


@app.command()
def device_info():
    """Show Apple Silicon device info and ML framework status."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Device & Framework Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    # Python
    import sys

    table.add_row("Python", f"{sys.version.split()[0]}")

    # PyTorch MPS
    try:
        import torch

        mps = "Available" if torch.backends.mps.is_available() else "Not available"
        table.add_row("PyTorch", f"{torch.__version__} (MPS: {mps})")
    except ImportError:
        table.add_row("PyTorch", "[red]Not installed[/red]")

    # MLX
    try:
        import mlx.core as mx

        table.add_row("MLX", f"Device: {mx.default_device()}")
    except ImportError:
        table.add_row("MLX", "[red]Not installed[/red]")

    # MLX-LM
    try:
        import mlx_lm

        table.add_row("MLX-LM", "Installed")
    except ImportError:
        table.add_row("MLX-LM", "[yellow]Not installed[/yellow]")

    console.print(table)


@app.command()
def run(
    config: str = typer.Argument(..., help="Experiment config name (e.g., exp001_baseline)"),
    overrides: list[str] = typer.Option([], "--override", "-o", help="Hydra overrides"),
):
    """Run an experiment by config name."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "latent_lab.experiments.runner",
        f"--config-name={config}",
        *overrides,
    ]
    subprocess.run(cmd, check=True)


@app.command()
def serve(
    model_path: str = typer.Argument(..., help="Path to model or HF model ID"),
    port: int = typer.Option(8080, help="Server port"),
):
    """Serve an MLX model with OpenAI-compatible API."""
    import subprocess

    subprocess.run(
        ["python", "-m", "mlx_lm.server", "--model", model_path, "--port", str(port)],
        check=True,
    )


if __name__ == "__main__":
    app()
