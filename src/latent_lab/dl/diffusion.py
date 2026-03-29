"""Diffusion model inference - Stable Diffusion on MPS."""

import torch
from omegaconf import DictConfig
from rich.console import Console

from latent_lab.models.torch_utils import get_device

console = Console()


def run(cfg: DictConfig) -> None:
    """Run Stable Diffusion inference on MPS."""
    from diffusers import StableDiffusionPipeline

    device = get_device()
    model_id = cfg.model.get("pretrained", "stabilityai/stable-diffusion-2-1-base")
    console.print(f"[cyan]Loading {model_id}...[/cyan]")

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    prompt = cfg.get("prompt", "a photo of a cat in space, digital art")
    num_steps = cfg.get("num_inference_steps", 30)

    console.print(f"[cyan]Generating: '{prompt}'[/cyan]")
    image = pipe(prompt, num_inference_steps=num_steps).images[0]

    output_path = f"reports/figures/{cfg.name}.png"
    image.save(output_path)
    console.print(f"[green]Image saved to {output_path}[/green]")
