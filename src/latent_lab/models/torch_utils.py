"""PyTorch utilities optimized for Apple Silicon MPS."""

import torch


def get_device() -> torch.device:
    """Get the best available device (MPS > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_summary(model: torch.nn.Module) -> dict:
    """Get model parameter summary."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "total_mb": total * 4 / (1024 * 1024),  # assuming float32
    }


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
