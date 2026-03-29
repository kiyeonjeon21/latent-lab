"""Memory monitoring for Apple Silicon unified memory."""

import os


def get_memory_info() -> dict:
    """Get current memory usage info."""
    import psutil

    vm = psutil.virtual_memory()
    return {
        "total_gb": vm.total / (1024**3),
        "available_gb": vm.available / (1024**3),
        "used_gb": vm.used / (1024**3),
        "percent": vm.percent,
    }


def estimate_model_memory(
    params_billions: float,
    quantization_bits: int = 16,
) -> float:
    """Estimate memory required for a model in GB.

    Args:
        params_billions: Number of parameters in billions.
        quantization_bits: Bits per parameter (4, 8, 16, 32).

    Returns:
        Estimated memory in GB (includes ~20% overhead for activations/KV cache).
    """
    bytes_per_param = quantization_bits / 8
    base_gb = params_billions * bytes_per_param
    return base_gb * 1.2  # 20% overhead


def fits_in_memory(params_billions: float, quantization_bits: int = 4) -> bool:
    """Check if a model will fit in available memory."""
    required = estimate_model_memory(params_billions, quantization_bits)
    available = get_memory_info()["available_gb"]
    return required < available * 0.85  # Leave 15% headroom


def set_mps_memory_limit():
    """Configure PyTorch MPS memory settings for stability."""
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
