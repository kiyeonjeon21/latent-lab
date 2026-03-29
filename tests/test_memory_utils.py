"""Tests for memory utilities."""

from latent_lab.utils.memory import estimate_model_memory


def test_estimate_4bit():
    # 7B model at 4-bit ≈ 3.5GB * 1.2 = 4.2GB
    est = estimate_model_memory(7, 4)
    assert 3.0 < est < 6.0


def test_estimate_16bit():
    # 7B model at 16-bit ≈ 14GB * 1.2 = 16.8GB
    est = estimate_model_memory(7, 16)
    assert 14.0 < est < 20.0


def test_estimate_scales_linearly():
    small = estimate_model_memory(3, 4)
    large = estimate_model_memory(6, 4)
    assert abs(large - 2 * small) < 0.01
