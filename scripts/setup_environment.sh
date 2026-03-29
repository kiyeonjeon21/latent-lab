#!/bin/bash
# Setup script for Latent Lab on Apple Silicon
set -euo pipefail

echo "=== Latent Lab Environment Setup ==="

# Check Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "WARNING: This setup is optimized for Apple Silicon (arm64)"
fi

# Check uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "Please restart your shell and re-run this script."
    exit 0
fi

echo "uv version: $(uv --version)"

# Create virtual environment and install
echo ""
echo "=== Installing dependencies ==="
uv sync

echo ""
echo "=== Installing optional dependencies ==="
read -p "Install all optional deps? (llm, cv, rl, serving, notebooks, dev) [y/N]: " install_all
if [[ "$install_all" =~ ^[Yy]$ ]]; then
    uv sync --all-extras
else
    echo "Available extras: llm, cv, rl, serving, notebooks, dev"
    read -p "Enter extras to install (comma-separated, or skip): " extras
    if [[ -n "$extras" ]]; then
        IFS=',' read -ra EXTRA_ARRAY <<< "$extras"
        for extra in "${EXTRA_ARRAY[@]}"; do
            extra=$(echo "$extra" | xargs)  # trim whitespace
            echo "Installing [$extra]..."
            uv sync --extra "$extra"
        done
    fi
fi

# Install pre-commit hooks
if command -v pre-commit &> /dev/null || uv run pre-commit --version &> /dev/null 2>&1; then
    echo ""
    echo "=== Setting up pre-commit hooks ==="
    uv run pre-commit install
fi

# Verify setup
echo ""
echo "=== Verifying ML setup ==="
uv run python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    mps = torch.backends.mps.is_available()
    print(f'PyTorch: {torch.__version__} (MPS: {mps})')
except ImportError:
    print('PyTorch: not installed')

try:
    import mlx.core as mx
    print(f'MLX: device={mx.default_device()}')
except ImportError:
    print('MLX: not installed')

try:
    import polars
    print(f'Polars: {polars.__version__}')
except ImportError:
    print('Polars: not installed')
"

echo ""
echo "=== Setup complete! ==="
echo "Run 'make help' to see available commands."
