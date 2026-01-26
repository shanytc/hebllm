#!/bin/bash
# Setup virtual environment for HebLLM
#
# Usage:
#   ./setup_env.sh          # Create and setup .venv
#   ./setup_env.sh --clean  # Remove and recreate .venv
#
# After setup, activate with:
#   source .venv/bin/activate

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYTHON=${PYTHON:-python3}

# Parse arguments
CLEAN=false
if [ "$1" == "--clean" ]; then
    CLEAN=true
fi

echo "========================================"
echo "HebLLM Environment Setup"
echo "========================================"

# Check Python version
PYTHON_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2)
echo "Python: $PYTHON ($PYTHON_VERSION)"

# Clean existing venv if requested
if [ "$CLEAN" == true ] && [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Detect OS and GPU
OS_TYPE="$(uname -s)"
echo "Operating system: $OS_TYPE"

# Check for NVIDIA GPU
HAS_NVIDIA=false
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
    if [ -n "$GPU_NAME" ]; then
        HAS_NVIDIA=true
        echo "NVIDIA GPU detected: $GPU_NAME"
    fi
fi

# Install PyTorch with CUDA on Linux if NVIDIA GPU is present
if [ "$OS_TYPE" = "Linux" ] && [ "$HAS_NVIDIA" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install Flash Attention on Linux with NVIDIA GPU
if [ "$OS_TYPE" = "Linux" ] && [ "$HAS_NVIDIA" = true ]; then
    echo "Installing Flash Attention 2 (Linux + NVIDIA GPU)..."
    pip install flash-attn --no-build-isolation || {
        echo "Warning: Flash Attention installation failed. This is optional."
        echo "Training will use SDPA attention instead."
    }
else
    if [ "$OS_TYPE" = "Darwin" ]; then
        echo "Skipping Flash Attention (not supported on macOS, will use MPS)"
    else
        echo "Skipping Flash Attention (requires Linux + NVIDIA GPU)"
    fi
fi

# Install package in development mode (if setup.py exists)
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Installing package in development mode..."
    pip install -e .
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
