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

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

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
