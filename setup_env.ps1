# Setup virtual environment for HebLLM (Windows PowerShell)
#
# Usage:
#   .\setup_env.ps1          # Create and setup .venv
#   .\setup_env.ps1 -Clean   # Remove and recreate .venv
#
# After setup, activate with:
#   .\.venv\Scripts\Activate.ps1

param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvDir = ".venv"
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Write-Host "========================================"
Write-Host "HebLLM Environment Setup"
Write-Host "========================================"

# Check Python version
try {
    $PythonVersion = & $Python --version 2>&1
    Write-Host "Python: $Python ($PythonVersion)"
} catch {
    Write-Host "Error: Python not found. Please install Python 3.10+ and try again." -ForegroundColor Red
    exit 1
}

# Clean existing venv if requested
if ($Clean -and (Test-Path $VenvDir)) {
    Write-Host "Removing existing virtual environment..."
    Remove-Item -Recurse -Force $VenvDir
}

# Create virtual environment
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment in $VenvDir..."
    & $Python -m venv $VenvDir
} else {
    Write-Host "Virtual environment already exists."
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& "$VenvDir\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..."
& pip install --upgrade pip

# Check for NVIDIA GPU and install appropriate PyTorch
Write-Host "Checking for NVIDIA GPU..."
$nvidiaSmi = $null
$driverVersion = $null
try {
    $nvidiaSmi = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    $driverVersion = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null
} catch {}

if ($nvidiaSmi) {
    Write-Host "NVIDIA GPU detected: $nvidiaSmi" -ForegroundColor Green
    Write-Host "Driver version: $driverVersion" -ForegroundColor Green

    # Determine CUDA version based on driver
    # Driver 550+ -> CUDA 12.4, Driver 525+ -> CUDA 12.1, Driver 450+ -> CUDA 11.8
    $cudaVersion = "cu121"  # Default to CUDA 12.1 (most compatible)

    if ($driverVersion) {
        $majorVersion = [int]($driverVersion -split '\.')[0]
        if ($majorVersion -ge 550) {
            $cudaVersion = "cu124"
        } elseif ($majorVersion -ge 525) {
            $cudaVersion = "cu121"
        } elseif ($majorVersion -ge 450) {
            $cudaVersion = "cu118"
        }
    }

    Write-Host "Installing PyTorch with CUDA $cudaVersion..." -ForegroundColor Cyan
    & pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$cudaVersion"

    # Verify CUDA works
    Write-Host "Verifying CUDA installation..."
    $cudaCheck = & python -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cudaCheck -eq "True") {
        Write-Host "CUDA is working!" -ForegroundColor Green
    } else {
        Write-Host "CUDA verification failed. Trying CUDA 11.8 as fallback..." -ForegroundColor Yellow
        & pip uninstall torch torchvision torchaudio -y 2>$null
        & pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    }
} else {
    Write-Host "No NVIDIA GPU detected, installing CPU-only PyTorch..." -ForegroundColor Yellow
    & pip install torch torchvision torchaudio
}

# Install remaining dependencies
Write-Host "Installing dependencies..."
& pip install -r requirements.txt --ignore-installed torch

# Note about Flash Attention on Windows
Write-Host ""
Write-Host "Note: Flash Attention is not supported on Windows." -ForegroundColor Yellow
Write-Host "Training will use SDPA attention instead (similar performance)." -ForegroundColor Yellow

# Install package in development mode (if setup.py or pyproject.toml exists)
if ((Test-Path "setup.py") -or (Test-Path "pyproject.toml")) {
    Write-Host "Installing package in development mode..."
    & pip install -e .
}

Write-Host ""
Write-Host "========================================"
Write-Host "Setup complete!"
Write-Host "========================================"
Write-Host ""
Write-Host "To activate the environment, run:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "To deactivate, run:"
Write-Host "  deactivate"
Write-Host ""
