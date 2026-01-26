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

# Install dependencies
Write-Host "Installing dependencies..."
& pip install -r requirements.txt

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
