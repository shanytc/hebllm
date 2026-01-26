# Generate synthetic training data for HebLLM
#
# Usage:
#   .\scripts\generate_data.ps1 [num_samples] [output_dir]
#
# Example:
#   .\scripts\generate_data.ps1 1000 ./training_data

param(
    [int]$NumSamples = 100,
    [string]$OutputDir = "./training_data"
)

$ErrorActionPreference = "Stop"

# Language distribution (can be overridden via environment)
$HebrewPct = if ($env:HEBREW_PCT) { $env:HEBREW_PCT } else { "0.4" }
$EnglishPct = if ($env:ENGLISH_PCT) { $env:ENGLISH_PCT } else { "0.4" }
$MixedPct = if ($env:MIXED_PCT) { $env:MIXED_PCT } else { "0.2" }

Write-Host "========================================"
Write-Host "HebLLM Synthetic Data Generator"
Write-Host "========================================"
Write-Host "Samples: $NumSamples"
Write-Host "Output:  $OutputDir"
Write-Host "Distribution: Hebrew=$HebrewPct, English=$EnglishPct, Mixed=$MixedPct"
Write-Host "========================================"

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# Activate virtual environment if exists
$VenvActivate = Join-Path $ProjectDir ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
}

# Run generator
& python "$ProjectDir\data\generator.py" `
    --output $OutputDir `
    --num-samples $NumSamples `
    --hebrew-pct $HebrewPct `
    --english-pct $EnglishPct `
    --mixed-pct $MixedPct

Write-Host ""
Write-Host "Data generation complete!"
Write-Host "Output directory: $OutputDir"
