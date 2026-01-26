# Train HebLLM model
#
# Usage:
#   .\scripts\train.ps1 [model] [data_dir] [output_dir]
#
# Example:
#   .\scripts\train.ps1 florence2 ./training_data ./output

param(
    [string]$Model = "florence2",
    [string]$DataDir = "./training_data",
    [string]$OutputDir = "./output"
)

$ErrorActionPreference = "Stop"

# Training parameters (can be overridden via environment)
$Epochs = if ($env:EPOCHS) { $env:EPOCHS } else { "30" }
$BatchSize = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { "4" }
$LR = if ($env:LR) { $env:LR } else { "2e-4" }
$LoraRank = if ($env:LORA_RANK) { $env:LORA_RANK } else { "16" }

# Curriculum stages
$Stage1Epochs = if ($env:STAGE1_EPOCHS) { $env:STAGE1_EPOCHS } else { "5" }
$Stage2Epochs = if ($env:STAGE2_EPOCHS) { $env:STAGE2_EPOCHS } else { "10" }

# GPU settings (default: enabled)
$UseGpu = if ($env:USE_GPU) { $env:USE_GPU } else { "true" }

Write-Host "========================================"
Write-Host "HebLLM Training"
Write-Host "========================================"
Write-Host "Model:       $Model"
Write-Host "Data:        $DataDir"
Write-Host "Output:      $OutputDir"
Write-Host "Epochs:      $Epochs"
Write-Host "Batch size:  $BatchSize"
Write-Host "LR:          $LR"
Write-Host "LoRA rank:   $LoraRank"
Write-Host "Curriculum:  Stage1=$Stage1Epochs, Stage2=$Stage2Epochs"
Write-Host "GPU:         $UseGpu"
Write-Host "========================================"

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# Activate virtual environment if exists
$VenvActivate = Join-Path $ProjectDir ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
}

# Check if data exists
if (-not (Test-Path $DataDir)) {
    Write-Host "Error: Data directory not found: $DataDir" -ForegroundColor Red
    Write-Host "Run .\scripts\generate_data.ps1 first"
    exit 1
}

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Build GPU flag
$GpuFlag = if ($UseGpu -eq "true" -or $UseGpu -eq "1") { "--gpu" } else { "--no-gpu" }

# Run training
& python "$ProjectDir\training\train.py" `
    --model $Model `
    --train-data $DataDir `
    --output $OutputDir `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --lr $LR `
    --lora-rank $LoraRank `
    --stage1-epochs $Stage1Epochs `
    --stage2-epochs $Stage2Epochs `
    $GpuFlag

Write-Host ""
Write-Host "Training complete!"
Write-Host "Model saved to: $OutputDir"
