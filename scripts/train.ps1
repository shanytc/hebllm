# Train HebLLM model
#
# Usage:
#   .\scripts\train.ps1 [model] [data_dir] [output_dir]
#
# Example:
#   .\scripts\train.ps1 florence2 ./training_data ./output
#
# Resume training:
#   $env:RESUME = "output\hebllm_qwen2-vl-2b_...\checkpoints\model_epoch_2"
#   .\scripts\train.ps1 qwen2-vl-2b ./training_data ./output

param(
    [string]$Model = "florence2",
    [string]$DataDir = "./training_data",
    [string]$OutputDir = "./output"
)

$ErrorActionPreference = "Stop"

# Training parameters (can be overridden via environment)
$Epochs = if ($env:EPOCHS) { $env:EPOCHS } else { "30" }
$BatchSize = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { "4" }
$GradAccum = if ($env:GRAD_ACCUM) { $env:GRAD_ACCUM } else { "4" }
$LR = if ($env:LR) { $env:LR } else { "2e-4" }
$LoraRank = if ($env:LORA_RANK) { $env:LORA_RANK } else { "16" }

# Curriculum stages
$Stage1Epochs = if ($env:STAGE1_EPOCHS) { $env:STAGE1_EPOCHS } else { "5" }
$Stage2Epochs = if ($env:STAGE2_EPOCHS) { $env:STAGE2_EPOCHS } else { "10" }

# GPU settings (default: enabled)
$UseGpu = if ($env:USE_GPU) { $env:USE_GPU } else { "true" }

# Performance settings
$Compile = if ($env:COMPILE) { $env:COMPILE } else { "false" }
$Workers = if ($env:WORKERS) { $env:WORKERS } else { "0" }
$PinMemory = if ($env:PIN_MEMORY) { $env:PIN_MEMORY } else { "false" }

# Resume from checkpoint
$Resume = if ($env:RESUME) { $env:RESUME } else { "" }

Write-Host "========================================"
Write-Host "HebLLM Training"
Write-Host "========================================"
$EffectiveBatch = [int]$BatchSize * [int]$GradAccum
Write-Host "Model:       $Model"
Write-Host "Data:        $DataDir"
Write-Host "Output:      $OutputDir"
Write-Host "Epochs:      $Epochs"
Write-Host "Batch size:  $BatchSize"
Write-Host "Grad accum:  $GradAccum (effective batch: $EffectiveBatch)"
Write-Host "LR:          $LR"
Write-Host "LoRA rank:   $LoraRank"
Write-Host "Curriculum:  Stage1=$Stage1Epochs, Stage2=$Stage2Epochs"
Write-Host "GPU:         $UseGpu"
Write-Host "Compile:     $Compile"
Write-Host "Workers:     $Workers"
Write-Host "Pin memory:  $PinMemory"
if ($Resume) { Write-Host "Resume:      $Resume" }
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

# Build flags
$GpuFlag = if ($UseGpu -eq "true" -or $UseGpu -eq "1") { "--gpu" } else { "--no-gpu" }
$CompileFlag = if ($Compile -eq "true" -or $Compile -eq "1") { "--compile" } else { "" }
$PinMemoryFlag = if ($PinMemory -eq "true" -or $PinMemory -eq "1") { "--pin-memory" } else { "" }

# Build command arguments
$TrainArgs = @(
    "--model", $Model,
    "--train-data", $DataDir,
    "--output", $OutputDir,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--gradient-accumulation", $GradAccum,
    "--lr", $LR,
    "--lora-rank", $LoraRank,
    "--stage1-epochs", $Stage1Epochs,
    "--stage2-epochs", $Stage2Epochs,
    "--workers", $Workers,
    $GpuFlag
)

if ($CompileFlag) { $TrainArgs += $CompileFlag }
if ($PinMemoryFlag) { $TrainArgs += $PinMemoryFlag }
if ($Resume) { $TrainArgs += "--resume"; $TrainArgs += $Resume }

# Run training
& python "$ProjectDir\training\train.py" @TrainArgs

Write-Host ""
Write-Host "Training complete!"
Write-Host "Model saved to: $OutputDir"
