#!/bin/bash
# Train HebLLM model
#
# Usage:
#   ./scripts/train.sh [model] [data_dir] [output_dir]
#
# Example:
#   ./scripts/train.sh florence2 ./training_data ./output
#
# Resume training:
#   RESUME="output/hebllm_qwen2-vl-2b_.../checkpoints/model_epoch_2" ./scripts/train.sh qwen2-vl-2b

set -e

# Default values
MODEL=${1:-"florence2"}
DATA_DIR=${2:-"./training_data"}
OUTPUT_DIR=${3:-"./output"}

# Training parameters (can be overridden via environment)
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-2e-4}
LORA_RANK=${LORA_RANK:-16}

# Curriculum stages
STAGE1_EPOCHS=${STAGE1_EPOCHS:-5}
STAGE2_EPOCHS=${STAGE2_EPOCHS:-10}

# GPU settings (default: enabled)
USE_GPU=${USE_GPU:-true}

# Performance settings
COMPILE=${COMPILE:-false}
WORKERS=${WORKERS:-0}
PIN_MEMORY=${PIN_MEMORY:-false}

# Resume from checkpoint
RESUME=${RESUME:-""}

echo "========================================"
echo "HebLLM Training"
echo "========================================"
echo "Model:       $MODEL"
echo "Data:        $DATA_DIR"
echo "Output:      $OUTPUT_DIR"
echo "Epochs:      $EPOCHS"
echo "Batch size:  $BATCH_SIZE"
echo "Grad accum:  $GRAD_ACCUM (effective batch: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "LR:          $LR"
echo "LoRA rank:   $LORA_RANK"
echo "Curriculum:  Stage1=$STAGE1_EPOCHS, Stage2=$STAGE2_EPOCHS"
echo "GPU:         $USE_GPU"
echo "Compile:     $COMPILE"
echo "Workers:     $WORKERS"
echo "Pin memory:  $PIN_MEMORY"
if [ -n "$RESUME" ]; then echo "Resume:      $RESUME"; fi
echo "========================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if exists
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Run ./scripts/generate_data.sh first"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build flags
GPU_FLAG=""
if [ "$USE_GPU" = "true" ] || [ "$USE_GPU" = "1" ]; then
    GPU_FLAG="--gpu"
else
    GPU_FLAG="--no-gpu"
fi

COMPILE_FLAG=""
if [ "$COMPILE" = "true" ] || [ "$COMPILE" = "1" ]; then
    COMPILE_FLAG="--compile"
fi

PIN_MEMORY_FLAG=""
if [ "$PIN_MEMORY" = "true" ] || [ "$PIN_MEMORY" = "1" ]; then
    PIN_MEMORY_FLAG="--pin-memory"
fi

RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume $RESUME"
fi

# Run training
python "$PROJECT_DIR/training/train.py" \
    --model "$MODEL" \
    --train-data "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation "$GRAD_ACCUM" \
    --lr "$LR" \
    --lora-rank "$LORA_RANK" \
    --stage1-epochs "$STAGE1_EPOCHS" \
    --stage2-epochs "$STAGE2_EPOCHS" \
    --workers "$WORKERS" \
    $GPU_FLAG $COMPILE_FLAG $PIN_MEMORY_FLAG $RESUME_FLAG

echo ""
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
