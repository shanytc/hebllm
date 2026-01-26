#!/bin/bash
# Train HebLLM model
#
# Usage:
#   ./scripts/train.sh [model] [data_dir] [output_dir]
#
# Example:
#   ./scripts/train.sh florence2 ./training_data ./output

set -e

# Default values
MODEL=${1:-"florence2"}
DATA_DIR=${2:-"./training_data"}
OUTPUT_DIR=${3:-"./output"}

# Training parameters (can be overridden via environment)
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-2e-4}
LORA_RANK=${LORA_RANK:-16}

# Curriculum stages
STAGE1_EPOCHS=${STAGE1_EPOCHS:-5}
STAGE2_EPOCHS=${STAGE2_EPOCHS:-10}

# GPU settings (default: enabled)
USE_GPU=${USE_GPU:-true}

echo "========================================"
echo "HebLLM Training"
echo "========================================"
echo "Model:       $MODEL"
echo "Data:        $DATA_DIR"
echo "Output:      $OUTPUT_DIR"
echo "Epochs:      $EPOCHS"
echo "Batch size:  $BATCH_SIZE"
echo "LR:          $LR"
echo "LoRA rank:   $LORA_RANK"
echo "Curriculum:  Stage1=$STAGE1_EPOCHS, Stage2=$STAGE2_EPOCHS"
echo "GPU:         $USE_GPU"
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

# Build GPU flag
GPU_FLAG=""
if [ "$USE_GPU" = "true" ] || [ "$USE_GPU" = "1" ]; then
    GPU_FLAG="--gpu"
else
    GPU_FLAG="--no-gpu"
fi

# Run training
python "$PROJECT_DIR/training/train.py" \
    --model "$MODEL" \
    --train-data "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --lora-rank "$LORA_RANK" \
    --stage1-epochs "$STAGE1_EPOCHS" \
    --stage2-epochs "$STAGE2_EPOCHS" \
    $GPU_FLAG

echo ""
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
