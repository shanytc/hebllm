#!/bin/bash
# Generate synthetic training data for HebLLM
#
# Usage:
#   ./scripts/generate_data.sh [num_samples] [output_dir]
#
# Example:
#   ./scripts/generate_data.sh 1000 ./training_data

set -e

# Default values
NUM_SAMPLES=${1:-100}
OUTPUT_DIR=${2:-"./training_data"}

# Language distribution
HEBREW_PCT=${HEBREW_PCT:-0.4}
ENGLISH_PCT=${ENGLISH_PCT:-0.4}
MIXED_PCT=${MIXED_PCT:-0.2}

echo "========================================"
echo "HebLLM Synthetic Data Generator"
echo "========================================"
echo "Samples: $NUM_SAMPLES"
echo "Output:  $OUTPUT_DIR"
echo "Distribution: Hebrew=$HEBREW_PCT, English=$ENGLISH_PCT, Mixed=$MIXED_PCT"
echo "========================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if exists
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Run generator
python "$PROJECT_DIR/data/generator.py" \
    --output "$OUTPUT_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --hebrew-pct "$HEBREW_PCT" \
    --english-pct "$ENGLISH_PCT" \
    --mixed-pct "$MIXED_PCT"

echo ""
echo "Data generation complete!"
echo "Output directory: $OUTPUT_DIR"
