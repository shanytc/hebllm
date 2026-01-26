#!/bin/bash
# Evaluate HebLLM model
#
# Usage:
#   ./scripts/evaluate.sh [model_path] [test_data]
#
# Example:
#   ./scripts/evaluate.sh ./output/best_model ./test_data

set -e

MODEL_PATH=${1:-"./output/best_model"}
TEST_DATA=${2:-"./test_data"}
MODEL_TYPE=${MODEL_TYPE:-"florence2"}

echo "========================================"
echo "HebLLM Evaluation"
echo "========================================"
echo "Model:     $MODEL_PATH"
echo "Test data: $TEST_DATA"
echo "Type:      $MODEL_TYPE"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if exists
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Check paths
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$TEST_DATA" ]; then
    echo "Error: Test data not found: $TEST_DATA"
    exit 1
fi

# Run evaluation
python -c "
import sys
import json
from pathlib import Path

sys.path.insert(0, '$PROJECT_DIR')

from inference.pipeline import create_pipeline
from data.dataset import HebLLMDataset

# Load model
print('Loading model...')
pipeline = create_pipeline('$MODEL_PATH', '$MODEL_TYPE')

# Load test data
print('Loading test data...')
dataset = HebLLMDataset('$TEST_DATA', stage='direct_ocr')
print(f'Test samples: {len(dataset)}')

# Evaluate
correct = 0
total = 0
results = []

for i in range(min(len(dataset), 100)):
    sample = dataset[i]

    # Get prediction
    from PIL import Image
    import numpy as np
    img_array = (sample['image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image = Image.fromarray(img_array)

    prediction = pipeline(image)
    target = sample['target']

    # Simple accuracy: exact match
    if prediction.strip() == target.strip():
        correct += 1
    total += 1

    results.append({
        'index': i,
        'prediction': prediction[:100],
        'target': target[:100],
        'match': prediction.strip() == target.strip()
    })

    if (i + 1) % 10 == 0:
        print(f'  Processed {i + 1}/{min(len(dataset), 100)}...')

# Print results
print()
print('='*40)
print('Results')
print('='*40)
print(f'Exact Match Accuracy: {correct}/{total} ({100*correct/total:.1f}%)')

# Save results
output_path = Path('$MODEL_PATH').parent / 'eval_results.json'
with open(output_path, 'w') as f:
    json.dump({'accuracy': correct/total, 'samples': results}, f, indent=2)
print(f'Results saved to: {output_path}')
"

echo ""
echo "Evaluation complete!"
