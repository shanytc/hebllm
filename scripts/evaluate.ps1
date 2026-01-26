# Evaluate HebLLM model
#
# Usage:
#   .\scripts\evaluate.ps1 [model_path] [test_data]
#
# Example:
#   .\scripts\evaluate.ps1 ./output/best_model ./test_data

param(
    [string]$ModelPath = "./output/best_model",
    [string]$TestData = "./test_data"
)

$ErrorActionPreference = "Stop"

$ModelType = if ($env:MODEL_TYPE) { $env:MODEL_TYPE } else { "florence2" }

Write-Host "========================================"
Write-Host "HebLLM Evaluation"
Write-Host "========================================"
Write-Host "Model:     $ModelPath"
Write-Host "Test data: $TestData"
Write-Host "Type:      $ModelType"
Write-Host "========================================"

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# Activate virtual environment if exists
$VenvActivate = Join-Path $ProjectDir ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
}

# Check paths
if (-not (Test-Path $ModelPath)) {
    Write-Host "Error: Model not found: $ModelPath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $TestData)) {
    Write-Host "Error: Test data not found: $TestData" -ForegroundColor Red
    exit 1
}

# Run evaluation
$EvalScript = @"
import sys
import json
from pathlib import Path

sys.path.insert(0, r'$ProjectDir')

from inference.pipeline import create_pipeline
from data.dataset import HebLLMDataset

# Load model
print('Loading model...')
pipeline = create_pipeline(r'$ModelPath', '$ModelType')

# Load test data
print('Loading test data...')
dataset = HebLLMDataset(r'$TestData', stage='direct_ocr')
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
output_path = Path(r'$ModelPath').parent / 'eval_results.json'
with open(output_path, 'w') as f:
    json.dump({'accuracy': correct/total, 'samples': results}, f, indent=2)
print(f'Results saved to: {output_path}')
"@

& python -c $EvalScript

Write-Host ""
Write-Host "Evaluation complete!"
