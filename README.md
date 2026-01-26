# HebLLM: Hebrew-English Document OCR

A vision-language model for OCR on Hebrew, English, and mixed documents. Uses the HebMark system during training to help the model learn Hebrew character patterns.

## Features

- **Edge-deployable**: Fine-tuned Florence-2 (0.23B) or Qwen2-VL (2B)
- **Multilingual**: Hebrew, English, and mixed content
- **Curriculum learning**: 3-stage training with HebMark markers
- **PDF support**: Direct OCR from PDF files

---

## Quick Start

**Linux/macOS:**
```bash
# 1. Setup
./setup_env.sh

# 2. Generate data
./scripts/generate_data.sh 1000 ./training_data

# 3. Train
./scripts/train.sh florence2 ./training_data ./output

# 4. Inference
source .venv/bin/activate
python inference/pipeline.py document.pdf --model ./output/best_model
```

**Windows (PowerShell):**
```powershell
# 1. Setup
.\setup_env.ps1

# 2. Generate data
python data/generator.py --output ./training_data --num-samples 1000

# 3. Train
python training/train.py --model florence2 --train-data ./training_data --output ./output

# 4. Inference
.\.venv\Scripts\Activate.ps1
python inference/pipeline.py document.pdf --model ./output/best_model
```

---

## Installation

### Prerequisites

- Python 3.10-3.13
- CUDA (optional, for GPU training)
- Note: GPU (CUDA/MPS) recommended for training; CPU works but is slow

### Setup Virtual Environment

**Linux/macOS:**
```bash
# Automatic setup
./setup_env.sh

# Or manual setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
# Automatic setup
.\setup_env.ps1

# Or manual setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Clean reinstall (remove and recreate .venv):**
```bash
./setup_env.sh --clean    # Linux/macOS
.\setup_env.ps1 -Clean    # Windows
```

### Verify Installation

```bash
# Linux/macOS
source .venv/bin/activate

# Windows: .\.venv\Scripts\Activate.ps1

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## Usage

### 1. Generate Training Data

Generate synthetic PDF pages with Hebrew/English text:

**Linux/macOS:**
```bash
# Basic: 100 samples
./scripts/generate_data.sh

# Custom: 1000 samples to specific directory
./scripts/generate_data.sh 1000 ./training_data

# Custom distribution (environment variables)
HEBREW_PCT=0.5 ENGLISH_PCT=0.3 MIXED_PCT=0.2 \
  ./scripts/generate_data.sh 5000 ./training_data
```

**Windows (PowerShell):**
```powershell
# Basic: 100 samples
.\scripts\generate_data.ps1

# Custom: 1000 samples to specific directory
.\scripts\generate_data.ps1 -NumSamples 1000 -OutputDir ./training_data

# Custom distribution (environment variables)
$env:HEBREW_PCT="0.5"; $env:ENGLISH_PCT="0.3"; $env:MIXED_PCT="0.2"
.\scripts\generate_data.ps1 -NumSamples 5000 -OutputDir ./training_data
```

**Output structure:**
```
training_data/
├── images/              # Original PDF page images
├── marked_images/       # Images with HebMark markers
├── mappings/            # Marker → Hebrew text mappings
├── metadata/            # Sample metadata (ground truth, etc.)
└── dataset_metadata.json
```

### 2. Train Model

Train with curriculum learning (3 stages):

**Linux/macOS:**
```bash
# Default: Florence-2 base model
./scripts/train.sh florence2 ./training_data ./output

# Qwen2-VL (better Hebrew, larger model)
./scripts/train.sh qwen2-vl-2b ./training_data ./output
```

**Windows (PowerShell):**
```powershell
# Default: Florence-2 base model
.\scripts\train.ps1 -Model florence2 -DataDir ./training_data -OutputDir ./output

# Qwen2-VL (better Hebrew, larger model)
.\scripts\train.ps1 -Model qwen2-vl-2b -DataDir ./training_data -OutputDir ./output
```

**Custom training parameters:**

**Linux/macOS:**
```bash
EPOCHS=50 \
BATCH_SIZE=8 \
LR=1e-4 \
LORA_RANK=32 \
STAGE1_EPOCHS=5 \
STAGE2_EPOCHS=15 \
  ./scripts/train.sh florence2 ./training_data ./output
```

**Windows (PowerShell):**
```powershell
$env:EPOCHS="50"; $env:BATCH_SIZE="8"; $env:LR="1e-4"
$env:LORA_RANK="32"; $env:STAGE1_EPOCHS="5"; $env:STAGE2_EPOCHS="15"
.\scripts\train.ps1 -Model florence2 -DataDir ./training_data -OutputDir ./output
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 30 | Total training epochs |
| `BATCH_SIZE` | 4 | Batch size |
| `LR` | 2e-4 | Learning rate |
| `LORA_RANK` | 16 | LoRA adapter rank |
| `STAGE1_EPOCHS` | 5 | Marker recognition stage |
| `STAGE2_EPOCHS` | 10 | Marker-to-text stage |
| `USE_GPU` | true | Use GPU if available (CUDA/MPS) |

**GPU control:**

```bash
# Linux/macOS
USE_GPU=true ./scripts/train.sh florence2 ./training_data ./output   # Enable GPU (default)
USE_GPU=false ./scripts/train.sh florence2 ./training_data ./output  # Disable GPU

# Windows (PowerShell)
$env:USE_GPU="true"; .\scripts\train.ps1 -Model florence2 -DataDir ./training_data -OutputDir ./output
$env:USE_GPU="false"; .\scripts\train.ps1 -Model florence2 -DataDir ./training_data -OutputDir ./output

# Or use train.py directly with --gpu / --no-gpu flags
python training/train.py --train-data ./data --gpu       # Use GPU
python training/train.py --train-data ./data --no-gpu    # Force CPU
```

**Training stages:**

1. **Marker Recognition** (epochs 1-5): Model learns to identify HebMark markers
2. **Marker to Text** (epochs 6-15): Model learns marker → Hebrew associations
3. **Direct OCR** (epochs 16-30): End-to-end Hebrew OCR without markers

### 3. Run Inference

#### Command Line

```bash
source .venv/bin/activate

# OCR a PDF file
python inference/pipeline.py document.pdf --model ./output/best_model

# OCR an image
python inference/pipeline.py page.png --model ./output/best_model

# Save output to file
python inference/pipeline.py document.pdf --model ./output/best_model --output result.txt
```

#### Python API

```python
from inference import create_pipeline

# Load fine-tuned model
pipeline = create_pipeline('./output/best_model', 'florence2')

# OCR a PDF (returns all pages)
text = pipeline('document.pdf')
print(text)

# OCR a single image
text = pipeline('page.png')
print(text)

# OCR with custom prompt
text = pipeline('document.pdf', prompt='Extract all Hebrew text from this document.')
print(text)

# Get per-page results from PDF
results = pipeline.ocr_pdf('document.pdf', output_format='pages')
for page in results:
    print(f"Page {page['page']}: {page['text'][:100]}...")
```

#### Using Base Model (without fine-tuning)

```python
from inference import create_pipeline

# Load base Florence-2 (no fine-tuning)
pipeline = create_pipeline(model_path=None, model_type='florence2')
text = pipeline('document.pdf')
```

### 4. Evaluate Model

**Linux/macOS:**
```bash
# Generate test data
./scripts/generate_data.sh 100 ./test_data

# Run evaluation
./scripts/evaluate.sh ./output/best_model ./test_data
```

**Windows (PowerShell):**
```powershell
# Generate test data
.\scripts\generate_data.ps1 -NumSamples 100 -OutputDir ./test_data

# Run evaluation
.\scripts\evaluate.ps1 -ModelPath ./output/best_model -TestData ./test_data
```

---

## Model Options

### Available Models

| Model | HuggingFace ID | Size | Edge-Ready | Hebrew | Use Case |
|-------|----------------|------|------------|--------|----------|
| `florence2` | `microsoft/Florence-2-base` | 0.23B | ✅ | Fine-tune | Default, fastest inference |
| `florence2-large` | `microsoft/Florence-2-large` | 0.77B | ✅ | Fine-tune | Better accuracy |
| `qwen2-vl-2b` | `Qwen/Qwen2-VL-2B-Instruct` | 2B | ✅ | Native | Best quality, multilingual |
| `paligemma-3b` | `google/paligemma-3b-pt-224` | 3B | ❌ | Native | High quality, larger model |
| `moondream2` | `vikhyatk/moondream2` | 1.6B | ✅ | Fine-tune | Lightweight alternative |

### Model Details

**florence2** (Recommended for edge deployment)
- Microsoft's compact vision-language model
- Optimized for document understanding tasks
- Requires fine-tuning for Hebrew OCR
- Best balance of speed and accuracy

**florence2-large**
- Larger variant with improved accuracy
- Same architecture, more parameters
- Good choice when accuracy matters more than speed

**qwen2-vl-2b** (Recommended for quality)
- Alibaba's multilingual vision-language model
- Native Hebrew and multilingual support
- Higher quality outputs, larger model
- Supports flash attention for faster training

**paligemma-3b**
- Google's vision-language model
- Strong multilingual capabilities
- Not recommended for edge deployment (3B parameters)
- Best for server-side processing

**moondream2**
- Lightweight open-source VLM
- Good for resource-constrained environments
- Requires fine-tuning for Hebrew

### Choosing a Model

| Requirement | Recommended Model |
|-------------|-------------------|
| Edge deployment + speed | `florence2` |
| Edge deployment + accuracy | `florence2-large` |
| Best Hebrew quality | `qwen2-vl-2b` |
| Server-side processing | `paligemma-3b` |
| Minimal resources | `moondream2` |

---

## Project Structure

```
hebllm/
├── data/
│   ├── generator.py      # Synthetic PDF generation
│   └── dataset.py        # PyTorch dataset
├── model/
│   ├── config.py         # Model configurations
│   ├── florence.py       # Florence-2 adapter
│   └── qwen_vl.py        # Qwen2-VL adapter
├── training/
│   ├── train.py          # Training loop
│   ├── curriculum.py     # Curriculum scheduler
│   └── augment.py        # Data augmentations
├── inference/
│   ├── pipeline.py       # Inference API
│   └── postprocess.py    # Text post-processing
├── scripts/
│   ├── generate_data.sh  # Data generation (Linux/macOS)
│   ├── generate_data.ps1 # Data generation (Windows)
│   ├── train.sh          # Training (Linux/macOS)
│   ├── train.ps1         # Training (Windows)
│   ├── evaluate.sh       # Evaluation (Linux/macOS)
│   └── evaluate.ps1      # Evaluation (Windows)
├── hebmark.py            # HebMark marker system
├── hebrew_box_detector.py # Hebrew text detection
├── setup_env.sh          # Environment setup (Linux/macOS)
├── setup_env.ps1         # Environment setup (Windows)
└── requirements.txt      # Dependencies
```

---

## HebMark System

HebMark replaces Hebrew text with visual markers during training:

```
Original:  שלום עולם
Marked:    ◆00 ◆01
Mapping:   ◆00="שלום", ◆01="עולם"
```

This helps the model learn Hebrew character patterns through curriculum learning.

### Using HebMark Directly

```python
from hebmark import create_marked_pdf

# Create marked PDF with mapping
pdf_path, mapping_path = create_marked_pdf('input.pdf')

# Decode markers in text
from hebmark import HebMarkDecoder
decoder = HebMarkDecoder(mapping_path)
hebrew_text = decoder.decode_text('The word ◆00 means hello')
```

---

## Troubleshooting

### Windows: GPU Not Detected

If you see "GPU requested but not available, using CPU" on Windows with an NVIDIA GPU:

**Option 1: Re-run setup script** (recommended)
```powershell
# The setup script auto-detects NVIDIA GPUs and installs CUDA PyTorch
.\setup_env.ps1 -Clean
```

**Option 2: Manual PyTorch CUDA installation**
```powershell
# Activate your environment first
.\.venv\Scripts\Activate.ps1

# Uninstall CPU PyTorch and install CUDA version
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

**Note:** Ensure you have the latest NVIDIA drivers installed. CUDA 12.4 requires driver version 550+.

### Windows: torch_cuda.dll Error

If you see `OSError: [WinError 127] Error loading torch_cuda.dll`:

```powershell
# This usually means CUDA version mismatch. Try CUDA 11.8 (most compatible):
.\.venv\Scripts\Activate.ps1
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify it works
python -c "import torch; print(torch.cuda.is_available())"
```

If still failing, install Visual C++ Redistributable:
- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

### CUDA Out of Memory

Reduce batch size:
```bash
BATCH_SIZE=2 ./scripts/train.sh florence2 ./training_data ./output
```

### Missing Hebrew Fonts

Install Hebrew fonts for PDF generation:
```bash
# macOS (usually pre-installed)
# Linux
sudo apt-get install fonts-dejavu

# Or specify custom font in data/generator.py
```

### Slow Training on CPU

Use a smaller dataset or model:
```bash
./scripts/generate_data.sh 100 ./training_data
EPOCHS=10 ./scripts/train.sh florence2 ./training_data ./output
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 4.36+
- PyMuPDF (for PDF processing)
- reportlab (for PDF generation)
- PEFT (for LoRA fine-tuning)

See `requirements.txt` for full list.

---

## License

MIT
