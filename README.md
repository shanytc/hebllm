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

```bash
# Basic: 100 samples
./scripts/generate_data.sh

# Custom: 1000 samples to specific directory
./scripts/generate_data.sh 1000 ./training_data

# Custom distribution (environment variables)
HEBREW_PCT=0.5 ENGLISH_PCT=0.3 MIXED_PCT=0.2 \
  ./scripts/generate_data.sh 5000 ./training_data
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

```bash
# Default: Florence-2 base model
./scripts/train.sh florence2 ./training_data ./output

# Qwen2-VL (better Hebrew, larger model)
./scripts/train.sh qwen2-vl-2b ./training_data ./output
```

**Custom training parameters:**

```bash
EPOCHS=50 \
BATCH_SIZE=8 \
LR=1e-4 \
LORA_RANK=32 \
STAGE1_EPOCHS=5 \
STAGE2_EPOCHS=15 \
  ./scripts/train.sh florence2 ./training_data ./output
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
# Enable GPU (default)
USE_GPU=true ./scripts/train.sh florence2 ./training_data ./output

# Disable GPU (CPU only)
USE_GPU=false ./scripts/train.sh florence2 ./training_data ./output

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

```bash
# Generate test data
./scripts/generate_data.sh 100 ./test_data

# Run evaluation
./scripts/evaluate.sh ./output/best_model ./test_data
```

---

## Model Options

| Model | Size | Edge-Ready | Hebrew Support | Use Case |
|-------|------|------------|----------------|----------|
| `florence2` | 0.23B | ✅ | After fine-tuning | Default, fast inference |
| `florence2-large` | 0.77B | ✅ | After fine-tuning | Better accuracy |
| `qwen2-vl-2b` | 2B | ✅ | Native | Best quality |

**Recommendation:** Start with `florence2` for edge deployment. Use `qwen2-vl-2b` if quality is insufficient.

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
│   ├── generate_data.sh  # Data generation script
│   ├── train.sh          # Training script
│   └── evaluate.sh       # Evaluation script
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
