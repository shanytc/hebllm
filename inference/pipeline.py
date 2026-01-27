#!/usr/bin/env python3
"""
Inference Pipeline for HebLLM

Provides a simple API for running OCR inference on PDF pages
using the fine-tuned HebLLM model with HebMark workflow.
"""

import sys
import tempfile
import re
from pathlib import Path
from typing import Union, Optional

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from hebmark import HebMarkEncoder, HebMarkRenderer, HebMarkDecoder, get_tokenizer


class HebLLMPipeline:
    """
    End-to-end inference pipeline for Hebrew/English OCR.

    Uses HebMark workflow:
    1. Convert PDF to HebMark (replace Hebrew with markers)
    2. Run inference to recognize markers
    3. Decode markers back to Hebrew

    Usage:
        pipeline = HebLLMPipeline.from_pretrained("./output/best_model")
        text = pipeline("document.pdf")
    """

    # Training prompts - must match what model was trained on
    PROMPT_MARKER_RECOGNITION = "List all visible markers (◆XX) in reading order, separated by spaces."
    PROMPT_MARKER_TO_TEXT = "Convert markers to Hebrew. Mapping: {mapping}"
    PROMPT_DIRECT_OCR = "Transcribe all text from this document image."

    def __init__(self,
                 model,
                 processor,
                 device: str = "auto",
                 image_size: tuple[int, int] = (768, 768),
                 stage: str = "marker_recognition"):
        """
        Initialize pipeline.

        Args:
            model: Loaded model
            processor: Model processor/tokenizer
            device: Target device
            image_size: Input image size
            stage: Inference stage ("marker_recognition", "marker_to_text", "direct_ocr")
        """
        self.model = model
        self.processor = processor
        self.image_size = image_size
        self.stage = stage

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    @classmethod
    def from_pretrained(cls,
                        model_path: str | Path,
                        model_type: str = "florence2",
                        device: str = "auto",
                        stage: str = "marker_recognition",
                        **kwargs) -> "HebLLMPipeline":
        """
        Load pipeline from pretrained checkpoint.

        Args:
            model_path: Path to saved model
            model_type: Model architecture ("florence2" or "qwen2-vl")
            device: Target device
            stage: Inference stage

        Returns:
            Configured HebLLMPipeline
        """
        model_path = Path(model_path)

        if model_type.startswith("florence"):
            from model.florence import Florence2Adapter
            adapter = Florence2Adapter.from_pretrained(model_path, device=device)
            actual_device = next(adapter.get_model().parameters()).device
            print(f"Model loaded on: {actual_device}")
            return cls(
                model=adapter,
                processor=adapter.processor,
                device=device,
                image_size=(768, 768),
                stage=stage
            )

        elif model_type.startswith("qwen"):
            from model.qwen_vl import Qwen2VLAdapter
            adapter = Qwen2VLAdapter.from_pretrained(model_path, device=device)
            return cls(
                model=adapter,
                processor=adapter.processor,
                device=device,
                image_size=(896, 896),
                stage=stage
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _load_image(self, source: Union[str, Path, Image.Image]) -> Image.Image:
        """Load and preprocess image."""
        if isinstance(source, Image.Image):
            image = source
        else:
            image = Image.open(source)

        image = image.convert("RGB")
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        return image

    def _pdf_to_images(self, pdf_path: str | Path, dpi: int = 150) -> list[Image.Image]:
        """Convert PDF pages to images."""
        if fitz is None:
            raise ImportError("PyMuPDF required for PDF processing. Install: pip install pymupdf")

        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        doc.close()
        return images

    def _is_hebrew(self, text: str) -> bool:
        """Check if text contains Hebrew characters."""
        return bool(re.search(r'[\u0590-\u05FF]', text))

    def _extract_hebrew_words(self, page: fitz.Page) -> list[dict]:
        """Extract Hebrew words with bounding boxes from a PDF page."""
        words = []
        blocks = page.get_text("dict")["blocks"]

        for block_idx, block in enumerate(blocks):
            if "lines" not in block:
                continue
            for line_idx, line in enumerate(block["lines"]):
                for word_idx, span in enumerate(line["spans"]):
                    text = span["text"].strip()
                    if text and self._is_hebrew(text):
                        bbox = fitz.Rect(span["bbox"])
                        words.append({
                            "text": text,
                            "bbox": bbox,
                            "block": block_idx,
                            "line": line_idx,
                            "word": word_idx
                        })
        return words

    def _apply_hebmark(self, pdf_path: str | Path, dpi: int = 150) -> tuple[list[Image.Image], HebMarkEncoder]:
        """
        Apply HebMark to PDF - replace Hebrew with markers.

        Returns:
            Tuple of (marked_images, encoder)
        """
        if fitz is None:
            raise ImportError("PyMuPDF required. Install: pip install pymupdf")

        # Initialize encoder
        tokenizer = get_tokenizer("utf8")
        encoder = HebMarkEncoder(tokenizer=tokenizer)
        renderer = HebMarkRenderer(encoder)

        # Open PDF
        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract Hebrew words
            hebrew_words = self._extract_hebrew_words(page)

            if hebrew_words:
                print(f"  Page {page_num + 1}: Found {len(hebrew_words)} Hebrew words")
                # Replace Hebrew with markers
                renderer.replace_hebrew_with_markers(page, hebrew_words, page_num)

            # Render page to image
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        doc.close()
        return images, encoder

    def _decode_markers(self, text: str, encoder: HebMarkEncoder) -> str:
        """Decode marker IDs in text back to Hebrew."""
        # Pattern: ◆XX where XX is alphanumeric
        pattern = r'◆([0-9A-Za-z]+)'

        def replace(match):
            marker_id = match.group(1).upper()
            entry = encoder.decode(marker_id)
            if entry:
                return entry.hebrew_text
            return match.group(0)  # Keep original if not found

        return re.sub(pattern, replace, text)

    def __call__(self,
                 source: Union[str, Path, Image.Image, list],
                 prompt: str = None,
                 max_new_tokens: int = 1024,
                 use_hebmark: bool = True,
                 **kwargs) -> Union[str, list[str]]:
        """
        Run OCR inference.

        Args:
            source: Image path, PIL Image, PDF path, or list of images
            prompt: Custom prompt (optional, uses stage-appropriate default)
            max_new_tokens: Maximum tokens to generate
            use_hebmark: Whether to use HebMark workflow for PDFs

        Returns:
            Transcribed text (string or list of strings)
        """
        encoder = None

        # Handle different input types
        if isinstance(source, (str, Path)):
            source = Path(source)
            if source.suffix.lower() == ".pdf":
                if use_hebmark:
                    print(f"Applying HebMark to PDF...")
                    images, encoder = self._apply_hebmark(source)
                    print(f"  Created {len(encoder._id_to_entry)} markers")
                else:
                    images = self._pdf_to_images(source)
            else:
                images = [self._load_image(source)]
        elif isinstance(source, Image.Image):
            images = [self._load_image(source)]
        elif isinstance(source, list):
            images = [self._load_image(img) for img in source]
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        # Select prompt based on stage
        if prompt is None:
            if self.stage == "marker_recognition":
                prompt = self.PROMPT_MARKER_RECOGNITION
            elif self.stage == "marker_to_text":
                # For marker_to_text, we need to provide the mapping
                if encoder:
                    mapping_str = ", ".join(
                        f"◆{mid}={entry.hebrew_text}"
                        for mid, entry in encoder._id_to_entry.items()
                    )
                    prompt = self.PROMPT_MARKER_TO_TEXT.format(mapping=mapping_str)
                else:
                    prompt = self.PROMPT_MARKER_TO_TEXT.format(mapping="")
            else:  # direct_ocr
                prompt = self.PROMPT_DIRECT_OCR

        print(f"Using prompt: {prompt[:80]}...")

        # Run inference
        results = self.model.generate(
            images,
            prompts=[prompt] * len(images),
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        print(f"Raw model output: {results}")

        # Decode markers if HebMark was used
        if encoder and self.stage == "marker_recognition":
            print("Decoding markers to Hebrew...")
            results = [self._decode_markers(text, encoder) for text in results]

        # Return single string if single image
        if len(results) == 1:
            return results[0]
        return results

    def ocr_pdf(self,
                pdf_path: str | Path,
                output_format: str = "text",
                dpi: int = 150,
                use_hebmark: bool = True) -> Union[str, list[dict]]:
        """
        OCR an entire PDF document.

        Args:
            pdf_path: Path to PDF file
            output_format: "text" (concatenated) or "pages" (list of dicts)
            dpi: Rendering resolution
            use_hebmark: Whether to use HebMark workflow

        Returns:
            Extracted text
        """
        result = self(pdf_path, use_hebmark=use_hebmark)

        if isinstance(result, str):
            results = [result]
        else:
            results = result

        if output_format == "text":
            return "\n\n---\n\n".join(results)
        else:
            return [
                {"page": i + 1, "text": text}
                for i, text in enumerate(results)
            ]


def create_pipeline(
    model_path: str | Path = None,
    model_type: str = "florence2",
    device: str = "auto",
    stage: str = "marker_recognition"
) -> HebLLMPipeline:
    """
    Factory function to create inference pipeline.

    Args:
        model_path: Path to fine-tuned model (None for base model)
        model_type: Model architecture
        device: Target device
        stage: Inference stage

    Returns:
        Configured HebLLMPipeline
    """
    if model_path:
        return HebLLMPipeline.from_pretrained(model_path, model_type, device, stage)

    # Load base model without fine-tuning
    if model_type.startswith("florence"):
        from model.florence import create_florence2_model
        adapter = create_florence2_model(
            model_size="base",
            use_lora=False,
            device=device
        )
        return HebLLMPipeline(adapter, adapter.processor, device, stage=stage)

    elif model_type.startswith("qwen"):
        from model.qwen_vl import create_qwen2vl_model
        adapter = create_qwen2vl_model(
            model_size="2B",
            use_lora=False,
            device=device
        )
        return HebLLMPipeline(adapter, adapter.processor, device, stage=stage)

    raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HebLLM OCR inference")
    parser.add_argument("input", help="Input image or PDF file")
    parser.add_argument("--model", "-m", help="Path to fine-tuned model")
    parser.add_argument("--type", "-t", default="florence2",
                        choices=["florence2", "qwen2-vl"],
                        help="Model type")
    parser.add_argument("--stage", "-s", default="marker_recognition",
                        choices=["marker_recognition", "marker_to_text", "direct_ocr"],
                        help="Inference stage (default: marker_recognition)")
    parser.add_argument("--no-hebmark", action="store_true",
                        help="Disable HebMark workflow (use direct inference)")
    parser.add_argument("--output", "-o", help="Output file (optional)")

    args = parser.parse_args()

    pipeline = create_pipeline(args.model, args.type, stage=args.stage)
    result = pipeline(args.input, use_hebmark=not args.no_hebmark)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result if isinstance(result, str) else "\n\n".join(result))
        print(f"Output saved to: {args.output}")
    else:
        print(result if isinstance(result, str) else "\n\n".join(result))
