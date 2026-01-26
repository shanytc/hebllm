#!/usr/bin/env python3
"""
Inference Pipeline for HebLLM

Provides a simple API for running OCR inference on PDF pages
using the fine-tuned HebLLM model.
"""

import sys
from pathlib import Path
from typing import Union, Optional

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class HebLLMPipeline:
    """
    End-to-end inference pipeline for Hebrew/English OCR.

    Usage:
        pipeline = HebLLMPipeline.from_pretrained("./output/best_model")
        text = pipeline("document.pdf")
    """

    def __init__(self,
                 model,
                 processor,
                 device: str = "auto",
                 image_size: tuple[int, int] = (768, 768)):
        """
        Initialize pipeline.

        Args:
            model: Loaded model
            processor: Model processor/tokenizer
            device: Target device
            image_size: Input image size
        """
        self.model = model
        self.processor = processor
        self.image_size = image_size

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
                        **kwargs) -> "HebLLMPipeline":
        """
        Load pipeline from pretrained checkpoint.

        Args:
            model_path: Path to saved model
            model_type: Model architecture ("florence2" or "qwen2-vl")
            device: Target device

        Returns:
            Configured HebLLMPipeline
        """
        model_path = Path(model_path)

        if model_type.startswith("florence"):
            from model.florence import Florence2Adapter
            adapter = Florence2Adapter.from_pretrained(model_path, device=device)
            return cls(
                model=adapter,
                processor=adapter.processor,
                device=device,
                image_size=(768, 768)
            )

        elif model_type.startswith("qwen"):
            from model.qwen_vl import Qwen2VLAdapter
            adapter = Qwen2VLAdapter.from_pretrained(model_path, device=device)
            return cls(
                model=adapter,
                processor=adapter.processor,
                device=device,
                image_size=(896, 896)
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

    def __call__(self,
                 source: Union[str, Path, Image.Image, list],
                 prompt: str = None,
                 max_new_tokens: int = 1024,
                 **kwargs) -> Union[str, list[str]]:
        """
        Run OCR inference.

        Args:
            source: Image path, PIL Image, PDF path, or list of images
            prompt: Custom prompt (optional)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Transcribed text (string or list of strings)
        """
        # Handle different input types
        if isinstance(source, (str, Path)):
            source = Path(source)
            if source.suffix.lower() == ".pdf":
                images = self._pdf_to_images(source)
            else:
                images = [self._load_image(source)]
        elif isinstance(source, Image.Image):
            images = [self._load_image(source)]
        elif isinstance(source, list):
            images = [self._load_image(img) for img in source]
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        # Default prompt
        if prompt is None:
            prompt = "Transcribe all text from this document image."

        # Run inference
        results = self.model.generate(
            images,
            prompts=[prompt] * len(images),
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        # Return single string if single image
        if len(results) == 1:
            return results[0]
        return results

    def ocr_pdf(self,
                pdf_path: str | Path,
                output_format: str = "text",
                dpi: int = 150) -> Union[str, list[dict]]:
        """
        OCR an entire PDF document.

        Args:
            pdf_path: Path to PDF file
            output_format: "text" (concatenated) or "pages" (list of dicts)
            dpi: Rendering resolution

        Returns:
            Extracted text
        """
        images = self._pdf_to_images(pdf_path, dpi)
        results = self(images)

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
    device: str = "auto"
) -> HebLLMPipeline:
    """
    Factory function to create inference pipeline.

    Args:
        model_path: Path to fine-tuned model (None for base model)
        model_type: Model architecture
        device: Target device

    Returns:
        Configured HebLLMPipeline
    """
    if model_path:
        return HebLLMPipeline.from_pretrained(model_path, model_type, device)

    # Load base model without fine-tuning
    if model_type.startswith("florence"):
        from model.florence import create_florence2_model
        adapter = create_florence2_model(
            model_size="base",
            use_lora=False,
            device=device
        )
        return HebLLMPipeline(adapter, adapter.processor, device)

    elif model_type.startswith("qwen"):
        from model.qwen_vl import create_qwen2vl_model
        adapter = create_qwen2vl_model(
            model_size="2B",
            use_lora=False,
            device=device
        )
        return HebLLMPipeline(adapter, adapter.processor, device)

    raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HebLLM OCR inference")
    parser.add_argument("input", help="Input image or PDF file")
    parser.add_argument("--model", "-m", help="Path to fine-tuned model")
    parser.add_argument("--type", "-t", default="florence2",
                        choices=["florence2", "qwen2-vl"],
                        help="Model type")
    parser.add_argument("--output", "-o", help="Output file (optional)")

    args = parser.parse_args()

    pipeline = create_pipeline(args.model, args.type)
    result = pipeline(args.input)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Output saved to: {args.output}")
    else:
        print(result)
