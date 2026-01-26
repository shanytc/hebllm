#!/usr/bin/env python3
"""
Florence-2 Model Adapter for HebLLM

Wraps Microsoft's Florence-2 model for Hebrew OCR fine-tuning.
Florence-2 is a lightweight (0.23B-0.77B) vision-language model
with strong OCR capabilities.
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from PIL import Image


class Florence2Adapter:
    """
    Adapter for Florence-2 model fine-tuning.

    Florence-2 uses a DaViT vision encoder and a BART-like language model.
    For OCR tasks, we use the <OCR> and <OCR_WITH_REGION> task prompts.
    """

    def __init__(self,
                 model_id: str = "microsoft/Florence-2-base",
                 device: str = "auto",
                 torch_dtype: torch.dtype = torch.float16,
                 use_lora: bool = True,
                 lora_config: dict = None):
        """
        Initialize Florence-2 adapter.

        Args:
            model_id: HuggingFace model ID
            device: Device to use ("auto", "cuda", "cpu")
            torch_dtype: Model dtype
            use_lora: Whether to use LoRA fine-tuning
            lora_config: LoRA configuration dict
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_lora = use_lora
        self.lora_config = lora_config or {}

        self.model = None
        self.processor = None
        self.peft_model = None

    def load_model(self):
        """Load the Florence-2 model and processor."""
        from transformers import AutoProcessor, AutoModelForCausalLM

        print(f"Loading Florence-2 model: {self.model_id}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True
        )

        if self.use_lora:
            self._apply_lora()

        return self

    def _apply_lora(self):
        """Apply LoRA adapters for efficient fine-tuning."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            print("Warning: PEFT not installed. Running without LoRA.")
            return

        # Florence-2 specific LoRA targets
        target_modules = self.lora_config.get("target_modules", [
            "q_proj", "v_proj", "k_proj", "out_proj",  # Attention
            "fc1", "fc2"  # FFN
        ])

        lora_config = LoraConfig(
            r=self.lora_config.get("r", 16),
            lora_alpha=self.lora_config.get("lora_alpha", 32),
            lora_dropout=self.lora_config.get("lora_dropout", 0.05),
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()

    def get_model(self):
        """Get the model (with LoRA if enabled)."""
        return self.peft_model if self.peft_model else self.model

    def prepare_inputs(self,
                       images: list[Image.Image],
                       prompts: list[str],
                       task: str = "<OCR>") -> dict:
        """
        Prepare inputs for Florence-2.

        Args:
            images: List of PIL images
            prompts: List of text prompts (or use default OCR task)
            task: Florence-2 task token

        Returns:
            Dict with model inputs
        """
        # Florence-2 uses specific task tokens
        # For OCR: <OCR>, <OCR_WITH_REGION>
        # For captioning: <CAPTION>, <DETAILED_CAPTION>

        # Prepend task token if not present
        processed_prompts = []
        for prompt in prompts:
            if not prompt.startswith("<"):
                prompt = f"{task} {prompt}"
            processed_prompts.append(prompt)

        inputs = self.processor(
            images=images,
            text=processed_prompts,
            return_tensors="pt",
            padding=True
        )

        return inputs

    def forward(self,
                images: list[Image.Image],
                prompts: list[str],
                labels: Optional[list[str]] = None) -> dict:
        """
        Forward pass with optional training.

        Args:
            images: Input images
            prompts: Input prompts
            labels: Target text (for training)

        Returns:
            Dict with loss and logits
        """
        model = self.get_model()
        device = next(model.parameters()).device

        inputs = self.prepare_inputs(images, prompts)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if labels is not None:
            # Tokenize labels
            label_inputs = self.processor.tokenizer(
                labels,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            inputs["labels"] = label_inputs["input_ids"].to(device)

        outputs = model(**inputs)
        return outputs

    def generate(self,
                 images: Union[Image.Image, list[Image.Image]],
                 prompts: Union[str, list[str]] = "<OCR>",
                 max_new_tokens: int = 1024,
                 num_beams: int = 3,
                 **kwargs) -> list[str]:
        """
        Generate text from images.

        Args:
            images: Input image(s)
            prompts: Input prompt(s)
            max_new_tokens: Maximum tokens to generate
            num_beams: Beam search width

        Returns:
            List of generated text strings
        """
        model = self.get_model()
        model.eval()

        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)

        device = next(model.parameters()).device
        inputs = self.prepare_inputs(images, prompts)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                **kwargs
            )

        # Decode outputs
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        # Post-process Florence-2 outputs
        results = []
        for text in generated_texts:
            # Florence-2 may include task tokens in output
            for task_token in ["<OCR>", "<OCR_WITH_REGION>", "<CAPTION>"]:
                text = text.replace(task_token, "").strip()
            results.append(text)

        return results

    def save_pretrained(self, output_dir: str | Path):
        """Save the fine-tuned model."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self.get_model()

        if self.peft_model:
            # Save LoRA weights
            self.peft_model.save_pretrained(output_dir)
        else:
            # Save full model
            model.save_pretrained(output_dir)

        self.processor.save_pretrained(output_dir)
        print(f"Model saved to: {output_dir}")

    @classmethod
    def from_pretrained(cls,
                        model_dir: str | Path,
                        **kwargs) -> "Florence2Adapter":
        """Load a fine-tuned model."""
        from transformers import AutoProcessor, AutoModelForCausalLM
        from peft import PeftModel

        model_dir = Path(model_dir)

        adapter = cls(model_id=str(model_dir), **kwargs)

        # Check if this is a LoRA checkpoint
        adapter_config = model_dir / "adapter_config.json"

        if adapter_config.exists():
            # Load base model + LoRA
            base_model_id = kwargs.get("base_model_id", "microsoft/Florence-2-base")

            adapter.processor = AutoProcessor.from_pretrained(
                model_dir, trust_remote_code=True
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=adapter.torch_dtype,
                device_map=adapter.device,
                trust_remote_code=True
            )

            adapter.peft_model = PeftModel.from_pretrained(base_model, model_dir)
            adapter.model = base_model
        else:
            # Load full model
            adapter.processor = AutoProcessor.from_pretrained(
                model_dir, trust_remote_code=True
            )
            adapter.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=adapter.torch_dtype,
                device_map=adapter.device,
                trust_remote_code=True
            )

        return adapter


class Florence2ForOCR(nn.Module):
    """
    Florence-2 wrapper specifically for OCR training.

    Handles the curriculum learning stages for HebMark training.
    """

    def __init__(self, adapter: Florence2Adapter):
        super().__init__()
        self.adapter = adapter

    @property
    def model(self):
        return self.adapter.get_model()

    def forward(self,
                images: torch.Tensor,
                prompts: list[str],
                targets: list[str],
                stage: str = "direct_ocr") -> dict:
        """
        Forward pass for training.

        Args:
            images: Batch of images (B, C, H, W)
            prompts: List of prompts
            targets: List of target texts
            stage: Training stage

        Returns:
            Dict with loss
        """
        # Convert tensor images to PIL
        pil_images = []
        for img in images:
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            pil_images.append(Image.fromarray(img_np))

        # Select task token based on stage
        if stage == "marker_recognition":
            task = "<OCR>"
        elif stage == "marker_to_text":
            task = "<OCR>"  # Custom task, same token
        else:
            task = "<OCR>"

        # Add task token to prompts
        task_prompts = [f"{task} {p}" for p in prompts]

        outputs = self.adapter.forward(pil_images, task_prompts, targets)
        return {"loss": outputs.loss, "logits": outputs.logits}

    def generate(self, images: torch.Tensor, prompts: list[str] = None, **kwargs) -> list[str]:
        """Generate text from images."""
        pil_images = []
        for img in images:
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            pil_images.append(Image.fromarray(img_np))

        return self.adapter.generate(pil_images, prompts or ["<OCR>"], **kwargs)


def create_florence2_model(
    model_size: str = "base",
    use_lora: bool = True,
    lora_rank: int = 16,
    device: str = "auto"
) -> Florence2Adapter:
    """
    Factory function to create Florence-2 model.

    Args:
        model_size: "base" (0.23B) or "large" (0.77B)
        use_lora: Enable LoRA fine-tuning
        lora_rank: LoRA rank
        device: Target device

    Returns:
        Configured Florence2Adapter
    """
    model_id = f"microsoft/Florence-2-{model_size}"

    lora_config = {
        "r": lora_rank,
        "lora_alpha": lora_rank * 2,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    }

    adapter = Florence2Adapter(
        model_id=model_id,
        device=device,
        use_lora=use_lora,
        lora_config=lora_config
    )

    return adapter.load_model()


if __name__ == "__main__":
    # Test the adapter
    print("Testing Florence-2 adapter...")

    # Create a simple test image
    test_image = Image.new("RGB", (768, 768), color="white")

    try:
        adapter = create_florence2_model(model_size="base", use_lora=False, device="cpu")
        results = adapter.generate(test_image, "<OCR>")
        print(f"Test output: {results}")
    except Exception as e:
        print(f"Test requires model download: {e}")
