#!/usr/bin/env python3
"""
Qwen2-VL Model Adapter for HebLLM

Wraps Alibaba's Qwen2-VL model for Hebrew OCR fine-tuning.
Qwen2-VL has excellent multilingual support including Hebrew,
making it ideal when higher quality is needed at the cost of size.
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from PIL import Image


class Qwen2VLAdapter:
    """
    Adapter for Qwen2-VL model fine-tuning.

    Qwen2-VL uses a ViT vision encoder and Qwen2 LLM backbone.
    It supports native resolution processing and has strong OCR capabilities.
    """

    def __init__(self,
                 model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
                 device: str = "auto",
                 torch_dtype: torch.dtype = torch.bfloat16,
                 use_lora: bool = True,
                 lora_config: dict = None,
                 use_flash_attention: bool = True):
        """
        Initialize Qwen2-VL adapter.

        Args:
            model_id: HuggingFace model ID
            device: Device to use
            torch_dtype: Model dtype (bfloat16 recommended)
            use_lora: Whether to use LoRA fine-tuning
            lora_config: LoRA configuration
            use_flash_attention: Use Flash Attention 2
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_lora = use_lora
        self.lora_config = lora_config or {}
        self.use_flash_attention = use_flash_attention

        self.model = None
        self.processor = None
        self.peft_model = None

    def _check_flash_attention_available(self) -> bool:
        """Check if Flash Attention 2 is available."""
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    def load_model(self):
        """Load the Qwen2-VL model and processor."""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        print(f"Loading Qwen2-VL model: {self.model_id}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            # Limit image resolution for faster processing
            min_pixels=256 * 28 * 28,   # ~200K pixels
            max_pixels=512 * 28 * 28,   # ~400K pixels (instead of default 1280*28*28)
        )

        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": self.device,
            "trust_remote_code": True
        }

        # Check if flash attention is requested and available
        if self.use_flash_attention:
            if self._check_flash_attention_available():
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("Using Flash Attention 2")
            else:
                print("Flash Attention not available, using SDPA attention")
                model_kwargs["attn_implementation"] = "sdpa"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs
        )

        if self.use_lora:
            self._apply_lora()

        print(f"Image resolution: {self.processor.image_processor.min_pixels}-{self.processor.image_processor.max_pixels} pixels")
        return self

    def _apply_lora(self):
        """Apply LoRA adapters for efficient fine-tuning."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            print("Warning: PEFT not installed. Running without LoRA.")
            return

        target_modules = self.lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ])

        lora_config = LoraConfig(
            r=self.lora_config.get("r", 32),
            lora_alpha=self.lora_config.get("lora_alpha", 64),
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
                       prompts: list[str]) -> dict:
        """
        Prepare inputs for Qwen2-VL.

        Args:
            images: List of PIL images
            prompts: List of text prompts

        Returns:
            Dict with model inputs
        """
        # Qwen2-VL uses a specific message format
        messages_batch = []

        for image, prompt in zip(images, prompts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            messages_batch.append(messages)

        # Process each message set
        # For batch processing, we need to handle images separately
        texts = []
        for messages in messages_batch:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)

        inputs = self.processor(
            text=texts,
            images=images,
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
        dtype = next(model.parameters()).dtype

        if labels is not None:
            # For training: create full conversation with response included
            # Labels must be aligned with input_ids (same length)
            messages_batch = []
            for image, prompt, label in zip(images, prompts, labels):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": label}
                        ]
                    }
                ]
                messages_batch.append(messages)

            # Process with labels included
            texts = []
            for messages in messages_batch:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False  # Response already included
                )
                texts.append(text)

            inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}

            # Create labels: copy input_ids but mask the prompt portion with -100
            input_ids = inputs["input_ids"]
            labels_tensor = input_ids.clone()

            # Find assistant marker token to locate where response starts
            # Qwen2-VL uses <|im_start|>assistant pattern
            # Token ID for "assistant" after <|im_start|> is typically 77091
            # But we can find it by looking for the pattern
            assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids("assistant")

            for i in range(input_ids.shape[0]):
                # Find the last occurrence of assistant token (marks start of response)
                seq = input_ids[i]
                assistant_positions = (seq == assistant_token_id).nonzero(as_tuple=True)[0]

                if len(assistant_positions) > 0:
                    # Mask everything up to and including the assistant token + newline
                    response_start = assistant_positions[-1].item() + 2  # +2 to skip "assistant\n"
                    labels_tensor[i, :response_start] = -100

            # Also mask padding tokens
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is not None:
                labels_tensor[input_ids == pad_token_id] = -100

            inputs["labels"] = labels_tensor

        else:
            # Inference mode
            inputs = self.prepare_inputs(images, prompts)
            inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        return outputs

    def generate(self,
                 images: Union[Image.Image, list[Image.Image]],
                 prompts: Union[str, list[str]] = "Transcribe all text in this image.",
                 max_new_tokens: int = 2048,
                 **kwargs) -> list[str]:
        """
        Generate text from images.

        Args:
            images: Input image(s)
            prompts: Input prompt(s)
            max_new_tokens: Maximum tokens to generate

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
                **kwargs
            )

        # Decode outputs, removing input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        results = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True
        )

        return results

    def save_pretrained(self, output_dir: str | Path):
        """Save the fine-tuned model."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self.get_model()

        if self.peft_model:
            self.peft_model.save_pretrained(output_dir)
        else:
            model.save_pretrained(output_dir)

        self.processor.save_pretrained(output_dir)
        print(f"Model saved to: {output_dir}")

    @classmethod
    def from_pretrained(cls,
                        model_dir: str | Path,
                        base_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
                        **kwargs) -> "Qwen2VLAdapter":
        """Load a fine-tuned model."""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from peft import PeftModel

        model_dir = Path(model_dir)
        adapter = cls(model_id=str(model_dir), **kwargs)

        adapter_config = model_dir / "adapter_config.json"

        if adapter_config.exists():
            # Load base model + LoRA
            adapter.processor = AutoProcessor.from_pretrained(
                model_dir, trust_remote_code=True
            )

            model_kwargs = {"torch_dtype": adapter.torch_dtype, "device_map": adapter.device}
            if adapter.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model_id, trust_remote_code=True, **model_kwargs
            )

            adapter.peft_model = PeftModel.from_pretrained(base_model, model_dir)
            adapter.model = base_model
        else:
            adapter.processor = AutoProcessor.from_pretrained(
                model_dir, trust_remote_code=True
            )

            model_kwargs = {"torch_dtype": adapter.torch_dtype, "device_map": adapter.device}
            if adapter.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            adapter.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_dir, trust_remote_code=True, **model_kwargs
            )

        return adapter


class Qwen2VLForOCR(nn.Module):
    """
    Qwen2-VL wrapper for OCR training with curriculum learning.
    """

    def __init__(self, adapter: Qwen2VLAdapter):
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
        """Forward pass for training.

        Args:
            images: Batch of images (B, C, H, W)
            prompts: List of prompts (already formatted by dataset for each stage)
            targets: List of target texts
            stage: Training stage (for logging only, prompts come from dataset)

        Returns:
            Dict with loss
        """
        # Convert tensor images to PIL
        pil_images = []
        for img in images:
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            pil_images.append(Image.fromarray(img_np))

        # Use prompts directly from dataset - they're already properly formatted
        # for each curriculum stage (marker_recognition, marker_to_text, direct_ocr)
        outputs = self.adapter.forward(pil_images, prompts, targets)
        return {"loss": outputs.loss, "logits": outputs.logits}

    def generate(self, images: torch.Tensor, prompts: list[str] = None, **kwargs) -> list[str]:
        """Generate text from images."""
        pil_images = []
        for img in images:
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            pil_images.append(Image.fromarray(img_np))

        default_prompt = "Transcribe all text from this document image."
        return self.adapter.generate(pil_images, prompts or [default_prompt], **kwargs)


def create_qwen2vl_model(
    model_size: str = "2B",
    use_lora: bool = True,
    lora_rank: int = 32,
    device: str = "auto"
) -> Qwen2VLAdapter:
    """
    Factory function to create Qwen2-VL model.

    Args:
        model_size: "2B" or "7B"
        use_lora: Enable LoRA fine-tuning
        lora_rank: LoRA rank
        device: Target device

    Returns:
        Configured Qwen2VLAdapter
    """
    model_id = f"Qwen/Qwen2-VL-{model_size}-Instruct"

    lora_config = {
        "r": lora_rank,
        "lora_alpha": lora_rank * 2,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    }

    adapter = Qwen2VLAdapter(
        model_id=model_id,
        device=device,
        use_lora=use_lora,
        lora_config=lora_config
    )

    return adapter.load_model()


if __name__ == "__main__":
    print("Testing Qwen2-VL adapter...")

    test_image = Image.new("RGB", (896, 896), color="white")

    try:
        adapter = create_qwen2vl_model(model_size="2B", use_lora=False, device="cpu")
        results = adapter.generate(test_image, "What text is in this image?")
        print(f"Test output: {results}")
    except Exception as e:
        print(f"Test requires model download: {e}")
