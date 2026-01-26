#!/usr/bin/env python3
"""
PyTorch Dataset for HebLLM Training

Loads generated training data with HebMark markers and prepares
batches for vision-language model training.
"""

import json
import random
from pathlib import Path
from typing import Optional, Callable, Literal

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class HebLLMDataset(Dataset):
    """
    Dataset for HebLLM training with curriculum learning support.

    Training Stages:
    1. marker_recognition: Predict marker IDs in reading order
    2. marker_to_text: Given markers + mapping, produce Hebrew text
    3. direct_ocr: Raw image → full text transcription
    """

    def __init__(self,
                 data_dir: str | Path,
                 stage: Literal["marker_recognition", "marker_to_text", "direct_ocr"] = "direct_ocr",
                 transform: Optional[Callable] = None,
                 tokenizer: Optional[Callable] = None,
                 max_length: int = 512,
                 image_size: tuple[int, int] = (768, 768)):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing generated training data
            stage: Training stage (affects input/output format)
            transform: Optional image transforms
            tokenizer: Text tokenizer (processor.tokenizer for VLMs)
            max_length: Maximum sequence length
            image_size: Target image size (width, height)
        """
        self.data_dir = Path(data_dir)
        self.stage = stage
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size

        # Load dataset metadata
        metadata_path = self.data_dir / "dataset_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"samples": []}

        # Build sample index
        self._build_index()

    def _build_index(self):
        """Build index of available samples."""
        self.samples = []

        images_dir = self.data_dir / "images"
        metadata_dir = self.data_dir / "metadata"

        if not images_dir.exists():
            return

        for img_path in sorted(images_dir.glob("sample_*.png")):
            idx = img_path.stem.split("_")[1]
            meta_path = metadata_dir / f"sample_{idx}.json"

            if meta_path.exists():
                self.samples.append({
                    "index": idx,
                    "image_path": img_path,
                    "metadata_path": meta_path
                })

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample(self, idx: int) -> dict:
        """Load a sample's data from disk."""
        sample_info = self.samples[idx]

        # Load metadata
        with open(sample_info["metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load image based on stage
        if self.stage == "direct_ocr":
            # Use original (unmarked) image
            image = Image.open(sample_info["image_path"]).convert("RGB")
        else:
            # Use marked image for stages 1 and 2
            marked_path = self.data_dir / "marked_images" / f"sample_{sample_info['index']}_marked.png"
            if marked_path.exists():
                image = Image.open(marked_path).convert("RGB")
            else:
                # Fallback to original if no marked image
                image = Image.open(sample_info["image_path"]).convert("RGB")

        # Load marker mapping
        mapping_path = self.data_dir / "mappings" / f"sample_{sample_info['index']}_mapping.json"
        markers = {}
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                markers = json.load(f)

        return {
            "image": image,
            "markers": markers,
            "ground_truth": metadata.get("ground_truth", ""),
            "language": metadata.get("language", "mixed"),
            "metadata": metadata
        }

    def _prepare_input_output(self, sample: dict) -> tuple[Image.Image, str, str]:
        """Prepare input/output based on training stage.

        Returns:
            Tuple of (image, prompt, target_text)
        """
        image = sample["image"]
        markers = sample["markers"]
        ground_truth = sample["ground_truth"]

        if self.stage == "marker_recognition":
            # Stage 1: Recognize markers in reading order
            prompt = "List all visible markers (◆XX) in reading order, separated by spaces."

            # Build target: sorted markers by position
            if markers:
                sorted_markers = sorted(
                    markers.items(),
                    key=lambda x: (x[1].get("page", 0), x[1].get("line", 0), x[1].get("word", 0))
                )
                target = " ".join(f"◆{mid}" for mid, _ in sorted_markers)
            else:
                target = ""

        elif self.stage == "marker_to_text":
            # Stage 2: Convert markers to Hebrew with mapping provided
            marker_map_str = ", ".join(
                f"◆{mid}={entry['hebrew_text']}"
                for mid, entry in markers.items()
            ) if markers else ""

            prompt = f"Convert markers to Hebrew. Mapping: {marker_map_str}"
            target = ground_truth

        else:  # direct_ocr
            # Stage 3: Direct OCR
            prompt = "Transcribe all text from this document image."
            target = ground_truth

        return image, prompt, target

    def __getitem__(self, idx: int) -> dict:
        """Get a training sample.

        Returns:
            Dictionary with keys: image, prompt, target, language, etc.
        """
        sample = self._load_sample(idx)
        image, prompt, target = self._prepare_input_output(sample)

        # Resize image
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        result = {
            "image": image,
            "prompt": prompt,
            "target": target,
            "language": sample["language"],
            "stage": self.stage
        }

        # Tokenize if tokenizer provided
        if self.tokenizer:
            result["prompt_ids"] = self.tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )["input_ids"].squeeze(0)

            result["target_ids"] = self.tokenizer(
                target,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )["input_ids"].squeeze(0)

        return result


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for variable-length sequences.

    Pads sequences to the maximum length in the batch.
    """
    images = torch.stack([item["image"] for item in batch])

    result = {
        "images": images,
        "prompts": [item["prompt"] for item in batch],
        "targets": [item["target"] for item in batch],
        "languages": [item["language"] for item in batch],
        "stages": [item["stage"] for item in batch]
    }

    # Pad tokenized sequences if present
    if "prompt_ids" in batch[0]:
        max_prompt_len = max(item["prompt_ids"].size(0) for item in batch)
        prompt_ids = torch.zeros(len(batch), max_prompt_len, dtype=torch.long)
        prompt_mask = torch.zeros(len(batch), max_prompt_len, dtype=torch.bool)

        for i, item in enumerate(batch):
            length = item["prompt_ids"].size(0)
            prompt_ids[i, :length] = item["prompt_ids"]
            prompt_mask[i, :length] = True

        result["prompt_ids"] = prompt_ids
        result["prompt_attention_mask"] = prompt_mask

    if "target_ids" in batch[0]:
        max_target_len = max(item["target_ids"].size(0) for item in batch)
        target_ids = torch.zeros(len(batch), max_target_len, dtype=torch.long)
        target_mask = torch.zeros(len(batch), max_target_len, dtype=torch.bool)

        for i, item in enumerate(batch):
            length = item["target_ids"].size(0)
            target_ids[i, :length] = item["target_ids"]
            target_mask[i, :length] = True

        result["target_ids"] = target_ids
        result["target_attention_mask"] = target_mask

    return result


class CurriculumDataLoader:
    """
    DataLoader wrapper that supports curriculum learning stages.

    Automatically switches between stages based on training progress.
    """

    def __init__(self,
                 data_dir: str | Path,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 transform: Optional[Callable] = None,
                 tokenizer: Optional[Callable] = None,
                 pin_memory: bool = False,
                 stage_epochs: dict = None):
        """
        Initialize curriculum data loader.

        Args:
            data_dir: Training data directory
            batch_size: Batch size
            num_workers: DataLoader workers
            transform: Image transforms
            tokenizer: Text tokenizer
            pin_memory: Pin memory for faster CPU->GPU transfer (CUDA only)
            stage_epochs: Dict mapping stage to (start_epoch, end_epoch)
                e.g., {"marker_recognition": (0, 5), "marker_to_text": (5, 15), "direct_ocr": (15, 30)}
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.tokenizer = tokenizer
        self.pin_memory = pin_memory

        self.stage_epochs = stage_epochs or {
            "marker_recognition": (0, 5),
            "marker_to_text": (5, 15),
            "direct_ocr": (15, 30)
        }

        self.current_stage = None
        self.current_loader = None

    def get_stage_for_epoch(self, epoch: int) -> str:
        """Determine training stage for given epoch."""
        for stage, (start, end) in self.stage_epochs.items():
            if start <= epoch < end:
                return stage
        return "direct_ocr"  # Default to final stage

    def get_loader(self, epoch: int) -> DataLoader:
        """Get DataLoader for given epoch, switching stages as needed."""
        stage = self.get_stage_for_epoch(epoch)

        if stage != self.current_stage:
            print(f"Switching to training stage: {stage}")
            self.current_stage = stage

            dataset = HebLLMDataset(
                data_dir=self.data_dir,
                stage=stage,
                transform=self.transform,
                tokenizer=self.tokenizer
            )

            self.current_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn
            )

        return self.current_loader


if __name__ == "__main__":
    # Test dataset loading
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./training_data", help="Data directory")
    parser.add_argument("--stage", default="direct_ocr", choices=["marker_recognition", "marker_to_text", "direct_ocr"])
    args = parser.parse_args()

    dataset = HebLLMDataset(args.data_dir, stage=args.stage)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Prompt: {sample['prompt'][:100]}...")
        print(f"Target: {sample['target'][:100]}...")
