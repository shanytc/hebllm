#!/usr/bin/env python3
"""
Training Script for HebLLM

Implements curriculum learning with three stages:
1. Marker Recognition: Model learns to identify HebMark markers
2. Marker to Text: Model learns marker-to-Hebrew associations
3. Direct OCR: Model performs end-to-end Hebrew OCR
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import HebLLMDataset, CurriculumDataLoader, collate_fn
from model.config import HebLLMConfig, get_model_config, TrainingConfig
from training.curriculum import CurriculumScheduler, StageConfig
from training.augment import get_training_transforms


class HebLLMTrainer:
    """
    Main trainer class for HebLLM.

    Handles curriculum learning, checkpointing, and evaluation.
    """

    def __init__(self,
                 config: HebLLMConfig,
                 model: nn.Module = None,
                 tokenizer=None,
                 use_gpu: bool = True):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            model: Model to train (optional, will create from config)
            tokenizer: Tokenizer for text processing
            use_gpu: Whether to use GPU if available (default: True)
        """
        self.config = config
        self.device = self._setup_device(use_gpu)

        # Create output directories
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Initialize model
        self.model = model
        self.tokenizer = tokenizer
        if model is None:
            self._init_model()

        # Initialize curriculum scheduler
        self.curriculum = CurriculumScheduler(
            total_epochs=config.training.num_epochs,
            stage_configs=[
                StageConfig("marker_recognition", 0, config.training.stage1_epochs),
                StageConfig("marker_to_text", config.training.stage1_epochs,
                           config.training.stage1_epochs + config.training.stage2_epochs),
                StageConfig("direct_ocr",
                           config.training.stage1_epochs + config.training.stage2_epochs,
                           config.training.num_epochs)
            ]
        )

        # Initialize data loaders
        self.train_loader = None
        self.val_loader = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        # Logging
        self.train_log = []

    def _setup_device(self, use_gpu: bool = True) -> torch.device:
        """Setup training device.

        Args:
            use_gpu: Whether to use GPU if available

        Returns:
            torch.device to use for training
        """
        if use_gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Using CUDA: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                print("Using Apple MPS")
            else:
                device = torch.device("cpu")
                print("GPU requested but not available, using CPU")
        else:
            device = torch.device("cpu")
            print("Using CPU (GPU disabled)")
        return device

    def _init_model(self):
        """Initialize model based on config."""
        model_name = self.config.model.name

        if model_name.startswith("florence"):
            from model.florence import create_florence2_model, Florence2ForOCR

            size = "large" if "large" in model_name else "base"
            adapter = create_florence2_model(
                model_size=size,
                use_lora=True,
                lora_rank=self.config.model.lora_config.r,
                device=str(self.device)
            )
            self.model = Florence2ForOCR(adapter)
            self.tokenizer = adapter.processor.tokenizer

        elif model_name.startswith("qwen"):
            from model.qwen_vl import create_qwen2vl_model, Qwen2VLForOCR

            size = "7B" if "7b" in model_name.lower() else "2B"
            adapter = create_qwen2vl_model(
                model_size=size,
                use_lora=True,
                lora_rank=self.config.model.lora_config.r,
                device=str(self.device)
            )
            self.model = Qwen2VLForOCR(adapter)
            self.tokenizer = adapter.processor.tokenizer

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def setup_data(self, train_dir: str | Path, val_dir: str | Path = None):
        """Setup data loaders."""
        transforms = get_training_transforms(self.config.model.image_size)

        self.curriculum_loader = CurriculumDataLoader(
            data_dir=train_dir,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.dataloader_num_workers,
            transform=transforms,
            tokenizer=self.tokenizer,
            stage_epochs={
                "marker_recognition": (0, self.config.training.stage1_epochs),
                "marker_to_text": (self.config.training.stage1_epochs,
                                   self.config.training.stage1_epochs + self.config.training.stage2_epochs),
                "direct_ocr": (self.config.training.stage1_epochs + self.config.training.stage2_epochs,
                              self.config.training.num_epochs)
            }
        )

        if val_dir:
            val_dataset = HebLLMDataset(
                val_dir,
                stage="direct_ocr",
                transform=transforms,
                tokenizer=self.tokenizer
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=self.config.training.dataloader_num_workers,
                collate_fn=collate_fn
            )

    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        model = self.model.model if hasattr(self.model, 'model') else self.model

        # Filter trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )

        total_steps = (
            self.config.training.num_epochs *
            len(self.curriculum_loader.get_loader(0)) //
            self.config.training.gradient_accumulation_steps
        )

        warmup_steps = int(total_steps * self.config.training.warmup_ratio)

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.training.learning_rate * 0.1
        )

        # Mixed precision (only for CUDA)
        self.use_amp = self.config.training.fp16 and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Returns:
            Average loss for the epoch
        """
        self.model.train() if hasattr(self.model, 'train') else None

        # Get current stage and data loader
        stage = self.curriculum.get_stage(epoch)
        train_loader = self.curriculum_loader.get_loader(epoch)

        print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
        print(f"Training stage: {stage}")

        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            images = batch["images"].to(self.device)
            prompts = batch["prompts"]
            targets = batch["targets"]

            # Forward pass with mixed precision (CUDA only)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images, prompts, targets, stage=stage)
                loss = outputs["loss"]
                loss = loss / self.config.training.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.config.training.gradient_accumulation_steps
            num_batches += 1

            # Logging
            if batch_idx % self.config.training.logging_steps == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Step {batch_idx}/{len(train_loader)} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        avg_epoch_loss = total_loss / num_batches
        return avg_epoch_loss

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on validation set.

        Returns:
            Dict with evaluation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval() if hasattr(self.model, 'eval') else None

        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            prompts = batch["prompts"]
            targets = batch["targets"]

            outputs = self.model(images, prompts, targets, stage="direct_ocr")
            total_loss += outputs["loss"].item()
            num_batches += 1

        return {
            "val_loss": total_loss / num_batches if num_batches > 0 else 0
        }

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "loss": loss,
            "config": {
                "model_name": self.config.model.name,
                "model_id": self.config.model.model_id
            }
        }

        # Save checkpoint metadata
        meta_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.json"
        with open(meta_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        # Save model weights
        if hasattr(self.model, 'adapter'):
            self.model.adapter.save_pretrained(
                self.checkpoint_dir / f"model_epoch_{epoch}"
            )
        else:
            model_path = self.checkpoint_dir / f"model_epoch_{epoch}.pt"
            torch.save(self.model.state_dict(), model_path)

        if is_best:
            if hasattr(self.model, 'adapter'):
                self.model.adapter.save_pretrained(self.output_dir / "best_model")
            else:
                torch.save(
                    self.model.state_dict(),
                    self.output_dir / "best_model.pt"
                )

        print(f"Checkpoint saved: epoch {epoch}, loss {loss:.4f}")

    def train(self):
        """Run full training loop."""
        print(f"Starting training: {self.config.experiment_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"Total epochs: {self.config.training.num_epochs}")
        print("=" * 50)

        self.setup_optimizer()

        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(epoch)

            # Log
            log_entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "stage": self.curriculum.get_stage(epoch),
                "lr": self.optimizer.param_groups[0]["lr"]
            }

            # Evaluate
            if self.val_loader and (epoch + 1) % self.config.training.eval_steps == 0:
                eval_metrics = self.evaluate()
                log_entry.update(eval_metrics)

            self.train_log.append(log_entry)

            # Save checkpoint
            is_best = train_loss < self.best_loss
            if is_best:
                self.best_loss = train_loss

            if (epoch + 1) % (self.config.training.save_steps // len(self.curriculum_loader.get_loader(0)) or 1) == 0:
                self.save_checkpoint(epoch, train_loss, is_best)

            print(f"Epoch {epoch + 1} complete | Train Loss: {train_loss:.4f}")

        # Save final model
        self.save_checkpoint(self.config.training.num_epochs - 1, train_loss, train_loss < self.best_loss)

        # Save training log
        log_path = self.output_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.train_log, f, indent=2)

        print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
        print(f"Model saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train HebLLM model")

    # Model
    parser.add_argument("--model", default="florence2",
                        choices=["florence2", "florence2-large", "qwen2-vl-2b", "paligemma-3b", "moondream2"],
                        help="Model to train")

    # Data
    parser.add_argument("--train-data", required=True, help="Training data directory")
    parser.add_argument("--val-data", default=None, help="Validation data directory")

    # Training
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch-size * gradient-accumulation)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")

    # Output
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--name", default=None, help="Experiment name")

    # Curriculum
    parser.add_argument("--stage1-epochs", type=int, default=5, help="Marker recognition epochs")
    parser.add_argument("--stage2-epochs", type=int, default=10, help="Marker-to-text epochs")

    # Device
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="Use GPU if available (default: enabled)")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu",
                        help="Disable GPU, use CPU only")

    args = parser.parse_args()

    # Create config
    experiment_name = args.name or f"hebllm_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    model_config = get_model_config(args.model)
    model_config.lora_config.r = args.lora_rank

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.epochs - args.stage1_epochs - args.stage2_epochs
    )

    config = HebLLMConfig(
        model=model_config,
        training=training_config,
        data_dir=args.train_data,
        output_dir=args.output,
        experiment_name=experiment_name
    )

    # Create trainer
    trainer = HebLLMTrainer(config, use_gpu=args.gpu)
    trainer.setup_data(args.train_data, args.val_data)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
