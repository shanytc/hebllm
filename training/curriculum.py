#!/usr/bin/env python3
"""
Curriculum Learning Scheduler for HebLLM

Implements the "Marker Curriculum" training strategy:
1. Marker Recognition: Model learns to identify HebMark markers
2. Marker to Text: Model learns marker-Hebrew associations
3. Direct OCR: End-to-end Hebrew OCR without markers

This curriculum helps the model gradually internalize Hebrew
character patterns through the HebMark system.
"""

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class StageConfig:
    """Configuration for a curriculum stage."""
    name: Literal["marker_recognition", "marker_to_text", "direct_ocr"]
    start_epoch: int
    end_epoch: int
    learning_rate_multiplier: float = 1.0
    loss_weight: float = 1.0

    @property
    def duration(self) -> int:
        """Number of epochs in this stage."""
        return self.end_epoch - self.start_epoch

    def contains_epoch(self, epoch: int) -> bool:
        """Check if epoch falls within this stage."""
        return self.start_epoch <= epoch < self.end_epoch


class CurriculumScheduler:
    """
    Manages curriculum learning progression.

    The curriculum follows three stages:

    Stage 1 - Marker Recognition (epochs 1-5):
        - Input: Marked PDF image
        - Target: Marker IDs in reading order (◆00 ◆01 ◆02...)
        - Goal: Model learns to visually identify markers

    Stage 2 - Marker to Text (epochs 6-15):
        - Input: Marked PDF image + marker mapping in prompt
        - Target: Full Hebrew text
        - Goal: Model learns marker → Hebrew association

    Stage 3 - Direct OCR (epochs 16-30):
        - Input: Original (unmarked) PDF image
        - Target: Full Hebrew text
        - Goal: Model applies learned Hebrew patterns
    """

    DEFAULT_STAGES = [
        StageConfig("marker_recognition", 0, 5, learning_rate_multiplier=1.0),
        StageConfig("marker_to_text", 5, 15, learning_rate_multiplier=0.8),
        StageConfig("direct_ocr", 15, 30, learning_rate_multiplier=0.5),
    ]

    def __init__(self,
                 total_epochs: int = 30,
                 stage_configs: list[StageConfig] = None):
        """
        Initialize curriculum scheduler.

        Args:
            total_epochs: Total training epochs
            stage_configs: Custom stage configurations
        """
        self.total_epochs = total_epochs
        self.stages = stage_configs or self._scale_default_stages(total_epochs)
        self._validate_stages()

    def _scale_default_stages(self, total_epochs: int) -> list[StageConfig]:
        """Scale default stages to fit total epochs."""
        # Default distribution: 15% stage 1, 35% stage 2, 50% stage 3
        s1_end = int(total_epochs * 0.15)
        s2_end = int(total_epochs * 0.50)

        return [
            StageConfig("marker_recognition", 0, max(1, s1_end)),
            StageConfig("marker_to_text", max(1, s1_end), max(2, s2_end)),
            StageConfig("direct_ocr", max(2, s2_end), total_epochs),
        ]

    def _validate_stages(self):
        """Validate stage configuration."""
        # Check for gaps or overlaps
        sorted_stages = sorted(self.stages, key=lambda s: s.start_epoch)

        for i, stage in enumerate(sorted_stages):
            if i == 0 and stage.start_epoch != 0:
                raise ValueError(f"First stage must start at epoch 0, got {stage.start_epoch}")

            if i > 0:
                prev_stage = sorted_stages[i - 1]
                if stage.start_epoch != prev_stage.end_epoch:
                    raise ValueError(
                        f"Stage gap/overlap: {prev_stage.name} ends at {prev_stage.end_epoch}, "
                        f"{stage.name} starts at {stage.start_epoch}"
                    )

        # Check last stage covers total epochs
        last_stage = sorted_stages[-1]
        if last_stage.end_epoch < self.total_epochs:
            raise ValueError(
                f"Last stage ends at {last_stage.end_epoch} but total epochs is {self.total_epochs}"
            )

    def get_stage(self, epoch: int) -> str:
        """Get the training stage for given epoch."""
        for stage in self.stages:
            if stage.contains_epoch(epoch):
                return stage.name
        return "direct_ocr"  # Fallback

    def get_stage_config(self, epoch: int) -> StageConfig:
        """Get the full stage config for given epoch."""
        for stage in self.stages:
            if stage.contains_epoch(epoch):
                return stage
        return self.stages[-1]  # Fallback to last stage

    def get_lr_multiplier(self, epoch: int) -> float:
        """Get learning rate multiplier for epoch."""
        stage = self.get_stage_config(epoch)
        return stage.learning_rate_multiplier

    def get_loss_weight(self, epoch: int) -> float:
        """Get loss weight for epoch."""
        stage = self.get_stage_config(epoch)
        return stage.loss_weight

    def get_progress(self, epoch: int) -> dict:
        """Get curriculum progress information."""
        stage = self.get_stage_config(epoch)
        stage_progress = (epoch - stage.start_epoch) / stage.duration
        overall_progress = epoch / self.total_epochs

        return {
            "stage": stage.name,
            "stage_epoch": epoch - stage.start_epoch + 1,
            "stage_total": stage.duration,
            "stage_progress": stage_progress,
            "overall_epoch": epoch + 1,
            "overall_total": self.total_epochs,
            "overall_progress": overall_progress
        }

    def should_use_markers(self, epoch: int) -> bool:
        """Check if current stage uses marked images."""
        stage = self.get_stage(epoch)
        return stage in ("marker_recognition", "marker_to_text")

    def should_include_mapping(self, epoch: int) -> bool:
        """Check if current stage includes marker mapping in prompt."""
        stage = self.get_stage(epoch)
        return stage == "marker_to_text"

    def format_target(self,
                      epoch: int,
                      markers: dict,
                      ground_truth: str) -> str:
        """Format the target based on current stage.

        Args:
            epoch: Current epoch
            markers: Marker ID to Hebrew text mapping
            ground_truth: Full ground truth text

        Returns:
            Formatted target string for training
        """
        stage = self.get_stage(epoch)

        if stage == "marker_recognition":
            # Target: marker IDs in reading order
            if not markers:
                return ""
            sorted_markers = sorted(
                markers.items(),
                key=lambda x: (
                    x[1].get("page", 0),
                    x[1].get("line", 0),
                    x[1].get("word", 0)
                )
            )
            return " ".join(f"◆{mid}" for mid, _ in sorted_markers)

        else:
            # Stages 2 and 3: full text
            return ground_truth

    def format_prompt(self,
                      epoch: int,
                      base_prompt: str,
                      markers: dict) -> str:
        """Format the prompt based on current stage.

        Args:
            epoch: Current epoch
            base_prompt: Base instruction prompt
            markers: Marker mapping

        Returns:
            Formatted prompt string
        """
        stage = self.get_stage(epoch)

        if stage == "marker_recognition":
            return "List all visible markers (◆XX) in reading order, separated by spaces."

        elif stage == "marker_to_text":
            # Include marker mapping in prompt
            if markers:
                mapping_str = ", ".join(
                    f"◆{mid}=\"{entry['hebrew_text']}\""
                    for mid, entry in markers.items()
                )
                return f"Markers: {mapping_str}. Transcribe the full document text."
            return base_prompt

        else:  # direct_ocr
            return base_prompt or "Transcribe all text from this document image."

    def __repr__(self) -> str:
        stages_str = ", ".join(
            f"{s.name}({s.start_epoch}-{s.end_epoch})"
            for s in self.stages
        )
        return f"CurriculumScheduler(epochs={self.total_epochs}, stages=[{stages_str}])"


class AdaptiveCurriculum(CurriculumScheduler):
    """
    Adaptive curriculum that adjusts based on training metrics.

    Can extend or shorten stages based on loss convergence.
    """

    def __init__(self,
                 total_epochs: int = 30,
                 stage_configs: list[StageConfig] = None,
                 convergence_threshold: float = 0.01,
                 patience: int = 3):
        """
        Initialize adaptive curriculum.

        Args:
            total_epochs: Total training epochs
            stage_configs: Custom stage configurations
            convergence_threshold: Loss change threshold for convergence
            patience: Epochs to wait before advancing stage
        """
        super().__init__(total_epochs, stage_configs)

        self.convergence_threshold = convergence_threshold
        self.patience = patience

        # Tracking state
        self.loss_history = []
        self.stage_losses = {s.name: [] for s in self.stages}
        self.forced_stage = None
        self.convergence_counter = 0

    def update(self, epoch: int, loss: float) -> dict:
        """Update curriculum based on training metrics.

        Args:
            epoch: Current epoch
            loss: Training loss

        Returns:
            Dict with curriculum state info
        """
        self.loss_history.append(loss)
        stage = self.get_stage(epoch)
        self.stage_losses[stage].append(loss)

        # Check for convergence
        if len(self.stage_losses[stage]) >= 2:
            recent_losses = self.stage_losses[stage][-self.patience:]
            if len(recent_losses) >= self.patience:
                loss_change = abs(recent_losses[-1] - recent_losses[0]) / (recent_losses[0] + 1e-8)

                if loss_change < self.convergence_threshold:
                    self.convergence_counter += 1
                else:
                    self.convergence_counter = 0

        return {
            "stage": stage,
            "converged": self.convergence_counter >= self.patience,
            "stage_losses": self.stage_losses[stage][-5:],
            "convergence_counter": self.convergence_counter
        }

    def should_advance(self, epoch: int) -> bool:
        """Check if curriculum should advance to next stage early."""
        if self.convergence_counter >= self.patience:
            current_stage = self.get_stage_config(epoch)
            stage_progress = (epoch - current_stage.start_epoch) / current_stage.duration

            # Advance if at least 50% through current stage and converged
            return stage_progress >= 0.5

        return False


def create_curriculum(
    epochs: int,
    stage1_pct: float = 0.15,
    stage2_pct: float = 0.35,
    stage3_pct: float = 0.50
) -> CurriculumScheduler:
    """
    Factory function to create curriculum scheduler.

    Args:
        epochs: Total training epochs
        stage1_pct: Percentage for marker recognition
        stage2_pct: Percentage for marker-to-text
        stage3_pct: Percentage for direct OCR

    Returns:
        Configured CurriculumScheduler
    """
    assert abs(stage1_pct + stage2_pct + stage3_pct - 1.0) < 0.01, "Stage percentages must sum to 1"

    s1_end = max(1, int(epochs * stage1_pct))
    s2_end = max(s1_end + 1, int(epochs * (stage1_pct + stage2_pct)))

    stages = [
        StageConfig("marker_recognition", 0, s1_end),
        StageConfig("marker_to_text", s1_end, s2_end),
        StageConfig("direct_ocr", s2_end, epochs),
    ]

    return CurriculumScheduler(epochs, stages)


if __name__ == "__main__":
    # Test curriculum scheduler
    curriculum = create_curriculum(30)
    print(curriculum)

    print("\nCurriculum progression:")
    for epoch in range(30):
        progress = curriculum.get_progress(epoch)
        if progress["stage_epoch"] == 1 or epoch == 29:
            print(f"  Epoch {epoch + 1}: {progress['stage']} "
                  f"(stage {progress['stage_epoch']}/{progress['stage_total']})")
