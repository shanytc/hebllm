#!/usr/bin/env python3
"""
Model Configurations for HebLLM

Defines configurations for different VLM architectures suitable for
edge deployment with Hebrew OCR capabilities.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha (scaling)
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"  # MLP
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    num_epochs: int = 30

    # Curriculum stages
    stage1_epochs: int = 5  # Marker recognition
    stage2_epochs: int = 10  # Marker to text
    stage3_epochs: int = 15  # Direct OCR

    # Optimization
    optimizer: str = "adamw_torch"
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False

    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500

    # Data
    max_seq_length: int = 512
    dataloader_num_workers: int = 4


@dataclass
class ModelConfig:
    """Base model configuration."""
    name: str
    model_id: str
    image_size: tuple[int, int]
    max_length: int
    supports_hebrew: bool
    edge_compatible: bool
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)

    # Model-specific settings
    use_flash_attention: bool = False
    torch_dtype: str = "float16"
    device_map: str = "auto"


# =============================================================================
# Pre-configured Models
# =============================================================================

FLORENCE2_CONFIG = ModelConfig(
    name="florence2",
    model_id="microsoft/Florence-2-base",
    image_size=(768, 768),
    max_length=1024,
    supports_hebrew=False,  # Needs fine-tuning
    edge_compatible=True,
    lora_config=LoRAConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    )
)

FLORENCE2_LARGE_CONFIG = ModelConfig(
    name="florence2-large",
    model_id="microsoft/Florence-2-large",
    image_size=(768, 768),
    max_length=1024,
    supports_hebrew=False,
    edge_compatible=True,
    lora_config=LoRAConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    )
)

QWEN2_VL_2B_CONFIG = ModelConfig(
    name="qwen2-vl-2b",
    model_id="Qwen/Qwen2-VL-2B-Instruct",
    image_size=(896, 896),
    max_length=2048,
    supports_hebrew=True,  # Good multilingual support
    edge_compatible=True,
    lora_config=LoRAConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ),
    use_flash_attention=True
)

PALIGEMMA_3B_CONFIG = ModelConfig(
    name="paligemma-3b",
    model_id="google/paligemma-3b-pt-224",
    image_size=(224, 224),
    max_length=512,
    supports_hebrew=True,
    edge_compatible=False,  # 3B is larger
    lora_config=LoRAConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
)

MOONDREAM2_CONFIG = ModelConfig(
    name="moondream2",
    model_id="vikhyatk/moondream2",
    image_size=(378, 378),
    max_length=512,
    supports_hebrew=False,  # Less tested
    edge_compatible=True,
    lora_config=LoRAConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"]
    )
)


# Model registry
MODEL_CONFIGS = {
    "florence2": FLORENCE2_CONFIG,
    "florence2-large": FLORENCE2_LARGE_CONFIG,
    "qwen2-vl-2b": QWEN2_VL_2B_CONFIG,
    "paligemma-3b": PALIGEMMA_3B_CONFIG,
    "moondream2": MOONDREAM2_CONFIG,
}


def get_model_config(name: str) -> ModelConfig:
    """Get model configuration by name."""
    if name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return MODEL_CONFIGS[name]


def get_recommended_config(
    edge_only: bool = True,
    hebrew_support: bool = True
) -> ModelConfig:
    """Get recommended model based on requirements.

    Args:
        edge_only: Only consider edge-deployable models
        hebrew_support: Prefer models with existing Hebrew support

    Returns:
        Recommended ModelConfig
    """
    candidates = []

    for config in MODEL_CONFIGS.values():
        if edge_only and not config.edge_compatible:
            continue
        candidates.append(config)

    if not candidates:
        return FLORENCE2_CONFIG

    # Prefer Hebrew support if requested
    if hebrew_support:
        hebrew_candidates = [c for c in candidates if c.supports_hebrew]
        if hebrew_candidates:
            return hebrew_candidates[0]

    # Return smallest edge-compatible model
    return candidates[0]


@dataclass
class HebLLMConfig:
    """Complete configuration for HebLLM training."""
    model: ModelConfig
    training: TrainingConfig

    # Data paths
    data_dir: str = "./training_data"
    output_dir: str = "./output"

    # Experiment settings
    experiment_name: str = "hebllm"
    seed: int = 42

    @classmethod
    def from_preset(cls,
                    model_name: str = "florence2",
                    **kwargs) -> "HebLLMConfig":
        """Create config from preset model name."""
        model_config = get_model_config(model_name)
        training_config = TrainingConfig(**{
            k: v for k, v in kwargs.items()
            if hasattr(TrainingConfig, k)
        })

        return cls(
            model=model_config,
            training=training_config,
            **{k: v for k, v in kwargs.items()
               if k in ["data_dir", "output_dir", "experiment_name", "seed"]}
        )
