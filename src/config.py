"""
Configuration module for the JDC system.

This module contains all configuration classes using Pydantic for validation.
All hyperparameters and settings are centralized here for easy management.
"""

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ModelType(str, Enum):
    """Supported base model types."""

    LLAMA3_8B_UNSLOSH = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    LLAMA3_8B_META = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"


class QuantizationType(str, Enum):
    """Supported quantization types."""

    BIT_4 = "4bit"
    BIT_8 = "8bit"
    NONE = "none"


class LoRAConfig(BaseModel):
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning.

    Attributes:
        r: Rank of the low-rank matrices. Higher = more parameters.
        lora_alpha: Scaling factor for LoRA updates.
        target_modules: Model modules to apply LoRA to.
        lora_dropout: Dropout probability for LoRA layers.
        bias: How to handle bias terms.
        task_type: Type of task (causal language modeling for text generation).
        modules_to_save: Modules to fully fine-tune and save (embeddings).
    """

    r: int = Field(default=16, ge=1, le=128, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA alpha scaling")

    target_modules: list[str] = Field(
        default_factory=lambda: [
            # Attention projections
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # MLP/Feed-Forward layers (CRITICAL for reasoning and knowledge storage)
            "gate_proj",  # Gating mechanism
            "up_proj",  # Up-projection
            "down_proj",  # Down-projection
        ],
        description="Modules to apply LoRA (attention + MLP for full capacity)",
    )

    modules_to_save: Optional[list[str]] = Field(
        default=None,
        description="Modules to fully fine-tune (e.g. 'embed_tokens', 'lm_head'). Only add these if adding NEW special tokens to the tokenizer; otherwise leave None to conserve memory.",
    )

    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"


class QuantizationConfig(BaseModel):
    """Configuration for model quantization using bitsandbytes.

    Attributes:
        quantization_type: Type of quantization (4-bit, 8-bit, or none).
        bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization.
        bnb_4bit_quant_type: Quantization type (fp4 or nf4).
        bnb_4bit_use_double_quant: Enable nested quantization.
    """

    quantization_type: QuantizationType = QuantizationType.BIT_4
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: Literal["fp4", "nf4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True


class TrainingHyperparameters(BaseModel):
    """Hyperparameters for the fine-tuning process.

    Attributes:
        num_epochs: Number of training epochs.
        batch_size: Batch size per device.
        gradient_accumulation_steps: Steps to accumulate gradients.
        learning_rate: Initial learning rate.
        max_seq_length: Maximum sequence length for tokenization.
        warmup_ratio: Fraction of steps for learning rate warmup.
        logging_steps: Steps between logging updates.
        save_steps: Steps between model checkpoints.
        eval_steps: Steps between evaluations.
        fp16: Enable mixed precision training (FP16).
        optim: Optimizer to use.
    """

    num_epochs: int = Field(default=3, ge=1, le=20)
    batch_size: int = Field(default=4, ge=1, le=32)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    max_seq_length: int = Field(default=2048, ge=512, le=8192)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.3)
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    fp16: bool = True
    optim: str = "paged_adamw_32bit"


class InferenceConfig(BaseModel):
    """Configuration for inference/generation.

    Attributes:
        temperature: Sampling temperature (0 = deterministic).
        max_new_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.
        do_sample: Whether to sample or use greedy decoding.
        repetition_penalty: Penalty for repeating tokens.
    """

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    do_sample: bool = False  # False for deterministic with temp=0
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class SystemConfig(BaseModel):
    """Overall system configuration.

    Attributes:
        model_type: Which base model to use.
        lora_config: LoRA configuration.
        quantization_config: Quantization configuration.
        training_hyperparameters: Training hyperparameters.
        inference_config: Inference configuration.
        output_dir: Directory for saving models and outputs.
        data_dir: Directory containing training/test data.
        cache_dir: Directory for caching downloaded models.
        seed: Random seed for reproducibility.
    """

    hf_token: Optional[str] = Field(
        default=None,
        description="Hugging Face API token for accessing gated repositories like Llama 3.",
    )
    model_type: ModelType = ModelType.LLAMA3_8B_META
    lora_config: LoRAConfig = Field(default_factory=LoRAConfig)
    quantization_config: QuantizationConfig = Field(default_factory=QuantizationConfig)
    training_hyperparameters: TrainingHyperparameters = Field(
        default_factory=TrainingHyperparameters
    )
    inference_config: InferenceConfig = Field(default_factory=InferenceConfig)

    output_dir: Path = Field(default=Path("./outputs"))
    data_dir: Path = Field(default=Path("/content/drive/MyDrive/autism_jdc_dataset/"))
    cache_dir: Path = Field(default=Path("./cache"))
    seed: int = Field(default=42, ge=0)

    @field_validator("output_dir", "data_dir", "cache_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """Ensure directory paths are Path objects."""
        return Path(v) if isinstance(v, str) else v


class KnowledgeBaseConfig(BaseModel):
    """Configuration for the symbolic knowledge base.

    The knowledge base contains principles for neurodiversity-aware
    classification of ableist language.
    """

    principles: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            "P0": {
                "name": "Neutral Language",
                "definition": (
                    "Language that does not pathologize, stigmatize, or "
                    "stereotype neurodivergent individuals. It respects "
                    "neurological diversity as a natural human variation."
                ),
            },
            "P1": {
                "name": "Pathologizing Language",
                "definition": (
                    "Language that frames neurodivergent traits as deficits, "
                    "diseases, or abnormalities requiring cure or correction."
                ),
            },
            "P2": {
                "name": "Dehumanizing Metaphors",
                "definition": (
                    "Use of metaphors that compare neurodivergent people to "
                    "objects, animals, or broken machinery, denying their agency."
                ),
            },
            "P3": {
                "name": "Stereotyping",
                "definition": (
                    "Overgeneralized assumptions about abilities, behaviors, "
                    "or characteristics of neurodivergent individuals."
                ),
            },
            "P4": {
                "name": "Exclusionary Language",
                "definition": (
                    "Language that explicitly or implicitly excludes "
                    "neurodivergent people from participation in society."
                ),
            },
        }
    )

    def get_principle(self, principle_id: str) -> Optional[dict[str, str]]:
        """Get a principle by ID."""
        return self.principles.get(principle_id)

    def get_all_principles_text(self) -> str:
        """Format all principles as text for prompt injection."""
        lines = ["KNOWLEDGE BASE - Neurodiversity Principles:"]
        for pid, principle in self.principles.items():
            lines.append(f"\n{pid}: {principle['name']}")
            lines.append(f"Definition: {principle['definition']}")
        return "\n".join(lines)
