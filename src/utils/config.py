"""
Configuration schemas for the JDC framework using Pydantic.
Provides type safety and validation for all hyperparameters.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, validator


class QuantizationConfig(BaseModel):
    """Configuration for model quantization."""
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


class LoRAConfig(BaseModel):
    """Configuration for LoRA adapters."""
    r: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA alpha scaling")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    bias: str = Field(default="none", description="Bias training strategy")
    task_type: str = Field(default="CAUSAL_LM")
    target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="Modules to apply LoRA to"
    )


class TrainingConfig(BaseModel):
    """Main training configuration."""
    
    # Model settings
    model_name: str = Field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        description="Hugging Face model identifier"
    )
    
    # Data settings
    train_data_path: str = Field(description="Path to training data JSON/JSONL")
    val_data_path: Optional[str] = Field(default=None, description="Path to validation data")
    max_seq_length: int = Field(default=2048, ge=128)
    
    # Training hyperparameters
    num_train_epochs: int = Field(default=3, ge=1)
    per_device_train_batch_size: int = Field(default=4, ge=1)
    per_device_eval_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    lr_scheduler_type: str = Field(default="cosine")
    weight_decay: float = Field(default=0.01, ge=0.0)
    
    # Optimization
    optim: str = Field(default="paged_adamw_32bit")
    gradient_checkpointing: bool = Field(default=True, description="Save VRAM")
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    
    # Logging and saving
    output_dir: str = Field(default="./checkpoints")
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    save_total_limit: int = Field(default=3, ge=1)
    
    # WandB (optional)
    use_wandb: bool = Field(default=False)
    wandb_project: Optional[str] = Field(default=None)
    wandb_run_name: Optional[str] = Field(default=None)
    
    # Configs
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    
    # Device
    device_map: str = Field(default="auto")
    
    @validator("model_name")
    def validate_model(cls, v):
        """Ensure supported model."""
        supported = ["llama", "mistral"]
        if not any(s in v.lower() for s in supported):
            raise ValueError(f"Model must be Llama or Mistral. Got: {v}")
        return v


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""
    
    # Model settings
    model_name: str = Field(description="Base model identifier")
    adapter_path: str = Field(description="Path to fine-tuned LoRA adapters")
    
    # Data settings
    test_data_path: str = Field(description="Path to test data JSON/JSONL")
    max_seq_length: int = Field(default=2048, ge=128)
    
    # Generation settings
    batch_size: int = Field(default=8, ge=1)
    max_new_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Lower for deterministic")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    do_sample: bool = Field(default=True)
    
    # Parsing settings
    max_parse_retries: int = Field(default=3, ge=1)
    fallback_principle: str = Field(default="P_ERR", description="Fallback on parse failure")
    
    # Output settings
    output_dir: str = Field(default="./evaluation_results")
    save_predictions: bool = Field(default=True)
    save_disagreements: bool = Field(default=True)
    
    # Quantization (usually same as training)
    load_in_4bit: bool = Field(default=True)
    device_map: str = Field(default="auto")
    
    # Principle mapping
    ableist_principles: list[str] = Field(
        default=["P1", "P2", "P3", "P4"],
        description="Principles mapped to ableist (1)"
    )
    non_ableist_principles: list[str] = Field(
        default=["P0"],
        description="Principles mapped to non-ableist (0)"
    )


def load_config_from_yaml(config_path: str, config_class: type[BaseModel]) -> BaseModel:
    """Load and validate configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_class(**config_dict)
