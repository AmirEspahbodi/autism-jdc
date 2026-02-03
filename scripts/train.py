"""
Training script for JDC Framework (Step 3: Fine-Tuning).

Usage:
    python scripts/train.py --config configs/train_config.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import TrainingArguments
from trl import SFTTrainer

from utils.config import TrainingConfig, load_config_from_yaml
from modeling.model_loader import ModelLoader
from data.dataset import JDCDataset, formatting_func, validate_data_format

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_training_args(config: TrainingConfig) -> TrainingArguments:
    """
    Create Hugging Face TrainingArguments from config.
    
    Args:
        config: Training configuration
        
    Returns:
        TrainingArguments for Trainer
    """
    # Setup WandB if enabled
    report_to = []
    if config.use_wandb:
        report_to.append("wandb")
        if config.wandb_project:
            os.environ["WANDB_PROJECT"] = config.wandb_project
        if config.wandb_run_name:
            os.environ["WANDB_RUN_NAME"] = config.wandb_run_name
    
    training_args = TrainingArguments(
        # Output
        output_dir=config.output_dir,
        
        # Training hyperparameters
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Optimization
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        optim=config.optim,
        max_grad_norm=config.max_grad_norm,
        
        # Gradient checkpointing
        gradient_checkpointing=config.gradient_checkpointing,
        
        # Logging
        logging_steps=config.logging_steps,
        logging_first_step=True,
        
        # Saving
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_strategy="steps",
        
        # Evaluation
        evaluation_strategy="steps" if config.val_data_path else "no",
        eval_steps=config.eval_steps if config.val_data_path else None,
        
        # Performance
        fp16=torch.cuda.is_available(),  # Use FP16 if CUDA available
        bf16=False,  # Use BF16 if supported (TPU/A100)
        
        # Misc
        report_to=report_to,
        load_best_model_at_end=False,
        push_to_hub=False,
        remove_unused_columns=False,  # Important for SFTTrainer
    )
    
    return training_args


def train(config: TrainingConfig):
    """
    Main training function.
    
    Args:
        config: Training configuration
    """
    logger.info("=" * 60)
    logger.info("JDC FRAMEWORK - STEP 3: FINE-TUNING")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Training data: {config.train_data_path}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info("")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 1. Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    bnb_config = ModelLoader.create_bnb_config(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_compute_dtype=config.quantization.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
    )
    
    lora_config = ModelLoader.create_lora_config(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
        target_modules=config.lora.target_modules,
    )
    
    model, tokenizer = ModelLoader.load_model_for_training(
        model_name=config.model_name,
        bnb_config=bnb_config,
        lora_config=lora_config,
        device_map=config.device_map,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    
    # 2. Load and prepare datasets
    logger.info("Loading training data...")
    dataset_handler = JDCDataset(
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        use_chat_template=True
    )
    
    # Load and validate training data
    train_data = dataset_handler.load_data(config.train_data_path)
    validate_data_format(train_data)
    train_dataset = dataset_handler.prepare_dataset(train_data, for_training=True)
    
    # Load validation data if provided
    eval_dataset = None
    if config.val_data_path:
        logger.info("Loading validation data...")
        val_data = dataset_handler.load_data(config.val_data_path)
        validate_data_format(val_data)
        eval_dataset = dataset_handler.prepare_dataset(val_data, for_training=True)
    
    logger.info(f"Training examples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Validation examples: {len(eval_dataset)}")
    logger.info("")
    
    # 3. Setup training arguments
    training_args = setup_training_args(config)
    
    # 4. Create SFTTrainer
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        max_seq_length=config.max_seq_length,
        packing=False,  # Don't pack sequences (keep examples separate)
    )
    
    # 5. Train
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    # 6. Save final model
    logger.info("Saving final model...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info(f"Model saved to: {config.output_dir}")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train JDC model for ableism detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config_from_yaml(args.config, TrainingConfig)
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(config.model_dump_json(indent=2))
    logger.info("")
    
    # Start training
    train(config)


if __name__ == "__main__":
    main()
