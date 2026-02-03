"""
Model and tokenizer loading with QLoRA configuration.
Handles edge cases like missing PAD tokens in Llama/Mistral models.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Handles loading and configuration of LLMs with QLoRA.
    
    Key responsibilities:
    - Load model with 4-bit quantization (BitsAndBytes)
    - Attach LoRA adapters (PEFT)
    - Handle missing PAD tokens in Llama/Mistral
    - Configure padding side appropriately for training vs inference
    """
    
    @staticmethod
    def load_tokenizer(
        model_name: str,
        padding_side: str = "right",
        trust_remote_code: bool = False
    ) -> AutoTokenizer:
        """
        Load tokenizer with proper configuration.
        
        Args:
            model_name: Hugging Face model identifier
            padding_side: 'right' for training (SFT), 'left' for batched generation
            trust_remote_code: Allow custom tokenizer code
            
        Returns:
            Configured tokenizer
            
        Note:
            Llama 3 and Mistral models often lack a PAD token. We set it to EOS token
            to avoid index errors during batched processing. Setting padding_side='right'
            is crucial for causal LM training to ensure labels are aligned correctly.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=True,  # Use fast tokenizer when available
        )
        
        # CRITICAL: Handle missing PAD token (common in Llama/Mistral)
        if tokenizer.pad_token is None:
            logger.warning(
                f"Model {model_name} lacks PAD token. Setting pad_token = eos_token."
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Set padding side
        # - RIGHT for training: ensures labels align with tokens in causal LM
        # - LEFT for inference: ensures generation starts from end of prompt
        tokenizer.padding_side = padding_side
        logger.info(f"Tokenizer padding side set to: {padding_side}")
        
        return tokenizer
    
    @staticmethod
    def create_bnb_config(
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
    ) -> BitsAndBytesConfig:
        """
        Create BitsAndBytes quantization configuration.
        
        Args:
            load_in_4bit: Enable 4-bit quantization
            bnb_4bit_compute_dtype: Compute dtype (float16 recommended for balance)
            bnb_4bit_quant_type: Quantization type (nf4 = NormalFloat4)
            bnb_4bit_use_double_quant: Quantize quantization constants (saves memory)
            
        Returns:
            BitsAndBytesConfig for model loading
        """
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        
        return BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )
    
    @staticmethod
    def load_base_model(
        model_name: str,
        bnb_config: BitsAndBytesConfig,
        device_map: str = "auto",
        trust_remote_code: bool = False,
    ) -> AutoModelForCausalLM:
        """
        Load base model with quantization.
        
        Args:
            model_name: Hugging Face model identifier
            bnb_config: BitsAndBytes configuration
            device_map: Device placement strategy
            trust_remote_code: Allow custom model code
            
        Returns:
            Quantized base model
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16,  # Use FP16 for efficiency
        )
        
        # Disable cache for training (required for gradient checkpointing)
        model.config.use_cache = False
        
        # Enable gradient checkpointing for memory efficiency
        model.config.pretraining_tp = 1  # Avoid tensor parallelism issues
        
        logger.info(f"Loaded base model: {model_name}")
        logger.info(f"Model device map: {model.hf_device_map}")
        
        return model
    
    @staticmethod
    def create_lora_config(
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
        target_modules: Optional[list[str]] = None,
    ) -> LoraConfig:
        """
        Create LoRA configuration.
        
        Args:
            r: LoRA rank (dimension of low-rank matrices)
            lora_alpha: Scaling factor (typically 2*r)
            lora_dropout: Dropout probability
            bias: Bias training strategy ("none", "all", "lora_only")
            task_type: Task type for PEFT
            target_modules: Modules to apply LoRA (auto-detected if None)
            
        Returns:
            LoraConfig for PEFT
            
        Note:
            Default target_modules cover attention and FFN layers in Llama/Mistral.
            Adjust if using a different architecture.
        """
        if target_modules is None:
            # Default for Llama 3 and Mistral
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj",      # FFN
            ]
        
        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type,
            target_modules=target_modules,
        )
    
    @staticmethod
    def setup_model_for_training(
        model: AutoModelForCausalLM,
        lora_config: LoraConfig,
        gradient_checkpointing: bool = True,
    ) -> AutoModelForCausalLM:
        """
        Prepare model for k-bit training and attach LoRA adapters.
        
        Args:
            model: Base quantized model
            lora_config: LoRA configuration
            gradient_checkpointing: Enable gradient checkpointing
            
        Returns:
            PEFT model ready for training
        """
        # Prepare model for k-bit training (enables gradient computation)
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing
        )
        
        # Enable gradient checkpointing
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Attach LoRA adapters
        model = get_peft_model(model, lora_config)
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / total_params
        
        logger.info(f"Trainable params: {trainable_params:,} ({trainable_percent:.2f}%)")
        logger.info(f"Total params: {total_params:,}")
        
        return model
    
    @staticmethod
    def load_model_for_training(
        model_name: str,
        bnb_config: BitsAndBytesConfig,
        lora_config: LoraConfig,
        device_map: str = "auto",
        gradient_checkpointing: bool = True,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Complete pipeline: Load model and tokenizer for training.
        
        Returns:
            (model, tokenizer) ready for training
        """
        # Load tokenizer (padding_side='right' for training)
        tokenizer = ModelLoader.load_tokenizer(
            model_name,
            padding_side="right"
        )
        
        # Load base model
        model = ModelLoader.load_base_model(
            model_name,
            bnb_config,
            device_map
        )
        
        # Setup for training
        model = ModelLoader.setup_model_for_training(
            model,
            lora_config,
            gradient_checkpointing
        )
        
        return model, tokenizer
    
    @staticmethod
    def load_model_for_inference(
        model_name: str,
        adapter_path: str,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load fine-tuned model for inference.
        
        Args:
            model_name: Base model identifier
            adapter_path: Path to saved LoRA adapters
            load_in_4bit: Use 4-bit quantization
            device_map: Device placement strategy
            
        Returns:
            (model, tokenizer) ready for inference
            
        Note:
            padding_side is set to 'left' for batched generation to ensure
            all sequences end at the same position, allowing the model to
            generate from the end of each prompt simultaneously.
        """
        # Load tokenizer (padding_side='left' for batched generation)
        tokenizer = ModelLoader.load_tokenizer(
            model_name,
            padding_side="left"
        )
        
        # Create quantization config
        bnb_config = ModelLoader.create_bnb_config(
            load_in_4bit=load_in_4bit
        ) if load_in_4bit else None
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            device_map=device_map,
        )
        
        # Merge adapters for faster inference (optional)
        # model = model.merge_and_unload()  # Uncomment if you want to merge
        
        model.eval()  # Set to evaluation mode
        logger.info(f"Loaded fine-tuned model from: {adapter_path}")
        
        return model, tokenizer
