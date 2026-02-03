"""
Infrastructure LLM module - Concrete implementations for LLM training and inference.

This module contains adapters that implement the domain interfaces using
HuggingFace Transformers, PEFT (LoRA), and bitsandbytes for quantization.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from trl import SFTTrainer

from src.config import SystemConfig
from src.domain import (
    InferenceEngine,
    InferenceError,
    LabeledExample,
    LLMTrainer,
    TrainingError,
)


class PromptTemplate:
    """Handles prompt formatting for the JDC system."""

    SYSTEM_INSTRUCTION = """You are an expert in neurodiversity-aware language analysis. Your task is to analyze text for ableist language using the provided knowledge base of principles.

You must output ONLY valid JSON in the following format:
{
  "principle_id": "P0" or "P1" or "P2" or "P3" or "P4",
  "justification_text": "explanation of your reasoning",
  "evidence_quote": "relevant quote from the input text"
}

Do not include any preamble, explanation, or markdown formatting. Output ONLY the JSON object."""

    @staticmethod
    def build_training_prompt(example: LabeledExample, kb_text: str) -> dict[str, str]:
        """Build a training prompt with input and expected output.

        Args:
            example: Labeled training example.
            kb_text: Knowledge base as formatted text.

        Returns:
            Dictionary with 'input' and 'output' keys.
        """
        # Build the input text
        context_parts = []
        if example.context_before:
            context_parts.append(f"Previous context: {example.context_before}")
        context_parts.append(f"Target sentence: {example.sentence}")
        if example.context_after:
            context_parts.append(f"Following context: {example.context_after}")

        context_text = "\n".join(context_parts)

        user_message = f"""{kb_text}

{context_text}

Analyze the target sentence for ableist language. Output your response as JSON."""

        # Build the expected output
        justification_json = {
            "principle_id": example.ground_truth_justification.principle_id,
            "justification_text": example.ground_truth_justification.justification_text,
            "evidence_quote": example.ground_truth_justification.evidence_quote,
        }

        return {
            "input": f"{PromptTemplate.SYSTEM_INSTRUCTION}\n\n{user_message}",
            "output": json.dumps(justification_json, indent=2),
        }

    @staticmethod
    def build_inference_prompt(
        sentence: str,
        kb_text: str,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
    ) -> str:
        """Build an inference prompt (input only).

        Args:
            sentence: Target sentence to analyze.
            kb_text: Knowledge base as formatted text.
            context_before: Optional preceding context.
            context_after: Optional following context.

        Returns:
            Formatted prompt string.
        """
        context_parts = []
        if context_before:
            context_parts.append(f"Previous context: {context_before}")
        context_parts.append(f"Target sentence: {sentence}")
        if context_after:
            context_parts.append(f"Following context: {context_after}")

        context_text = "\n".join(context_parts)

        user_message = f"""{kb_text}

{context_text}

Analyze the target sentence for ableist language. Output your response as JSON."""

        return f"{PromptTemplate.SYSTEM_INSTRUCTION}\n\n{user_message}"


class LoRAAdapter(LLMTrainer):
    """Adapter for fine-tuning LLMs using LoRA (Low-Rank Adaptation).

    This adapter implements the LLMTrainer interface using HuggingFace
    Transformers and PEFT for efficient fine-tuning.

    Attributes:
        config: System configuration.
        model: The base language model.
        tokenizer: The model's tokenizer.
        peft_model: The PEFT model with LoRA adapters.
    """

    def __init__(self, config: SystemConfig) -> None:
        """Initialize the LoRA adapter.

        Args:
            config: System configuration.
        """
        self.config = config
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.peft_model: Optional[PeftModel] = None

        # Initialize model and tokenizer
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the base model with quantization."""
        print(f"Loading base model: {self.config.model_type.value}")

        # Configure quantization
        if self.config.quantization_config.quantization_type.value != "none":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=(
                    self.config.quantization_config.quantization_type.value == "4bit"
                ),
                load_in_8bit=(
                    self.config.quantization_config.quantization_type.value == "8bit"
                ),
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=self.config.quantization_config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.quantization_config.bnb_4bit_use_double_quant,
            )
        else:
            bnb_config = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_type.value,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=str(self.config.cache_dir),
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_type.value,
            trust_remote_code=True,
            cache_dir=str(self.config.cache_dir),
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        print("✓ Model and tokenizer loaded")

    def _prepare_peft_model(self) -> None:
        """Prepare the model for PEFT training with LoRA."""
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        peft_config = LoraConfig(
            r=self.config.lora_config.r,
            lora_alpha=self.config.lora_config.lora_alpha,
            target_modules=self.config.lora_config.target_modules,
            lora_dropout=self.config.lora_config.lora_dropout,
            bias=self.config.lora_config.bias,
            task_type=self.config.lora_config.task_type,
        )

        # Apply LoRA
        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()

    def _format_examples_for_training(
        self,
        examples: list[LabeledExample],
        kb_text: str,
    ) -> list[dict[str, str]]:
        """Format examples into training data format.

        Args:
            examples: Labeled examples.
            kb_text: Knowledge base text.

        Returns:
            List of formatted examples with 'text' key.
        """
        formatted = []
        for example in examples:
            prompt_data = PromptTemplate.build_training_prompt(example, kb_text)
            # Combine input and output for causal language modeling
            full_text = f"{prompt_data['input']}\n\n{prompt_data['output']}{self.tokenizer.eos_token}"
            formatted.append({"text": full_text})
        return formatted

    def train(
        self,
        training_examples: list[LabeledExample],
        validation_examples: Optional[list[LabeledExample]] = None,
    ) -> None:
        """Fine-tune the model using LoRA.

        Args:
            training_examples: Training examples.
            validation_examples: Optional validation examples.

        Raises:
            TrainingError: If training fails.
        """
        try:
            # Prepare PEFT model
            self._prepare_peft_model()

            # Get knowledge base text
            from config import KnowledgeBaseConfig

            kb_config = KnowledgeBaseConfig()
            kb_text = kb_config.get_all_principles_text()

            # Format training data
            train_dataset = self._format_examples_for_training(
                training_examples, kb_text
            )

            eval_dataset = None
            if validation_examples:
                eval_dataset = self._format_examples_for_training(
                    validation_examples, kb_text
                )

            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=str(self.config.output_dir / "checkpoints"),
                num_train_epochs=self.config.training_hyperparameters.num_epochs,
                per_device_train_batch_size=self.config.training_hyperparameters.batch_size,
                gradient_accumulation_steps=self.config.training_hyperparameters.gradient_accumulation_steps,
                learning_rate=self.config.training_hyperparameters.learning_rate,
                fp16=self.config.training_hyperparameters.fp16,
                logging_steps=self.config.training_hyperparameters.logging_steps,
                save_steps=self.config.training_hyperparameters.save_steps,
                eval_steps=self.config.training_hyperparameters.eval_steps
                if eval_dataset
                else None,
                evaluation_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                warmup_ratio=self.config.training_hyperparameters.warmup_ratio,
                optim=self.config.training_hyperparameters.optim,
                save_total_limit=3,
                load_best_model_at_end=True if eval_dataset else False,
                report_to=["none"],  # Disable wandb/tensorboard
            )

            # Create trainer
            trainer = SFTTrainer(
                model=self.peft_model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=training_args,
                max_seq_length=self.config.training_hyperparameters.max_seq_length,
                dataset_text_field="text",
            )

            # Start training
            trainer.train()

        except Exception as e:
            raise TrainingError(f"Training failed: {str(e)}") from e

    def save_model(self, output_path: str) -> None:
        """Save the fine-tuned LoRA adapter.

        Args:
            output_path: Path where the model should be saved.

        Raises:
            IOError: If saving fails.
        """
        if self.peft_model is None:
            raise IOError("No trained model to save")

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PEFT adapter
        self.peft_model.save_pretrained(str(output_dir))

        # Save tokenizer
        self.tokenizer.save_pretrained(str(output_dir))

        # Save config for reference
        config_path = output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "model_type": self.config.model_type.value,
                    "lora_r": self.config.lora_config.r,
                    "lora_alpha": self.config.lora_config.lora_alpha,
                },
                f,
                indent=2,
            )

    def load_model(self, model_path: str) -> None:
        """Load a previously fine-tuned LoRA adapter.

        Args:
            model_path: Path to the saved model.

        Raises:
            IOError: If loading fails.
        """
        try:
            # Load PEFT model
            self.peft_model = PeftModel.from_pretrained(
                self.model,
                model_path,
                is_trainable=False,
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        except Exception as e:
            raise IOError(f"Failed to load model: {str(e)}") from e


class HuggingFaceInferenceAdapter(InferenceEngine):
    """Adapter for running inference with a fine-tuned model.

    This adapter implements the InferenceEngine interface using
    HuggingFace Transformers for text generation.

    Attributes:
        config: System configuration.
        model: The fine-tuned PEFT model.
        tokenizer: The model's tokenizer.
    """

    def __init__(self, config: SystemConfig, model_path: str) -> None:
        """Initialize the inference adapter.

        Args:
            config: System configuration.
            model_path: Path to the fine-tuned model.
        """
        self.config = config
        self.model_path = model_path
        self.model: Optional[PeftModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the fine-tuned model for inference."""
        print(f"Loading fine-tuned model from: {self.model_path}")

        # Load base model with quantization
        if self.config.quantization_config.quantization_type.value != "none":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=(
                    self.config.quantization_config.quantization_type.value == "4bit"
                ),
                load_in_8bit=(
                    self.config.quantization_config.quantization_type.value == "8bit"
                ),
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=self.config.quantization_config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.quantization_config.bnb_4bit_use_double_quant,
            )
        else:
            bnb_config = None

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_type.value,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=str(self.config.cache_dir),
        )

        # Load PEFT adapter
        self.model = PeftModel.from_pretrained(
            base_model,
            self.model_path,
            is_trainable=False,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Set to evaluation mode
        self.model.eval()

        print("✓ Model loaded for inference")

    def generate_justification(
        self,
        sentence: str,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
        knowledge_base_text: Optional[str] = None,
    ) -> str:
        """Generate a justification for a single sentence.

        Args:
            sentence: Target sentence to analyze.
            context_before: Optional preceding context.
            context_after: Optional following context.
            knowledge_base_text: Knowledge base as text.

        Returns:
            Raw model output (should be JSON).

        Raises:
            InferenceError: If generation fails.
        """
        try:
            # Build prompt
            prompt = PromptTemplate.build_inference_prompt(
                sentence=sentence,
                kb_text=knowledge_base_text,
                context_before=context_before,
                context_after=context_after,
            )

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.training_hyperparameters.max_seq_length,
            ).to(self.model.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.inference_config.max_new_tokens,
                    temperature=self.config.inference_config.temperature,
                    top_p=self.config.inference_config.top_p,
                    top_k=self.config.inference_config.top_k,
                    do_sample=self.config.inference_config.do_sample,
                    repetition_penalty=self.config.inference_config.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode (skip the input prompt)
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )

            return generated_text.strip()

        except Exception as e:
            raise InferenceError(f"Generation failed: {str(e)}") from e

    def batch_generate(
        self,
        examples: list[LabeledExample],
        knowledge_base_text: str,
    ) -> list[str]:
        """Generate justifications for a batch of examples.

        Args:
            examples: List of labeled examples.
            knowledge_base_text: Knowledge base as text.

        Returns:
            List of raw outputs.

        Raises:
            InferenceError: If generation fails.
        """
        outputs = []
        for example in examples:
            output = self.generate_justification(
                sentence=example.sentence,
                context_before=example.context_before,
                context_after=example.context_after,
                knowledge_base_text=knowledge_base_text,
            )
            outputs.append(output)
        return outputs
