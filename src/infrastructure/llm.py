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
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

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
        """
        Build a training prompt with input and expected output.

        Args:
            example: Labeled training example.
            kb_text: Knowledge base as formatted text.

        Returns:
            Dictionary with 'input' and 'output' keys (for backward compatibility),
            but also includes 'messages' for chat template usage.
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
            "input": user_message,
            "output": json.dumps(justification_json, indent=2),
        }

    @staticmethod
    def build_inference_prompt(
        sentence: str,
        kb_text: str,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Build an inference prompt (input only).

        Args:
            sentence: Target sentence to analyze.
            kb_text: Knowledge base as formatted text.
            context_before: Optional preceding context.
            context_after: Optional following context.

        Returns:
            List of message dicts with 'role' and 'content' keys for chat template.
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

        return [
            {"role": "system", "content": PromptTemplate.SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_message},
        ]


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
        """
        Initialize the base model with quantization.

        FIXED: ERROR #3 - Prevents double BOS token injection
        """
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

        # Configure padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Right padding ensures causal attention masks align correctly during training
        self.tokenizer.padding_side = "right"

        # FIX ERROR #3: Disable automatic BOS token addition
        # The chat template handles BOS tokens, so tokenizer shouldn't add them
        if hasattr(self.tokenizer, "add_bos_token"):
            self.tokenizer.add_bos_token = False
            print("✓ Disabled tokenizer.add_bos_token (chat template handles BOS)")

        # Validate BOS token count
        self._validate_bos_tokens()

        print(
            f"✓ Model and tokenizer loaded (padding_side={self.tokenizer.padding_side})"
        )

    def _validate_bos_tokens(self) -> None:
        """
        Validate that chat template produces exactly one BOS token.

        FIX ERROR #3: Ensures training-inference distribution match

        Raises:
            ValueError: If multiple BOS tokens are detected
        """
        # Create test messages
        test_messages = [
            {"role": "system", "content": "Test system message"},
            {"role": "user", "content": "Test user message"},
            {"role": "assistant", "content": "Test assistant response"},
        ]

        # Apply chat template with tokenization
        token_ids = self.tokenizer.apply_chat_template(
            test_messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        # Convert to tensor if needed
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids)

        # Count BOS tokens
        bos_count = (token_ids == self.tokenizer.bos_token_id).sum().item()

        if bos_count > 1:
            raise ValueError(
                f"Double BOS token detected! Found {bos_count} BOS tokens in chat template output. "
                f"This causes training-inference distribution mismatch. "
                f"Expected: 1 BOS token (from chat template). "
                f"Fix: Ensure tokenizer.add_bos_token=False"
            )

        print(f"✓ BOS token validation passed ({bos_count} BOS token)")

    def _prepare_peft_model(self) -> None:
        """
        Prepare the model for PEFT training with LoRA.
        """
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # This trades computation for memory by not storing all activations
        # Critical for training with consumer GPUs (RTX 4090, etc.)
        self.model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled (saves ~30-40% GPU memory)")

        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
            print("✓ Input gradients enabled for quantized model")
        else:
            # Fallback for older transformers versions (< 4.35.0)
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self.model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )
            print("✓ Input gradients enabled via forward hook (legacy method)")

        # Configure LoRA
        peft_config = LoraConfig(
            r=self.config.lora_config.r,
            lora_alpha=self.config.lora_config.lora_alpha,
            target_modules=self.config.lora_config.target_modules,  # Includes MLP layers
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
        """
        Format examples into training data using chat template.

        DataCollatorForCompletionOnlyLM will mask the input tokens (labels = -100).

        Args:
            examples: Labeled examples.
            kb_text: Knowledge base text.

        Returns:
            List of formatted examples with 'text' key.
        """
        formatted = []

        for example in examples:
            prompt_data = PromptTemplate.build_training_prompt(example, kb_text)

            # This enables DataCollatorForCompletionOnlyLM to identify and mask input tokens
            messages = [
                {"role": "system", "content": PromptTemplate.SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt_data["input"]},
                {"role": "assistant", "content": prompt_data["output"]},
            ]

            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # Assistant response already present
            )

            formatted.append({"text": formatted_text})

        return formatted

    def _prepare_data_collator(self):
        """
        Create data collator that masks instruction tokens.

        FIXED: ERROR #1 - Extracts exact response template with whitespace validation

        Returns:
            DataCollatorForCompletionOnlyLM configured for the model.
        """
        from src.config import ModelType

        # FIX ERROR #1: Extract the EXACT response template from chat template
        # We need to include all whitespace/newlines that appear after the header

        # Create test messages to extract the exact template
        test_messages = [
            {"role": "system", "content": "Test"},
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "ASSISTANT_RESPONSE_MARKER"},
        ]

        test_formatted = self.tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Extract everything from assistant header to the marker
        # This captures the exact whitespace/newlines
        if self.config.model_type == ModelType.LLAMA3_8B:
            # Find the assistant header
            header_start = test_formatted.find(
                "<|start_header_id|>assistant<|end_header_id|>"
            )
            if header_start == -1:
                raise ValueError(
                    "Could not find assistant header in chat template output. "
                    "This will cause instruction masking to fail."
                )

            # Find the marker (this tells us where actual content starts)
            marker_start = test_formatted.find(
                "ASSISTANT_RESPONSE_MARKER", header_start
            )
            if marker_start == -1:
                raise ValueError("Could not find response marker in template")

            # Extract the template including whitespace
            response_template = test_formatted[header_start:marker_start]

        elif self.config.model_type == ModelType.MISTRAL_7B:
            # Mistral specific response pattern
            header_start = test_formatted.find("[/INST]")
            if header_start == -1:
                raise ValueError(
                    "Could not find Mistral response marker in chat template output. "
                    "This will cause instruction masking to fail."
                )

            marker_start = test_formatted.find(
                "ASSISTANT_RESPONSE_MARKER", header_start
            )
            if marker_start == -1:
                raise ValueError("Could not find response marker in template")

            response_template = test_formatted[header_start:marker_start]
        else:
            # Fallback - attempt to detect
            # You can inspect: self.tokenizer.apply_chat_template(test_messages)
            response_template = "assistant"
            print(f"⚠ Warning: Using generic response template. May need adjustment.")

        # Validate that the template exists in actual formatted data
        # This is critical - if template doesn't match, masking silently fails
        print(f"✓ Response template extracted: {repr(response_template)}")

        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
            mlm=False,  # Not masked language modeling
        )

        print(
            f"✓ Data collator configured with response_template: {repr(response_template)}"
        )

        return data_collator

    def _validate_label_masking(
        self,
        data_collator: DataCollatorForCompletionOnlyLM,
        formatted_data: list[dict[str, str]],
    ) -> None:
        """
        Validate that label masking is working correctly.

        FIX ERROR #2: Ensures padding tokens and instruction tokens are masked

        This prevents the silent failure mode where:
        - Padding tokens aren't masked → model learns to predict EOS
        - Instruction tokens aren't masked → model learns to repeat prompts

        Args:
            data_collator: The configured data collator
            formatted_data: List of formatted training examples

        Raises:
            ValueError: If masking validation fails
        """
        if not formatted_data:
            raise ValueError("No formatted data provided for validation")

        # Take a sample batch (first 4 examples or all if fewer)
        sample_size = min(4, len(formatted_data))
        sample_data = formatted_data[:sample_size]

        # Tokenize the samples
        sample_texts = [item["text"] for item in sample_data]
        tokenized = self.tokenizer(
            sample_texts,
            padding=True,
            truncation=True,
            max_length=self.config.training_hyperparameters.max_seq_length,
            return_tensors="pt",
        )

        # Process through data collator to get labels
        batch = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

        # The collator adds the 'labels' field with masking
        processed_batch = data_collator.torch_call([batch])

        if "labels" not in processed_batch:
            raise ValueError(
                "Data collator did not produce 'labels' field. "
                "Label masking cannot be validated."
            )

        labels = processed_batch["labels"]
        input_ids = processed_batch["input_ids"]

        # Validation 1: Check padding token masking
        # All positions where input_ids == pad_token_id should have labels == -100
        pad_positions = input_ids == self.tokenizer.pad_token_id
        pad_labels = labels[pad_positions]

        if pad_positions.sum() > 0:
            incorrectly_unmasked_padding = (pad_labels != -100).sum().item()
            total_padding = pad_positions.sum().item()

            if incorrectly_unmasked_padding > 0:
                raise ValueError(
                    f"CRITICAL: Padding tokens are not masked! "
                    f"Found {incorrectly_unmasked_padding}/{total_padding} padding positions with labels != -100. "
                    f"This will cause the model to learn to predict EOS tokens, "
                    f"resulting in empty outputs during inference. "
                    f"Check that pad_token_id is correctly set."
                )

            print(f"✓ Padding tokens correctly masked: {total_padding} tokens")

        # Validation 2: Check instruction masking ratio
        # Most tokens should be masked (instructions), few should be unmasked (responses)
        masked_count = (labels == -100).sum().item()
        unmasked_count = (labels != -100).sum().item()
        total_tokens = labels.numel()

        masking_ratio = masked_count / total_tokens if total_tokens > 0 else 0

        # Sanity check: For instruction masking, we expect most tokens to be masked
        # Typical ratio is 70-90% masked (instructions + padding)
        if masking_ratio < 0.5:
            raise ValueError(
                f"CRITICAL: Instruction masking appears to have failed! "
                f"Only {masked_count}/{total_tokens} ({masking_ratio:.1%}) tokens are masked. "
                f"Expected >50% masking ratio. "
                f"This indicates the response_template doesn't match the chat template output. "
                f"Model will learn to repeat instructions instead of generating responses."
            )

        print(
            f"✓ Label masking validated: {masked_count} masked ({masking_ratio:.1%}), "
            f"{unmasked_count} unmasked"
        )

    def _validate_sequence_lengths(
        self,
        formatted_data: list[dict[str, str]],
        max_length: int,
    ) -> None:
        """
        Validate that training examples don't exceed max sequence length.

        FIX ERROR #4: Prevents silent truncation of JSON outputs

        Args:
            formatted_data: List of formatted training examples
            max_length: Maximum sequence length

        Raises:
            ValueError: If >10% of examples exceed max_length
        """
        if not formatted_data:
            return

        texts = [item["text"] for item in formatted_data]

        # Tokenize WITHOUT truncation to get true lengths
        tokenized = self.tokenizer(
            texts,
            truncation=False,  # CRITICAL: Don't truncate during validation
            return_attention_mask=False,
        )

        # Count examples that exceed max_length
        long_examples = []
        for i, input_ids in enumerate(tokenized["input_ids"]):
            length = len(input_ids)
            if length > max_length:
                long_examples.append((i, length))

        num_long = len(long_examples)
        total = len(formatted_data)
        percentage = (num_long / total * 100) if total > 0 else 0

        if num_long == 0:
            print(f"✓ All {total} examples within max_seq_length ({max_length} tokens)")
            return

        # If >10% are truncated, this is a critical error
        if percentage > 10:
            # Show some examples
            examples_str = "\n".join(
                f"  Example {i}: {length} tokens (exceeds by {length - max_length})"
                for i, length in long_examples[:5]
            )

            raise ValueError(
                f"CRITICAL: {num_long}/{total} examples ({percentage:.1f}%) exceed max_seq_length! "
                f"\n{examples_str}"
                f"\n\nTruncated examples will have invalid JSON (cut mid-generation), "
                f"causing the model to learn malformed outputs. "
                f"\nSolutions:\n"
                f"  1. Increase max_seq_length in config (current: {max_length})\n"
                f"  2. Reduce knowledge base text length\n"
                f"  3. Filter out long examples from training data"
            )

        # If 1-10% are truncated, warn but continue
        print(
            f"⚠ WARNING: {num_long}/{total} examples ({percentage:.1f}%) exceed max_seq_length ({max_length})"
        )
        print(f"  These examples will be truncated during training.")
        print(
            f"  Consider increasing max_seq_length if truncation affects performance."
        )

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
            from src.config import KnowledgeBaseConfig

            kb_config = KnowledgeBaseConfig()
            kb_text = kb_config.get_all_principles_text()

            train_dataset = self._format_examples_for_training(
                training_examples, kb_text
            )

            eval_dataset = None
            if validation_examples:
                eval_dataset = self._format_examples_for_training(
                    validation_examples, kb_text
                )

            # FIX ERROR #4: Validate sequence lengths BEFORE training
            print("\n[Validation] Checking sequence lengths...")
            self._validate_sequence_lengths(
                train_dataset,
                self.config.training_hyperparameters.max_seq_length,
            )

            data_collator = self._prepare_data_collator()

            # FIX ERROR #2: Validate label masking BEFORE training
            print("\n[Validation] Checking label masking...")
            self._validate_label_masking(data_collator, train_dataset)

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

            trainer = SFTTrainer(
                model=self.peft_model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=training_args,
                data_collator=data_collator,
                max_seq_length=self.config.training_hyperparameters.max_seq_length,
                dataset_text_field="text",
            )

            # Start training
            print(
                "\n✓ All validations passed - Starting training with instruction masking enabled..."
            )
            print("   (Loss will only be computed on assistant responses)")
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
                    "target_modules": self.config.lora_config.target_modules,
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
        """
        Load the fine-tuned model for inference.

        FIXED: ERROR #3 - Prevents double BOS token during inference
        """
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

        # Left padding ensures sequences are end-aligned, so generation starts
        # from the last real token (not padding tokens)
        self.tokenizer.padding_side = "left"

        # Configure padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # FIX ERROR #3: Disable automatic BOS token addition (same as training)
        if hasattr(self.tokenizer, "add_bos_token"):
            self.tokenizer.add_bos_token = False
            print("✓ Disabled tokenizer.add_bos_token for inference (matches training)")

        # Validate BOS token count (same validation as training)
        self._validate_bos_tokens()

        # Set to evaluation mode
        self.model.eval()

        print(
            f"✓ Model loaded for inference (padding_side={self.tokenizer.padding_side})"
        )

    def _validate_bos_tokens(self) -> None:
        """
        Validate that chat template produces exactly one BOS token.

        FIX ERROR #3: Ensures training-inference distribution match

        Raises:
            ValueError: If multiple BOS tokens are detected
        """
        # Create test messages
        test_messages = [
            {"role": "system", "content": "Test system message"},
            {"role": "user", "content": "Test user message"},
        ]

        # Apply chat template with tokenization and generation prompt
        token_ids = self.tokenizer.apply_chat_template(
            test_messages,
            tokenize=True,
            add_generation_prompt=True,  # This is what we use during inference
        )

        # Convert to tensor if needed
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids)

        # Count BOS tokens
        bos_count = (token_ids == self.tokenizer.bos_token_id).sum().item()

        if bos_count > 1:
            raise ValueError(
                f"Double BOS token detected in inference! Found {bos_count} BOS tokens. "
                f"This creates training-inference distribution mismatch. "
                f"Expected: 1 BOS token (from chat template). "
                f"Fix: Ensure tokenizer.add_bos_token=False"
            )

        print(f"✓ BOS token validation passed for inference ({bos_count} BOS token)")

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
            messages = PromptTemplate.build_inference_prompt(
                sentence=sentence,
                kb_text=knowledge_base_text,
                context_before=context_before,
                context_after=context_after,
            )

            # This adds the <|start_header_id|>assistant<|end_header_id|> cue
            # that the model was trained to recognize
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,  # Get string first
                add_generation_prompt=True,  # Add assistant prompt - CRITICAL!
            )

            # Tokenize the chat-formatted prompt
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

    def batch_generate(self, examples: list[LabeledExample], kb_text: str) -> list[str]:
        """
        Generate justifications for a batch of examples.

        Args:
            examples: List of labeled examples.
            kb_text: Knowledge base as text.

        Returns:
            List of raw outputs.

        Raises:
            InferenceError: If generation fails.
        """
        try:
            prompts = []
            for ex in examples:
                messages = PromptTemplate.build_inference_prompt(
                    sentence=ex.sentence,
                    kb_text=kb_text,
                    context_before=ex.context_before,
                    context_after=ex.context_after,
                )

                # Apply chat template with generation prompt
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)

            # Tokenize batch with left padding (already set in _load_model)
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,  # Pad to longest in batch
                truncation=True,
                max_length=self.config.training_hyperparameters.max_seq_length,
            ).to(self.model.device)

            # Batch generation
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

            # Decode outputs (skip input tokens for each example)
            results = []
            for i, output in enumerate(outputs):
                input_length = inputs.input_ids[i].shape[0]
                generated = self.tokenizer.decode(
                    output[input_length:],
                    skip_special_tokens=True,
                )
                results.append(generated.strip())

            return results

        except Exception as e:
            raise InferenceError(f"Batch generation failed: {str(e)}") from e
