"""
Infrastructure LLM module - Concrete implementations for LLM training and inference.

This module contains adapters that implement the domain interfaces using
HuggingFace Transformers, PEFT (LoRA), and bitsandbytes for quantization.
Refactored to enforce ID-based tokenization for Special Token safety.
"""

import json
from pathlib import Path
from typing import Any, Optional

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

        # Disable automatic BOS token addition
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
        self.model = prepare_model_for_kbit_training(self.model)

        self.model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled (saves ~30-40% GPU memory)")

        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self.model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

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
    ) -> list[dict[str, Any]]:
        """
        Format examples into tokenized training data using chat template.

        CRITICAL FIX: Now returns 'input_ids' and 'attention_mask' directly.
        This prevents the 'text' string from being re-tokenized by the Trainer,
        which would split special tokens (e.g. <|start_header_id|>) into plain text.

        Args:
            examples: Labeled examples.
            kb_text: Knowledge base text.

        Returns:
            List of dicts with 'input_ids' and 'attention_mask'.
        """
        formatted = []

        for example in examples:
            prompt_data = PromptTemplate.build_training_prompt(example, kb_text)

            messages = [
                {"role": "system", "content": PromptTemplate.SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt_data["input"]},
                {"role": "assistant", "content": prompt_data["output"]},
            ]

            # Direct tokenization to preserve Special Tokens
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )

            # SFTTrainer expects list of dicts (if not using dataset object),
            # and values should be lists, not tensors, for compatibility
            formatted.append(
                {
                    "input_ids": encoded["input_ids"][0].tolist(),
                    "attention_mask": encoded["attention_mask"][0].tolist(),
                }
            )

        return formatted

    def _prepare_data_collator(self):
        """
        Create data collator that masks instruction tokens using explicit ID-based matching.
        """
        from src.config import ModelType

        response_template_ids = None

        if self.config.model_type == ModelType.LLAMA3_8B:
            # Llama 3 Template: <|start_header_id|>assistant<|end_header_id|>\n\n
            template_str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            response_template_ids = self.tokenizer.encode(
                template_str, add_special_tokens=False
            )

        elif self.config.model_type == ModelType.MISTRAL_7B:
            # Mistral Template: [/INST]
            template_str = "[/INST]"
            response_template_ids = self.tokenizer.encode(
                template_str, add_special_tokens=False
            )

        else:
            print(
                f"⚠ Warning: Unknown model type {self.config.model_type}. Using generic 'assistant' template."
            )
            template_str = "assistant"
            response_template_ids = self.tokenizer.encode(
                template_str, add_special_tokens=False
            )

        if not response_template_ids:
            raise ValueError(
                f"Critical Error: Failed to encode response template IDs for {self.config.model_type}."
            )

        print(f"✓ Response template encoded to IDs: {response_template_ids}")

        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=self.tokenizer,
            mlm=False,
        )

        print(f"✓ Data collator configured with ID-based masking")
        return data_collator

    def _validate_label_masking(
        self,
        data_collator: DataCollatorForCompletionOnlyLM,
        formatted_data: list[dict[str, Any]],
    ) -> None:
        """
        Validate that label masking is working correctly on pre-tokenized data.

        Args:
            data_collator: The configured data collator
            formatted_data: List of pre-tokenized training examples

        Raises:
            ValueError: If masking validation fails
        """
        if not formatted_data:
            raise ValueError("No formatted data provided for validation")

        # Take a sample batch
        sample_size = min(4, len(formatted_data))
        sample_data = formatted_data[:sample_size]

        # Since data is already tokenized, we just pad it
        # Note: DataCollatorForCompletionOnlyLM handles padding if input_ids are passed

        # We need to manually construct the batch for validation
        input_ids = [torch.tensor(item["input_ids"]) for item in sample_data]
        attention_mask = [torch.tensor(item["attention_mask"]) for item in sample_data]

        # Use tokenizer to pad for batching validation
        padded = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        batch = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
        }

        # Process through data collator to get labels
        # Note: The collator expects a list of dicts or a BatchEncoding
        processed_batch = data_collator.torch_call(
            [
                {"input_ids": ids, "attention_mask": mask}
                for ids, mask in zip(input_ids, attention_mask)
            ]
        )

        if "labels" not in processed_batch:
            raise ValueError("Data collator did not produce 'labels' field.")

        labels = processed_batch["labels"]

        # Validation stats
        masked_count = (labels == -100).sum().item()
        unmasked_count = (labels != -100).sum().item()
        total_tokens = labels.numel()
        masking_ratio = masked_count / total_tokens if total_tokens > 0 else 0

        if masking_ratio < 0.5:
            raise ValueError(
                f"CRITICAL: Instruction masking appears to have failed! "
                f"Only {masked_count}/{total_tokens} ({masking_ratio:.1%}) tokens are masked. "
                f"Expected >50% masking ratio."
            )

        print(f"✓ Label masking validated: {masked_count} masked ({masking_ratio:.1%})")

    def _validate_sequence_lengths(
        self,
        formatted_data: list[dict[str, Any]],
        max_length: int,
    ) -> None:
        """
        Validate that training examples don't exceed max sequence length.

        Args:
            formatted_data: List of pre-tokenized training examples
            max_length: Maximum sequence length
        """
        if not formatted_data:
            return

        long_examples = []
        for i, item in enumerate(formatted_data):
            length = len(item["input_ids"])
            if length > max_length:
                long_examples.append((i, length))

        num_long = len(long_examples)
        total = len(formatted_data)
        percentage = (num_long / total * 100) if total > 0 else 0

        if num_long == 0:
            print(f"✓ All {total} examples within max_seq_length ({max_length} tokens)")
            return

        if percentage > 10:
            examples_str = "\n".join(
                f"  Example {i}: {length} tokens (exceeds by {length - max_length})"
                for i, length in long_examples[:5]
            )
            raise ValueError(
                f"CRITICAL: {num_long}/{total} examples ({percentage:.1f}%) exceed max_seq_length! "
                f"\n{examples_str}"
            )

        print(
            f"⚠ WARNING: {num_long}/{total} examples ({percentage:.1f}%) exceed max_seq_length ({max_length})"
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
            self._prepare_peft_model()
            from src.config import KnowledgeBaseConfig

            kb_config = KnowledgeBaseConfig()
            kb_text = kb_config.get_all_principles_text()

            # Format data (now returns input_ids/attention_mask directly)
            train_dataset = self._format_examples_for_training(
                training_examples, kb_text
            )

            eval_dataset = None
            if validation_examples:
                eval_dataset = self._format_examples_for_training(
                    validation_examples, kb_text
                )

            print("\n[Validation] Checking sequence lengths...")
            self._validate_sequence_lengths(
                train_dataset,
                self.config.training_hyperparameters.max_seq_length,
            )

            data_collator = self._prepare_data_collator()

            print("\n[Validation] Checking label masking...")
            self._validate_label_masking(data_collator, train_dataset)

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
                report_to=["none"],
                # IMPORTANT: Remove unused columns since we provide input_ids
                remove_unused_columns=False,
            )

            # We pass the pre-tokenized dataset directly
            # dataset_text_field is REMOVED because we are not passing text anymore
            trainer = SFTTrainer(
                model=self.peft_model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=training_args,
                data_collator=data_collator,
                max_seq_length=self.config.training_hyperparameters.max_seq_length,
                # tokenizer needed here for padding in collator
                tokenizer=self.tokenizer,
            )

            print(
                "\n✓ All validations passed - Starting training with ID-based tokenization..."
            )
            trainer.train()

        except Exception as e:
            raise TrainingError(f"Training failed: {str(e)}") from e

    def save_model(self, output_path: str) -> None:
        """Save the fine-tuned LoRA adapter."""
        if self.peft_model is None:
            raise IOError("No trained model to save")

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

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
        """Load a previously fine-tuned LoRA adapter."""
        try:
            self.peft_model = PeftModel.from_pretrained(
                self.model,
                model_path,
                is_trainable=False,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            raise IOError(f"Failed to load model: {str(e)}") from e


class HuggingFaceInferenceAdapter(InferenceEngine):
    """Adapter for running inference with a fine-tuned model."""

    def __init__(self, config: SystemConfig, model_path: str) -> None:
        self.config = config
        self.model_path = model_path
        self.model: Optional[PeftModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the fine-tuned model for inference."""
        print(f"Loading fine-tuned model from: {self.model_path}")

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

        self.model = PeftModel.from_pretrained(
            base_model,
            self.model_path,
            is_trainable=False,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        if hasattr(self.tokenizer, "add_bos_token"):
            self.tokenizer.add_bos_token = False
            print("✓ Disabled tokenizer.add_bos_token for inference (matches training)")

        self._validate_bos_tokens()
        self.model.eval()
        print(
            f"✓ Model loaded for inference (padding_side={self.tokenizer.padding_side})"
        )

    def _validate_bos_tokens(self) -> None:
        """Validate that chat template produces exactly one BOS token."""
        test_messages = [
            {"role": "system", "content": "Test system message"},
            {"role": "user", "content": "Test user message"},
        ]

        token_ids = self.tokenizer.apply_chat_template(
            test_messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids)

        bos_count = (token_ids == self.tokenizer.bos_token_id).sum().item()

        if bos_count > 1:
            raise ValueError(
                f"Double BOS token detected in inference! Found {bos_count} BOS tokens."
            )

        print(f"✓ BOS token validation passed for inference ({bos_count} BOS token)")

    def generate_justification(
        self,
        sentence: str,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
        knowledge_base_text: Optional[str] = None,
    ) -> str:
        """Generate a justification for a single sentence."""
        try:
            messages = PromptTemplate.build_inference_prompt(
                sentence=sentence,
                kb_text=knowledge_base_text,
                context_before=context_before,
                context_after=context_after,
            )

            # CRITICAL FIX: Use tokenize=True directly.
            # Do NOT generate string and then re-tokenize.
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.model.device)

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
        """Generate justifications for a batch of examples."""
        try:
            # We need to tokenize individually first to handle chat template logic
            # for each example, then pad them together.
            batch_input_ids = []

            for ex in examples:
                messages = PromptTemplate.build_inference_prompt(
                    sentence=ex.sentence,
                    kb_text=kb_text,
                    context_before=ex.context_before,
                    context_after=ex.context_after,
                )

                # Get IDs directly (1D tensor)
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                batch_input_ids.append(encoded[0])  # Append the 1D tensor

            # Use tokenizer to pad the list of tensors
            # This handles attention_mask creation automatically
            inputs = self.tokenizer.pad(
                {"input_ids": batch_input_ids}, padding=True, return_tensors="pt"
            ).to(self.model.device)

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
