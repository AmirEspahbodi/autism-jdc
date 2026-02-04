"""
Infrastructure LLM module - Concrete implementations for LLM training and inference.
Refactored to support Pre-formatted SFT data without dynamic prompt assembly.
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
    """Handles prompt formatting.
    Refactored: Now acts as a pass-through for preformatted data.
    """

    @staticmethod
    def build_training_prompt(
        example: LabeledExample, kb_text: str = ""
    ) -> dict[str, str]:
        """
        Build a training prompt.

        Args:
            example: Labeled training example.
            kb_text: Knowledge base text (IGNORED for preformatted data).

        Returns:
            Dictionary with 'input' and 'output'.
        """
        # Case 1: Pre-formatted SFT Data
        if example.input_prompt is not None and example.model_output is not None:
            # Return raw strings. No modification, no system prompt injection.
            return {"input": example.input_prompt, "output": example.model_output}

        # Case 2: Legacy Structured Data (Fallback)
        # ... (Previous implementation for backward compatibility if needed) ...
        # Simplified for this refactor to focus on the request:
        raise NotImplementedError(
            "Legacy prompt construction is deprecated in this refactor."
        )

    @staticmethod
    def build_inference_prompt(
        sentence: str,
        kb_text: str,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """
        Build an inference prompt (input only).
        Maintained for Evaluation Use Case which might still use dynamic construction.
        """
        SYSTEM_INSTRUCTION = (
            """You are an expert in neurodiversity-aware language analysis..."""
        )

        context_parts = []
        if context_before:
            context_parts.append(f"Previous context: {context_before}")
        context_parts.append(f"Target sentence: {sentence}")
        if context_after:
            context_parts.append(f"Following context: {context_after}")

        context_text = "\n".join(context_parts)
        user_message = f"{kb_text}\n\n{context_text}\n\nAnalyze the target sentence..."

        return [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_message},
        ]


class LoRAAdapter(LLMTrainer):
    """Adapter for fine-tuning LLMs using LoRA."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.peft_model: Optional[PeftModel] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        print(f"Loading base model: {self.config.model_type.value}")

        # (Quantization logic remains unchanged)
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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_type.value,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=str(self.config.cache_dir),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_type.value,
            trust_remote_code=True,
            cache_dir=str(self.config.cache_dir),
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.padding_side = "right"

        # IMPORTANT: For pre-formatted data, we want full control over tokens
        if hasattr(self.tokenizer, "add_bos_token"):
            self.tokenizer.add_bos_token = False

    def _prepare_peft_model(self) -> None:
        # (PEFT logic remains unchanged)
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            r=self.config.lora_config.r,
            lora_alpha=self.config.lora_config.lora_alpha,
            target_modules=self.config.lora_config.target_modules,
            lora_dropout=self.config.lora_config.lora_dropout,
            bias=self.config.lora_config.bias,
            task_type=self.config.lora_config.task_type,
        )

        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()

    def _format_examples_for_training(
        self,
        examples: list[LabeledExample],
        kb_text: str,  # Ignored for pre-formatted
    ) -> list[dict[str, Any]]:
        """
        Format examples into tokenized training data.
        Refactored: Concatenates input+output and tokenizes directly.
        """
        formatted = []

        for example in examples:
            prompt_data = PromptTemplate.build_training_prompt(example)

            # SFT Logic: Concatenate Input + Output + EOS
            # We assume input_prompt already contains necessary system/role headers
            text = (
                prompt_data["input"] + prompt_data["output"] + self.tokenizer.eos_token
            )

            # Direct tokenization of the raw string
            # We do NOT use apply_chat_template here to avoid double-wrapping
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.training_hyperparameters.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )

            formatted.append(
                {
                    "input_ids": encoded["input_ids"][0].tolist(),
                    "attention_mask": encoded["attention_mask"][0].tolist(),
                }
            )

        return formatted

    def _prepare_data_collator(self):
        """
        Create data collator.
        Note: Depends on finding the separator to mask the input.
        """
        from src.config import ModelType

        response_template_ids = None

        # We assume the pre-formatted data uses standard templates corresponding
        # to the model type selected in config.
        if self.config.model_type == ModelType.LLAMA3_8B:
            # Llama 3: <|start_header_id|>assistant<|end_header_id|>\n\n
            template_str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            response_template_ids = self.tokenizer.encode(
                template_str, add_special_tokens=False
            )

        elif self.config.model_type == ModelType.MISTRAL_7B:
            # Mistral: [/INST]
            template_str = "[/INST]"
            response_template_ids = self.tokenizer.encode(
                template_str, add_special_tokens=False
            )

        if not response_template_ids:
            # Fallback
            template_str = "assistant"
            response_template_ids = self.tokenizer.encode(
                template_str, add_special_tokens=False
            )

        print(f"✓ Data collator configured. Masking inputs before: {template_str}")

        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=self.tokenizer,
            mlm=False,
        )

        return data_collator

    def train(
        self,
        training_examples: list[LabeledExample],
        validation_examples: Optional[list[LabeledExample]] = None,
    ) -> None:
        try:
            self._prepare_peft_model()

            # KB Text is technically not used by formatting anymore, but kept for signature compatibility
            kb_text = ""

            train_dataset = self._format_examples_for_training(
                training_examples, kb_text
            )

            eval_dataset = None
            if validation_examples:
                eval_dataset = self._format_examples_for_training(
                    validation_examples, kb_text
                )

            data_collator = self._prepare_data_collator()

            training_args = TrainingArguments(
                output_dir=str(self.config.output_dir / "checkpoints"),
                num_train_epochs=self.config.training_hyperparameters.num_epochs,
                per_device_train_batch_size=self.config.training_hyperparameters.batch_size,
                gradient_accumulation_steps=self.config.training_hyperparameters.gradient_accumulation_steps,
                learning_rate=self.config.training_hyperparameters.learning_rate,
                fp16=self.config.training_hyperparameters.fp16,
                logging_steps=self.config.training_hyperparameters.logging_steps,
                save_steps=self.config.training_hyperparameters.save_steps,
                evaluation_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                warmup_ratio=self.config.training_hyperparameters.warmup_ratio,
                optim=self.config.training_hyperparameters.optim,
                save_total_limit=3,
                report_to=["none"],
                remove_unused_columns=False,
            )

            trainer = SFTTrainer(
                model=self.peft_model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=training_args,
                data_collator=data_collator,
                max_seq_length=self.config.training_hyperparameters.max_seq_length,
                tokenizer=self.tokenizer,
            )

            trainer.train()

        except Exception as e:
            raise TrainingError(f"Training failed: {str(e)}") from e

    def save_model(self, output_path: str) -> None:
        if self.peft_model is None:
            raise IOError("No trained model to save")
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

    def load_model(self, model_path: str) -> None:
        pass  # Implementation omitted for brevity


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
