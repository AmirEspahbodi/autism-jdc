"""
Infrastructure LLM module - Concrete implementations for LLM training and inference.
Refactored to fix SFT masking alignment and EOS/BOS token handling.
"""

import json
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset
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
        Maintained for backwards compatibility if needed.
        """
        return [{"role": "user", "content": f"{kb_text}\n{sentence}"}]


class LoRAAdapter(LLMTrainer):
    """Adapter for fine-tuning LLMs using LoRA."""

    # CONSTANT SEPARATOR for Robust Masking
    # This ensures the DataCollator always finds the split point, regardless of model type.
    RESPONSE_SEPARATOR = "\n### Response:\n"

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.peft_model: Optional[PeftModel] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        print(f"Loading base model: {self.config.model_type.value}")

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

        # ---------------------------------------------------------------------
        # FIX 1: EOS Token Blindness & BOS Duplication
        # ---------------------------------------------------------------------
        # 1. Disable automatic BOS adding. We want control or to rely on the prompt text.
        #    Llama 3 often adds BOS automatically, leading to double BOS if not handled.
        self.tokenizer.add_bos_token = False

        # 2. Add PAD token if missing (Essential for SFTTrainer/Collator)
        if self.tokenizer.pad_token is None:
            print("Adding distinct [PAD] token to tokenizer...")
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

        # 3. Set padding side to right for SFTTrainer compatibility
        self.tokenizer.padding_side = "right"

    def _prepare_peft_model(self) -> None:
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            r=self.config.lora_config.r,
            lora_alpha=self.config.lora_config.lora_alpha,
            target_modules=self.config.lora_config.target_modules,
            lora_dropout=self.config.lora_config.lora_dropout,
            bias=self.config.lora_config.bias,
            task_type=self.config.lora_config.task_type,
            modules_to_save=self.config.lora_config.modules_to_save,
        )

        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()

    def _format_examples_for_training(
        self,
        examples: list[LabeledExample],
        kb_text: str,
    ) -> Dataset:
        """
        Refactored to inject the robust separator explicitly.
        Structure: [Input] + [Separator] + [Output] + [EOS]
        """
        data_list = []

        for example in examples:
            prompt_data = PromptTemplate.build_training_prompt(example)

            # -----------------------------------------------------------------
            # FIX 2: Explicit Template Injection
            # -----------------------------------------------------------------
            # We explicitly inject the separator so the Collator can find it.
            # We also append the EOS token string to ensure the model learns to stop.
            full_text = (
                f"{prompt_data['input']}"
                f"{self.RESPONSE_SEPARATOR}"
                f"{prompt_data['output']}"
                f"{self.tokenizer.eos_token}"
            )

            data_list.append({"text": full_text})

        return Dataset.from_list(data_list)

    def _prepare_data_collator(self):
        """
        Refactored to use the explicit separator for masking.
        Do NOT guess the template based on ModelType.
        """

        # -----------------------------------------------------------------
        # FIX 3: Robust Masking Configuration
        # -----------------------------------------------------------------
        # Use the exact same string we injected in `_format_examples_for_training`.
        response_template_str = self.RESPONSE_SEPARATOR

        # We generally do not need to encode it manually; the Collator accepts the string.
        # However, to be safe against tokenizer edge cases (like whitespace stripping),
        # passing the string is usually preferred for `response_template`.

        print(
            f"✓ Data collator configured. Masking inputs before: {repr(response_template_str)}"
        )

        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_str,
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
                dataset_text_field="text",
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
        pass


class HuggingFaceInferenceAdapter(InferenceEngine):
    """Concrete implementation of InferenceEngine using Hugging Face transformers.
    Supports loading fine-tuned LoRA models and batch generation.
    """

    def __init__(self, config: SystemConfig, model_path: str) -> None:
        """
        Args:
            config: System configuration.
            model_path: Path to the fine-tuned LoRA adapter.
        """
        self.config = config
        self.model_path = model_path
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._load_model()

    def _load_model(self) -> None:
        print(f"Loading base model for inference: {self.config.model_type.value}")

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

        try:
            # 1. Load Tokenizer FIRST
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_type.value,
                trust_remote_code=True,
                cache_dir=str(self.config.cache_dir),
            )

            # 2. Replicate Tokenizer Modifications from LoRAAdapter
            if self.tokenizer.pad_token is None:
                print("Adding distinct [PAD] token to tokenizer (matching training)...")
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            # Ensure BOS settings match training
            if hasattr(self.tokenizer, "add_bos_token"):
                self.tokenizer.add_bos_token = False

            # Configure for inference
            self.tokenizer.padding_side = "left"

            # 3. Load Base Model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_type.value,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=str(self.config.cache_dir),
            )

            print(f"Resizing model embeddings to {len(self.tokenizer)}...")
            base_model.resize_token_embeddings(len(self.tokenizer))

            # 5. Load LoRA Adapter
            print(f"Loading LoRA adapter from: {self.model_path}")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)

            if self.tokenizer.pad_token_id is not None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

        except Exception as e:
            raise InferenceError(f"Failed to load model: {str(e)}") from e

    def generate_justification(
        self,
        sentence: str,
        context_before: str | None = None,
        context_after: str | None = None,
        kb_text: str | None = None,
    ) -> str:
        """Single generation (Legacy/Wrapper)."""
        prompt = f"{kb_text or ''}\n{context_before or ''}\n{sentence}\n{context_after or ''}{LoRAAdapter.RESPONSE_SEPARATOR}"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.inference_config.max_new_tokens,
                temperature=self.config.inference_config.temperature,
                do_sample=self.config.inference_config.do_sample,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(
        self,
        examples: list[LabeledExample],
        kb_text: str,
    ) -> list[str]:
        """
        Generate completions for a batch of examples.
        Uses 'input_prompt' from the examples.
        """
        if not examples:
            return []

        # Update: We should likely append the separator here too if the input_prompt
        # doesn't contain it, but SFT input_prompts usually end where the model should start.
        # Assuming input_prompt is the question/instruction.
        prompts = []
        for ex in examples:
            if ex.input_prompt:
                # Append separator to prompt trigger generation
                p = f"{ex.input_prompt}{LoRAAdapter.RESPONSE_SEPARATOR}"
                prompts.append(p)

        if not prompts:
            raise InferenceError("No valid input_prompts found in examples.")

        results = []
        batch_size = self.config.training_hyperparameters.batch_size

        print(f"Starting inference on {len(prompts)} examples...")

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.training_hyperparameters.max_seq_length,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=self.config.inference_config.max_new_tokens,
                    temperature=self.config.inference_config.temperature,
                    top_p=self.config.inference_config.top_p,
                    top_k=self.config.inference_config.top_k,
                    do_sample=self.config.inference_config.do_sample,
                    repetition_penalty=self.config.inference_config.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            decoded_batch = []
            for j, output in enumerate(outputs):
                # Slice off the prompt length to return ONLY new tokens
                input_len = inputs.input_ids[j].shape[0]
                generated_tokens = output[input_len:]
                text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                decoded_batch.append(text)

            results.extend(decoded_batch)

        return results
