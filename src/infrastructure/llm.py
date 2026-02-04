"""
Infrastructure LLM module - Concrete implementations for LLM training and inference.
Refactored to support Pre-formatted SFT data without dynamic prompt assembly.
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
        Maintained for backwards compatibility if needed.
        """
        return [{"role": "user", "content": f"{kb_text}\n{sentence}"}]


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
        # FIX FOR BUG 1: Stop Token Blindness (EOS == PAD)
        # ---------------------------------------------------------------------
        # Instead of setting pad_token = eos_token (which masks EOS in loss),
        # we ensure a distinct padding token exists.
        if self.tokenizer.pad_token is None:
            print("Adding distinct [PAD] token to tokenizer...")
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            # CRITICAL: Resize embeddings so the model knows about the new token
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Ensure padding side is right for Training (SFTTrainer requirement usually)
        # Note: Inference usually prefers left-padding, but SFTTrainer handles right-padding best.
        self.tokenizer.padding_side = "right"

        if hasattr(self.tokenizer, "add_bos_token"):
            self.tokenizer.add_bos_token = False

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
        )

        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()

    def _format_examples_for_training(
        self,
        examples: list[LabeledExample],
        kb_text: str,  # Ignored for pre-formatted
    ) -> Dataset:
        """
        Refactored to return a HuggingFace Dataset containing raw text.
        NO manual tokenization or padding is performed here.
        """
        data_list = []

        for example in examples:
            prompt_data = PromptTemplate.build_training_prompt(example)

            # Construct the full text string.
            # We explicitly append EOS to ensure the model learns to stop.
            full_text = (
                prompt_data["input"] + prompt_data["output"] + self.tokenizer.eos_token
            )

            data_list.append({"text": full_text})

        # Return a standard HF Dataset which SFTTrainer expects
        return Dataset.from_list(data_list)

    def _prepare_data_collator(self):
        from src.config import ModelType

        response_template_ids = None

        if self.config.model_type == ModelType.LLAMA3_8B:
            template_str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            response_template_ids = self.tokenizer.encode(
                template_str, add_special_tokens=False
            )
        elif self.config.model_type == ModelType.MISTRAL_7B:
            template_str = "[/INST]"
            response_template_ids = self.tokenizer.encode(
                template_str, add_special_tokens=False
            )

        if not response_template_ids:
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
            kb_text = ""

            # -----------------------------------------------------------------
            # FIX FOR BUG 2: Manual Tokenization Anti-Pattern
            # -----------------------------------------------------------------
            # We now get datasets of raw strings, not pre-tokenized tensors.
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
                remove_unused_columns=False,  # Essential when using dataset_text_field
            )

            # Initialize SFTTrainer with the raw text dataset.
            # SFTTrainer will handle tokenization and packing/padding internally.
            trainer = SFTTrainer(
                model=self.peft_model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field="text",  # Tells SFTTrainer which column to tokenize
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

        # Load Base Model
        # Note: We usually use 4-bit/8-bit for inference if configured, similar to training
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
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_type.value,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=str(self.config.cache_dir),
            )

            # Load LoRA Adapter
            print(f"Loading LoRA adapter from: {self.model_path}")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)

            # Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_type.value,
                trust_remote_code=True,
                cache_dir=str(self.config.cache_dir),
            )

            # Configure tokenizer for generation (left padding is crucial for batch generation)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

            self.tokenizer.padding_side = "left"

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
        # Create a temporary example to leverage common logic if needed,
        # but for SFT we normally prefer batch_generate with full prompts.
        # This is a fallback implementation.
        prompt = f"{kb_text or ''}\n{context_before or ''}\n{sentence}\n{context_after or ''}"

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
        kb_text: str,  # Kept for interface consistency, might be unused in SFT
    ) -> list[str]:
        """
        Generate completions for a batch of examples.
        Uses 'input_prompt' from the examples.
        """
        if not examples:
            return []

        # Extract prompts. Ensure we rely on the pre-formatted input_prompt
        prompts = [ex.input_prompt for ex in examples if ex.input_prompt]

        if not prompts:
            raise InferenceError("No valid input_prompts found in examples.")

        results = []
        batch_size = (
            self.config.training_hyperparameters.batch_size
        )  # Reuse batch size or define new

        print(f"Starting inference on {len(prompts)} examples...")

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            # Tokenize
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

            # Decode
            # We only want the *new* tokens, not the input prompt
            decoded_batch = []
            for j, output in enumerate(outputs):
                input_len = inputs.input_ids[j].shape[0]
                # In some cases generate returns input + output.
                # We slice off the input length if it matches.
                # However, with padding, calculating exact input length per row is tricky
                # if not careful.
                # Safer approach: decode full and strip prompt, or slice strictly by generated tokens.

                # Standard approach for causal LM generation in HF often returns full sequence.
                # Let's decode only the new tokens.
                generated_tokens = output[inputs.input_ids.shape[1] :]
                text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                decoded_batch.append(text)

            results.extend(decoded_batch)

        return results
