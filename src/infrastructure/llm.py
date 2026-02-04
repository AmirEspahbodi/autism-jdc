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

from src.config import ModelType, SystemConfig
from src.domain import (
    InferenceEngine,
    InferenceError,
    LabeledExample,
    LLMTrainer,
    TrainingError,
)


class PromptTemplate:
    """Handles prompt formatting.
    Refactored: Now acts as a pass-through for preformatted data, relying on
    tokenizer.apply_chat_template for structure.
    """

    @staticmethod
    def build_training_messages(
        example: LabeledExample,
    ) -> list[dict[str, str]]:
        """
        Build a list of messages for chat templating.
        """
        if example.input_prompt is not None and example.model_output is not None:
            return [
                {"role": "user", "content": example.input_prompt},
                {"role": "assistant", "content": example.model_output},
            ]

        raise NotImplementedError("Legacy prompt construction is deprecated.")

    @staticmethod
    def build_inference_messages(
        sentence: str,
        kb_text: str = "",
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """
        Build inference message list.
        """
        # Construct the full user content
        parts = []
        if kb_text:
            parts.append(kb_text)
        if context_before:
            parts.append(context_before)
        parts.append(sentence)
        if context_after:
            parts.append(context_after)

        full_content = "\n\n".join(parts)
        return [{"role": "user", "content": full_content}]


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

        # 1. We DO NOT resize embeddings on quantized models. It corrupts weights.
        # 2. We reuse the EOS token as PAD. This is standard for Llama/Mistral.
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 3. Set padding side to right for SFTTrainer
        self.tokenizer.padding_side = "right"

        # 4. Disable manual BOS addition; let apply_chat_template handle it
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
            modules_to_save=self.config.lora_config.modules_to_save,
        )

        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()

    def _format_examples_for_training(self, examples: list[LabeledExample]) -> Dataset:
        # (Same logic as before, ensuring text field is created)
        data_list = []
        for example in examples:
            if example.input_prompt is not None and example.model_output is not None:
                messages = [
                    {"role": "user", "content": example.input_prompt},
                    {"role": "assistant", "content": example.model_output},
                ]
                full_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                data_list.append({"text": full_text})
        return Dataset.from_list(data_list)

    def _derive_response_template(self) -> str:
        """
        Dynamically derive the response template for loss masking.
        Refactored from _prepare_data_collator to return the string template.
        """
        messages = [{"role": "user", "content": "DETECT_RESPONSE_TEMPLATE"}]
        prompt_no_gen = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_with_gen = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response_template = ""
        if prompt_with_gen.startswith(prompt_no_gen):
            response_template = prompt_with_gen[len(prompt_no_gen) :]

        if not response_template.strip():
            if (
                self.config.model_type == ModelType.LLAMA3_8B
                or "llama-3" in self.config.model_type.value.lower()
            ):
                response_template = "<|start_header_id|>assistant<|end_header_id|>"
            elif "[/INST]" in prompt_no_gen:
                response_template = "[/INST]"
            else:
                response_template = "### Response:\n"

        return response_template

    def train(
        self,
        training_examples: list[LabeledExample],
        validation_examples: Optional[list[LabeledExample]] = None,
    ) -> None:
        try:
            self._prepare_peft_model()
            train_dataset = self._format_examples_for_training(training_examples)
            eval_dataset = None
            if validation_examples:
                eval_dataset = self._format_examples_for_training(validation_examples)

            # Derive the template string instead of creating a collator object
            response_template = self._derive_response_template()

            # Use SFTConfig instead of TrainingArguments
            # completion_only_loss=True enables the internal DataCollatorForCompletionOnlyLM
            training_args = SFTConfig(
                output_dir=str(self.config.output_dir / "checkpoints"),
                num_train_epochs=self.config.training_hyperparameters.num_epochs,
                per_device_train_batch_size=self.config.training_hyperparameters.batch_size,
                gradient_accumulation_steps=self.config.training_hyperparameters.gradient_accumulation_steps,
                learning_rate=self.config.training_hyperparameters.learning_rate,
                fp16=self.config.training_hyperparameters.fp16,
                logging_steps=self.config.training_hyperparameters.logging_steps,
                save_steps=self.config.training_hyperparameters.save_steps,
                eval_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                warmup_ratio=self.config.training_hyperparameters.warmup_ratio,
                optim=self.config.training_hyperparameters.optim,
                save_total_limit=3,
                report_to="none",
                remove_unused_columns=False,
                dataset_text_field="text",
                max_seq_length=self.config.training_hyperparameters.max_seq_length,
                # NEW Completion Logic:
                completion_only_loss=True,
                response_template=response_template,
            )

            trainer = SFTTrainer(
                model=self.peft_model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=training_args,
                # Note: SFTTrainer handles the collator internally when
                # completion_only_loss and response_template are in args.
                tokenizer=self.tokenizer,
            )

            trainer.train()

        except Exception as e:
            raise TrainingError(f"Training failed: {str(e)}") from e

    def _prepare_data_collator(self):
        """
        Configure the data collator with the correct response separator
        dynamically derived from the tokenizer.
        """
        # Create a dummy user message to detect the template structure
        # We use a placeholder content that won't confuse regex logic
        messages = [{"role": "user", "content": "DETECT_RESPONSE_TEMPLATE"}]

        # 1. Apply template WITHOUT generation prompt (just the user instruction)
        prompt_no_gen = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # 2. Apply template WITH generation prompt (includes the assistant start token)
        prompt_with_gen = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 3. Extract the response template (the difference between the two)
        response_template = ""
        if prompt_with_gen.startswith(prompt_no_gen):
            response_template = prompt_with_gen[len(prompt_no_gen) :]

        # 4. Fallback Logic:
        # If the diff is empty (common in older templates that don't support add_generation_prompt)
        # or if derivation failed, we try heuristics or standard templates.
        if not response_template.strip():
            print(
                "⚠ Warning: Could not dynamically derive response template via suffix diff."
            )

            # Explicit Model Type Checks for Known Architectures
            if (
                self.config.model_type == ModelType.LLAMA3_8B
                or "llama-3" in self.config.model_type.value.lower()
            ):
                # Llama 3 specific header token.
                # CRITICAL: Do NOT add newline here; Llama 3 uses special tokens.
                response_template = "<|start_header_id|>assistant<|end_header_id|>"
            elif "[/INST]" in prompt_no_gen:
                # Mistral / Llama 2 style where the closing tag IS the separator
                response_template = "[/INST]"
            else:
                # Generic fallback (Instruction format)
                response_template = "### Response:\n"

            print(f"  Falling back to heuristic: {repr(response_template)}")

        # Validate response template was successfully derived
        if not response_template or not response_template.strip():
            raise ValueError(
                "Failed to derive response template from tokenizer. "
                "Data collator cannot mask input tokens without a valid separator. "
                f"Model type: {self.config.model_type.value}"
            )

        print(
            f"✓ Data collator configured using response template: {repr(response_template)}"
        )

        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
            mlm=False,
        )

        return data_collator

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

            # 2. Tokenizer Hygiene (Matching Training)
            # Use EOS as PAD. Do NOT add new tokens or resize embeddings.
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Disable auto BOS (we rely on chat template)
            if hasattr(self.tokenizer, "add_bos_token"):
                self.tokenizer.add_bos_token = False

            # Configure for inference (left padding is crucial for generation)
            self.tokenizer.padding_side = "left"

            # 3. Load Base Model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_type.value,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=str(self.config.cache_dir),
            )

            # 4. Load LoRA Adapter
            print(f"Loading LoRA adapter from: {self.model_path}")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)

            # Ensure pad_token_id is synced
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
        """
        Single generation with strict slicing to return ONLY new tokens.
        """
        # Build messages and apply template
        messages = PromptTemplate.build_inference_messages(
            sentence, kb_text, context_before, context_after
        )
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Calculate input length for slicing later
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.inference_config.max_new_tokens,
                temperature=self.config.inference_config.temperature,
                do_sample=self.config.inference_config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Slice the output to exclude the input prompt
        generated_tokens = outputs[0][input_len:]

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def batch_generate(
        self,
        examples: list[LabeledExample],
        kb_text: str,
    ) -> list[str]:
        """
        Generate completions for a batch of examples.
        Uses apply_chat_template for consistency.
        """
        if not examples:
            return []

        prompts = []
        for ex in examples:
            if ex.input_prompt:
                # Wrap preformatted prompt in user message
                messages = [{"role": "user", "content": ex.input_prompt}]
                p = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
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

            input_ids = inputs.input_ids
            # Keep track of input lengths for each item in batch
            # (Note: due to left padding, simple slicing is harder, but since
            # we return new tokens only, we can use the input width relative to output)
            input_width = input_ids.shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
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
            for j, output_seq in enumerate(outputs):
                # Extract only the generated tokens
                generated_tokens = output_seq[input_width:]
                text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                decoded_batch.append(text)

            results.extend(decoded_batch)

        return results
