import json
from pathlib import Path
from typing import Any, Optional

import torch

# --- CRITICAL FIX: UNSLOTH MUST BE IMPORTED BEFORE TRANSFORMERS/PEFT/TRL ---
try:
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
# ---------------------------------------------------------------------------

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
from trl import SFTConfig, SFTTrainer

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
    """Adapter for fine-tuning LLMs using LoRA.

    Refactored to support Unsloth's FastLanguageModel for optimized 4-bit QLoRA
    while maintaining standard HuggingFace support for other models.
    """

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.model: Any = None  # Union[AutoModelForCausalLM, FastLanguageModel]
        self.tokenizer: Optional[AutoTokenizer] = None
        self.peft_model: Any = None
        self._initialize_model()

    def _is_unsloth_model(self) -> bool:
        """Check if the configured model is an Unsloth variant."""
        return "unsloth" in self.config.model_type.value.lower() or (
            hasattr(self.config.model_type, "name")
            and "unsloth" in self.config.model_type.name.lower()
        )

    def _initialize_model(self) -> None:
        print(f"Loading base model: {self.config.model_type.value}")

        # --- PATH A: UNSLOTH OPTIMIZED LOADING ---
        if self._is_unsloth_model():
            if not UNSLOTH_AVAILABLE:
                raise ImportError(
                    "ModelType requires 'unsloth' library, but it is not installed."
                )

            print("⚡ Using Unsloth FastLanguageModel optimization")

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_type.value,
                max_seq_length=self.config.training_hyperparameters.max_seq_length,
                dtype=None,  # Auto-detection (float16/bfloat16)
                load_in_4bit=True,  # Force 4-bit for QLoRA
                token=self.config.hf_token,
                # device_map="auto" is handled internally by Unsloth usually,
                # but explicit passing is sometimes deprecated in their API.
                # We rely on their defaults.
            )

            # Ensure tokenizer hygiene consistent with our pipeline expectations
            # Unsloth usually sets this up, but we enforce specific needs
            if (
                hasattr(self.tokenizer, "pad_token")
                and self.tokenizer.pad_token is None
            ):
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Disable manual BOS if the attribute exists (Unsloth tokenizers might differ)
            if hasattr(self.tokenizer, "add_bos_token"):
                self.tokenizer.add_bos_token = False

            return

        # --- PATH B: STANDARD HUGGING FACE / BITSANDBYTES ---

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

        # Mistral v0.3 defaults to bfloat16 in config.json.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_type.value,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=str(self.config.cache_dir),
            token=self.config.hf_token,
            torch_dtype=torch.float16,
        )

        # --- CRITICAL FIX FOR T4 GPU / MISTRAL v0.3 ---
        self.model.config.torch_dtype = torch.float16
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        # ----------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_type.value,
            trust_remote_code=True,
            cache_dir=str(self.config.cache_dir),
            token=self.config.hf_token,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "right"

        if hasattr(self.tokenizer, "add_bos_token"):
            self.tokenizer.add_bos_token = False

    def _prepare_peft_model(self) -> None:
        """Applies LoRA adapters to the model."""

        # --- PATH A: UNSLOTH ADAPTERS ---
        if self._is_unsloth_model():
            print("⚡ Attaching Unsloth LoRA adapters...")
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.lora_config.r,
                target_modules=self.config.lora_config.target_modules,
                lora_alpha=self.config.lora_config.lora_alpha,
                lora_dropout=self.config.lora_config.lora_dropout,
                bias=self.config.lora_config.bias,
                # Optimization specific to Unsloth
                use_gradient_checkpointing="unsloth",
                random_state=self.config.seed,
            )
            # Unsloth returns the model directly ready for training
            self.peft_model = self.model
            return

        # --- PATH B: STANDARD PEFT ---
        print("Standard PEFT: Preparing model for k-bit training...")
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
        """
        Formats examples into a dataset with a 'messages' column.
        SFTTrainer with completion_only_loss=True will automatically handle
        chat templating and masking for the assistant's turn.
        """
        data_list = []
        for example in examples:
            if example.input_prompt is not None and example.model_output is not None:
                # Structure as standard chat messages
                messages = [
                    {"role": "user", "content": example.input_prompt},
                    {"role": "assistant", "content": example.model_output},
                ]
                data_list.append({"messages": messages})

        return Dataset.from_list(data_list)

    # In src/infrastructure/llm.py

    def train(
        self,
        training_examples: list[LabeledExample],
        validation_examples: Optional[list[LabeledExample]] = None,
    ) -> None:
        self._prepare_peft_model()
        train_dataset = self._format_examples_for_training(training_examples)
        eval_dataset = None
        if validation_examples:
            eval_dataset = self._format_examples_for_training(validation_examples)

        training_args = SFTConfig(
            output_dir=str(self.config.output_dir / "checkpoints"),
            num_train_epochs=self.config.training_hyperparameters.num_epochs,
            per_device_train_batch_size=self.config.training_hyperparameters.batch_size,
            gradient_accumulation_steps=self.config.training_hyperparameters.gradient_accumulation_steps,
            learning_rate=self.config.training_hyperparameters.learning_rate,
            fp16=self.config.training_hyperparameters.fp16,
            bf16=False,
            logging_steps=self.config.training_hyperparameters.logging_steps,
            save_steps=self.config.training_hyperparameters.save_steps,
            eval_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            warmup_ratio=self.config.training_hyperparameters.warmup_ratio,
            optim=self.config.training_hyperparameters.optim,
            save_total_limit=3,
            report_to="none",
            remove_unused_columns=False,
            max_length=self.config.training_hyperparameters.max_seq_length,
            dataset_text_field="text",
        )

        def formatting_prompts_func(examples):
            convos = examples["messages"]
            texts = [
                self.tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False
                )
                for convo in convos
            ]
            return texts

        trainer = SFTTrainer(
            model=self.peft_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            processing_class=self.tokenizer,
            formatting_func=formatting_prompts_func,  # <--- Added this argument
        )

        trainer.train()

    def save_model(self, output_path: str) -> None:
        if self.peft_model is None:
            raise IOError("No trained model to save")
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Unsloth models support save_pretrained exactly like PEFT/Transformers
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
                token=self.config.hf_token,
            )

            # 2. Tokenizer Hygiene (Matching Training)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Disable auto BOS (we rely on chat template)
            if hasattr(self.tokenizer, "add_bos_token"):
                self.tokenizer.add_bos_token = False

            self.tokenizer.padding_side = "left"

            # 3. Load Base Model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_type.value,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=str(self.config.cache_dir),
                token=self.config.hf_token,
                torch_dtype=torch.float16,
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
        messages = PromptTemplate.build_inference_messages(
            sentence, kb_text, context_before, context_after
        )
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.inference_config.max_new_tokens,
                temperature=self.config.inference_config.temperature,
                do_sample=self.config.inference_config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def batch_generate(
        self,
        examples: list[LabeledExample],
        kb_text: str,
    ) -> list[str]:
        """
        Generate completions for a batch of examples.
        """
        if not examples:
            return []

        prompts = []
        for ex in examples:
            if ex.input_prompt:
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
                generated_tokens = output_seq[input_width:]
                text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                decoded_batch.append(text)

            results.extend(decoded_batch)

        return results
