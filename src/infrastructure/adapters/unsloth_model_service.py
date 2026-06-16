# CRITICAL: unsloth must be imported at module level before transformers so
# that all kernel patches are applied at process startup.
# Do NOT move this import below any other ML library import.
from __future__ import annotations

import unsloth  # noqa: F401

"""Unsloth + QLoRA model service implementation.

Wraps FastLanguageModel, SFTTrainer, and the full generation/parsing
pipeline into a single IModelService adapter.
"""

import json
import re
from pathlib import Path
from typing import Any

import json_repair  # Robust JSON parsing for LLMs
import torch
from loguru import logger
from omegaconf import DictConfig
from pydantic import ValidationError
from transformers import PreTrainedTokenizerBase, TrainerCallback, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer  # type: ignore[import]

# Unsloth & TRL imports
from unsloth import FastLanguageModel  # type: ignore[import]
from unsloth.chat_templates import get_chat_template  # type: ignore[import]

from src.application.ports.model_port import IModelService
from src.domain.entities import JDCSample, ParsedOutput
from src.domain.exceptions import InferenceError, ModelLoadError
from src.domain.value_objects import derive_label

# ---------------------------------------------------------------------------
# Custom Trainer Callback for per-step logging via loguru
# ---------------------------------------------------------------------------


class LoguruTrainerCallback(TrainerCallback):  # type: ignore[misc]
    """Routes HuggingFace Trainer logs through loguru."""

    def on_log(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs:
            logger.info(f"Trainer log: {logs}")


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------


class UnslothModelService(IModelService):
    """IModelService implementation using Unsloth's FastLanguageModel."""

    def __init__(self, config: DictConfig, model_local_dir: Path | None = None) -> None:
        self._config = config
        self._model_local_dir = model_local_dir
        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    # ------------------------------------------------------------------
    # IModelService interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        self._assert_cuda()
        model_cfg = self._config.model
        lora_cfg = self._config.lora

        model_source = (
            str(self._model_local_dir.resolve()) if self._model_local_dir else str(model_cfg.name)
        )
        logger.info(f"Loading model from: {model_source}")

        try:
            dtype = torch.bfloat16 if str(model_cfg.dtype) == "bfloat16" else torch.float16

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_source,
                max_seq_length=int(model_cfg.max_seq_length),
                load_in_4bit=bool(model_cfg.load_in_4bit),
                dtype=dtype,
            )

            # --- Apply Llama-3 Chat Template natively to the Tokenizer ---
            logger.info("Applying Llama-3 chat template to tokenizer...")
            tokenizer = get_chat_template(
                tokenizer,
                chat_template="llama-3",
                mapping={
                    "role": "role",
                    "content": "content",
                    "user": "user",
                    "assistant": "assistant",
                },
            )

        except Exception as exc:
            raise ModelLoadError(f"FastLanguageModel.from_pretrained failed: {exc}") from exc

        logger.info("Applying QLoRA via FastLanguageModel.get_peft_model …")

        try:
            model = FastLanguageModel.get_peft_model(
                model,
                r=int(lora_cfg.r),
                lora_alpha=int(lora_cfg.lora_alpha),
                lora_dropout=float(lora_cfg.lora_dropout),
                bias=str(lora_cfg.bias),
                target_modules=list(lora_cfg.target_modules),
                use_gradient_checkpointing="unsloth",
            )
        except Exception as exc:
            raise ModelLoadError(f"FastLanguageModel.get_peft_model failed: {exc}") from exc

        self._model = model
        self._tokenizer = tokenizer

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.debug("Set pad_token = eos_token because pad_token was None.")

        logger.info("Model and tokenizer loaded successfully.")

    def load_from_checkpoint(self, path: Path) -> None:
        if self._model is None or self._tokenizer is None:
            raise ModelLoadError("Base model has not been loaded. Call load_model() first.")
        if not path.exists():
            raise ModelLoadError(f"Checkpoint directory does not exist: {path}.")

        logger.info(f"Loading LoRA adapter from checkpoint: {path} …")
        try:
            from peft import PeftModel  # type: ignore[import]

            self._model = PeftModel.from_pretrained(self._model, str(path))
            logger.info("Checkpoint loaded successfully.")
        except Exception as exc:
            raise ModelLoadError(f"Failed to load checkpoint: {exc}") from exc

    def train(self, train_data: list[JDCSample], val_data: list[JDCSample]) -> None:
        if self._model is None or self._tokenizer is None:
            raise ModelLoadError("Model must be loaded before training. Call load_model() first.")

        train_cfg = self._config.training
        model_cfg = self._config.model

        logger.info("Building HuggingFace Dataset objects for SFTTrainer …")
        train_dataset = self._build_hf_dataset(train_data)
        val_dataset = self._build_hf_dataset(val_data)
        logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}.")

        # --- Loss Alignment via Completion-Only Data Collator ---
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        try:
            collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template,
                tokenizer=self._tokenizer,
            )
        except Exception as exc:
            raise RuntimeError(f"Data Collator initialization failed: {exc}") from exc

        sft_config = SFTConfig(
            output_dir=str(train_cfg.output_dir),
            num_train_epochs=int(train_cfg.num_train_epochs),
            per_device_train_batch_size=int(train_cfg.per_device_train_batch_size),
            per_device_eval_batch_size=int(train_cfg.per_device_train_batch_size),
            gradient_accumulation_steps=int(train_cfg.gradient_accumulation_steps),
            learning_rate=float(train_cfg.learning_rate),
            lr_scheduler_type=str(train_cfg.lr_scheduler_type),
            warmup_ratio=float(train_cfg.warmup_ratio),
            weight_decay=float(train_cfg.weight_decay),
            optim=str(train_cfg.optim),
            bf16=bool(train_cfg.bf16),
            fp16=bool(train_cfg.fp16),
            logging_steps=int(train_cfg.logging_steps),
            eval_strategy=str(train_cfg.eval_strategy),
            save_strategy=str(train_cfg.save_strategy),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=int(train_cfg.dataloader_num_workers),
            max_seq_length=int(model_cfg.max_seq_length),
            dataset_text_field="text",
            packing=False,  # MUST be false for the Completion-Only collator to work
        )

        trainer = SFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=sft_config,
            data_collator=collator,
            callbacks=[LoguruTrainerCallback()],
        )

        logger.info("Starting SFT training …")
        try:
            trainer.train()
        except Exception as exc:
            logger.exception(f"Training loop raised an unexpected error: {exc}")
            raise RuntimeError(f"Training failed: {exc}") from exc

        logger.info("SFT training complete.")
        self._model = trainer.model

    def predict(self, prompt: str) -> ParsedOutput:
        """Refactored inference pipeline with injection protection and robust generation."""
        if self._model is None or self._tokenizer is None:
            raise ModelLoadError("Model must be loaded before inference. Call load_model() first.")

        eval_cfg = self._config.evaluation
        model_cfg = self._config.model

        try:
            FastLanguageModel.for_inference(self._model)
        except Exception as exc:
            logger.warning(f"FastLanguageModel.for_inference failed: {exc}.")

        # --- Sanitize against prompt injection ---
        safe_prompt = self._sanitize_input(prompt)

        # --- Apply Inference Chat Template ---
        messages = [{"role": "user", "content": safe_prompt}]
        formatted_prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(model_cfg.max_seq_length),
        ).to("cuda")

        try:
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    # Buffer max_new_tokens to prevent frequent cutoffs during JSON formatting
                    max_new_tokens=int(eval_cfg.max_new_tokens) + 256,
                    do_sample=False,  # Enforces Greedy Decoding
                    # Note: temperature is inherently ignored when do_sample=False.
                    pad_token_id=self._tokenizer.eos_token_id,
                )
        except Exception as exc:
            raise InferenceError(f"model.generate() failed: {exc}", raw_output="") from exc

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_length:]
        raw_text: str = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        logger.debug(f"Raw generated text (first 300 chars): {raw_text[:300]!r}")
        return self._parse_generated_output(raw_text)

    def save(self, path: Path) -> None:
        if self._model is None or self._tokenizer is None:
            raise ModelLoadError("Cannot save: model has not been loaded.")

        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model adapter to {path} …")

        try:
            self._model.save_pretrained(str(path))
            self._tokenizer.save_pretrained(str(path))
        except Exception as exc:
            raise OSError(f"Failed to save model to {path}: {exc}") from exc

        logger.info(f"Model saved to {path}.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sanitize_input(self, text: str) -> str:
        """Strips Llama-3 control tokens from raw text.
        Prevents prompt injection boundaries from triggering premature generation.
        """
        dangerous_tokens = [
            "<|eot_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_",
        ]
        for token in dangerous_tokens:
            text = text.replace(token, "")
        return text.strip()

    def _build_hf_dataset(self, samples: list[JDCSample]) -> Any:
        from datasets import Dataset  # type: ignore[import]

        rows: list[dict[str, str]] = []
        truncated = 0
        model_cfg = self._config.model

        for sample in samples:
            output_dict = sample.parsed_output.model_dump()
            output_json_str = json.dumps(output_dict, ensure_ascii=False)

            # --- Format training text with Llama-3 Chat Template ---
            messages = [
                {"role": "user", "content": sample.input_prompt},
                {"role": "assistant", "content": output_json_str},
            ]
            full_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # Measure length natively BEFORE tokenization to prevent Trainer from truncating JSON ends
            token_ids = self._tokenizer(full_text, truncation=False)["input_ids"]

            if len(token_ids) > int(model_cfg.max_seq_length):
                logger.debug(
                    f"Sample id='{sample.id}' exceeds max_seq_length. Dropping to prevent target corruption."
                )
                truncated += 1
                continue

            rows.append({"text": full_text})

        if truncated:
            logger.warning(
                f"Dropped {truncated}/{len(samples)} samples because they exceeded max_seq_length "
                f"({model_cfg.max_seq_length})."
            )

        return Dataset.from_list(rows)

    @staticmethod
    def _parse_generated_output(raw_text: str) -> ParsedOutput:
        """Extract and validate a ParsedOutput from raw generated text safely using json_repair and Semantic Salvage."""
        try:
            # 1. Use json_repair to intelligently reconstruct missing braces, quotes, and trailing text.
            parsed_dict: Any = json_repair.loads(raw_text)

            if not isinstance(parsed_dict, dict):
                raise ValueError(f"Decoded JSON is not a dictionary. Got type: {type(parsed_dict)}")

            # 2. Map payload into structured Pydantic object
            parsed_output = ParsedOutput.model_validate(parsed_dict)
            label = derive_label(parsed_output.principle_id)

            return ParsedOutput(
                justification_reasoning=parsed_output.justification_reasoning,
                evidence_quote=parsed_output.evidence_quote,
                principle_id=parsed_output.principle_id,
                principle_name=parsed_output.principle_name,
                is_ableist=label,
            )

        except (ValueError, ValidationError) as primary_exc:
            # 3. SEMANTIC SALVAGE STRATEGY
            # If the JSON is fundamentally broken beyond repair, we scan the text for a valid
            # principle classification so we don't skew False Negative evaluation metrics by defaulting to P0.
            logger.warning(
                f"JSON validation failed, attempting Semantic Salvage... Error: {primary_exc}"
            )

            salvage_match = re.search(r"(P[0-4])", raw_text)

            if salvage_match:
                salvaged_principle = salvage_match.group(1)
                logger.info(f"Successfully salvaged principle: {salvaged_principle}")
                return ParsedOutput(
                    justification_reasoning=f"[SALVAGED] Partial reasoning: {raw_text[:200]}...",
                    evidence_quote="[SALVAGED]",
                    principle_id=salvaged_principle,
                    principle_name="Salvaged Principle",
                    is_ableist=derive_label(salvaged_principle),
                )

            # 4. If all fails, raise the Inference Error
            raise InferenceError(
                f"Fatal parse failure. Could not salvage principle. {primary_exc}",
                raw_output=raw_text,
            ) from primary_exc

    @staticmethod
    def _assert_cuda() -> None:
        from src.domain.exceptions import CUDANotAvailableError

        if not torch.cuda.is_available():
            raise CUDANotAvailableError(
                "CUDA is not available on this machine. "
                "JDC training requires an NVIDIA GPU with CUDA 12.6+. "
                f"torch.cuda.is_available() returned False. "
                f"torch version: {torch.__version__}."
            )
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA device detected: {device_name}")
