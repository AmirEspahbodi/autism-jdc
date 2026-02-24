# CRITICAL: unsloth must be imported at module level before transformers so
# that all kernel patches are applied at process startup.
# Do NOT move this import below any other ML library import.
from __future__ import annotations

import unsloth  # noqa: F401

"""Unsloth + QLoRA model service implementation.

Wraps FastLanguageModel, SFTTrainer, and the full generation/parsing
pipeline into a single IModelService adapter.

CRITICAL: FastLanguageModel.from_pretrained() is used EXCLUSIVELY.
          AutoModelForCausalLM must never appear in this file.
"""

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from omegaconf import DictConfig
from pydantic import ValidationError
from transformers import PreTrainedTokenizerBase, TrainerCallback, TrainingArguments
from unsloth import FastLanguageModel  # type: ignore[import]

from src.application.ports.model_port import IModelService
from src.domain.entities import JDCSample, ParsedOutput
from src.domain.exceptions import InferenceError, ModelLoadError
from src.domain.value_objects import derive_label

# ---------------------------------------------------------------------------
# Custom Trainer Callback for per-step logging via loguru
# ---------------------------------------------------------------------------


class LoguruTrainerCallback(TrainerCallback):  # type: ignore[misc]
    """Routes HuggingFace Trainer logs through loguru.

    Args:
        None — instantiated without arguments.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log training metrics at each logging step.

        Args:
            args: TrainingArguments (unused but required by callback API).
            state: TrainerState (unused).
            control: TrainerControl (unused).
            logs: Dict of metric names → values from the Trainer.
            **kwargs: Additional keyword arguments (ignored).
        """
        if logs:
            logger.info(f"Trainer log: {logs}")


# ---------------------------------------------------------------------------
# Custom compute_metrics for SFTTrainer
# ---------------------------------------------------------------------------


def _build_compute_metrics(tokenizer: PreTrainedTokenizerBase) -> Any:
    """Build a compute_metrics function compatible with HuggingFace Trainer.

    The JDC evaluation during training only tracks loss; full F1 evaluation
    is handled by EvaluateUseCase.  This function simply passes through loss.

    Args:
        tokenizer: The model tokenizer (unused here, reserved for future use).

    Returns:
        A callable that accepts EvalPrediction and returns a dict.
    """

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        # SFTTrainer computes eval_loss automatically; we expose it as eval_f1
        # placeholder so load_best_model_at_end works with metric_for_best_model.
        # Real F1 is computed in EvaluateUseCase.
        return {"eval_f1": 0.0}

    return compute_metrics


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------


class UnslothModelService(IModelService):
    """IModelService implementation using Unsloth's FastLanguageModel.

    All model operations go through FastLanguageModel to preserve Unsloth's
    fused kernel optimisations and VRAM savings.

    Args:
        config: OmegaConf DictConfig with ``model``, ``lora``, ``training``,
                and ``evaluation`` sections.
    """

    def __init__(self, config: DictConfig) -> None:
        self._config = config
        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    # ------------------------------------------------------------------
    # IModelService interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the base model and tokenizer via FastLanguageModel.

        Raises:
            ModelLoadError: If FastLanguageModel.from_pretrained() fails.
            CUDANotAvailableError: If no CUDA device is present.
        """
        self._assert_cuda()
        model_cfg = self._config.model
        lora_cfg = self._config.lora

        logger.info(f"Loading model '{model_cfg.name}' with Unsloth FastLanguageModel …")

        try:
            dtype = torch.bfloat16 if str(model_cfg.dtype) == "bfloat16" else torch.float16

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(model_cfg.name),
                max_seq_length=int(model_cfg.max_seq_length),
                load_in_4bit=bool(model_cfg.load_in_4bit),
                dtype=dtype,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"FastLanguageModel.from_pretrained failed for '{model_cfg.name}': {exc}"
            ) from exc

        logger.info("Applying QLoRA via FastLanguageModel.get_peft_model …")

        try:
            model = FastLanguageModel.get_peft_model(
                model,
                r=int(lora_cfg.r),
                lora_alpha=int(lora_cfg.lora_alpha),
                lora_dropout=float(lora_cfg.lora_dropout),
                bias=str(lora_cfg.bias),
                target_modules=list(lora_cfg.target_modules),
                use_gradient_checkpointing="unsloth",  # Unsloth's own GC
            )
        except Exception as exc:
            raise ModelLoadError(f"FastLanguageModel.get_peft_model failed: {exc}") from exc

        self._model = model
        self._tokenizer = tokenizer

        # Ensure pad token is set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.debug("Set pad_token = eos_token because pad_token was None.")

        logger.info("Model and tokenizer loaded successfully.")

    def load_from_checkpoint(self, path: Path) -> None:
        """Load a LoRA adapter checkpoint on top of the already-loaded base model.

        The base model must have been loaded via load_model() first.

        Args:
            path: Directory containing the saved adapter weights.

        Raises:
            ModelLoadError: If the checkpoint directory is missing or loading fails.
        """
        if self._model is None or self._tokenizer is None:
            raise ModelLoadError("Base model has not been loaded. Call load_model() first.")

        if not path.exists():
            raise ModelLoadError(
                f"Checkpoint directory does not exist: {path}. "
                "Run training first to generate a checkpoint."
            )

        logger.info(f"Loading LoRA adapter from checkpoint: {path} …")

        try:
            from peft import PeftModel  # type: ignore[import]

            self._model = PeftModel.from_pretrained(
                self._model,
                str(path),
            )
            logger.info("Checkpoint loaded successfully.")
        except Exception as exc:
            raise ModelLoadError(f"Failed to load checkpoint from {path}: {exc}") from exc

    def train(
        self,
        train_data: list[JDCSample],
        val_data: list[JDCSample],
    ) -> None:
        """Run supervised fine-tuning using SFTTrainer.

        Args:
            train_data: Training samples.
            val_data: Validation samples used for eval_loss during training.

        Raises:
            ModelLoadError: If the model has not been loaded.
            RuntimeError: If the training loop encounters an error.
        """
        if self._model is None or self._tokenizer is None:
            raise ModelLoadError("Model must be loaded before training. Call load_model() first.")

        from trl import SFTConfig, SFTTrainer  # type: ignore[import]

        train_cfg = self._config.training
        model_cfg = self._config.model

        logger.info("Building HuggingFace Dataset objects for SFTTrainer …")
        train_dataset = self._build_hf_dataset(train_data)
        val_dataset = self._build_hf_dataset(val_data)
        logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}.")

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
            load_best_model_at_end=bool(train_cfg.load_best_model_at_end),
            metric_for_best_model=str(train_cfg.metric_for_best_model),
            dataloader_num_workers=int(train_cfg.dataloader_num_workers),
            max_seq_length=int(model_cfg.max_seq_length),
            dataset_text_field="text",
            packing=False,
        )

        trainer = SFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=sft_config,
            callbacks=[LoguruTrainerCallback()],
        )

        logger.info("Starting SFT training …")
        try:
            trainer.train()
        except Exception as exc:
            logger.exception(f"Training loop raised an unexpected error: {exc}")
            raise RuntimeError(f"Training failed: {exc}") from exc

        logger.info("SFT training complete.")
        # Keep the best model in self._model for immediate saving
        self._model = trainer.model

    def predict(self, prompt: str) -> ParsedOutput:
        """Generate a structured JSON output for a single prompt.

        Args:
            prompt: The full input_prompt string.

        Returns:
            A ParsedOutput instance with all fields set.

        Raises:
            ModelLoadError: If the model has not been loaded.
            InferenceError: If generation fails or the output cannot be parsed.
        """
        if self._model is None or self._tokenizer is None:
            raise ModelLoadError("Model must be loaded before inference. Call load_model() first.")

        eval_cfg = self._config.evaluation
        model_cfg = self._config.model

        # Switch model to inference mode (Unsloth-optimised)
        try:
            FastLanguageModel.for_inference(self._model)
        except Exception as exc:
            logger.warning(
                f"FastLanguageModel.for_inference failed: {exc}. "
                "Proceeding without Unsloth inference mode."
            )

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(model_cfg.max_seq_length),
        ).to("cuda")

        try:
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=int(eval_cfg.max_new_tokens),
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
        except Exception as exc:
            raise InferenceError(
                f"model.generate() failed: {exc}",
                raw_output="",
            ) from exc

        # Decode only the newly generated tokens (strip the prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_length:]
        raw_text: str = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        logger.debug(f"Raw generated text (first 300 chars): {raw_text[:300]!r}")

        return self._parse_generated_output(raw_text)

    def save(self, path: Path) -> None:
        """Save the LoRA adapter weights and tokenizer to ``path``.

        Args:
            path: Destination directory.

        Raises:
            ModelLoadError: If the model has not been loaded.
            OSError: If writing to ``path`` fails.
        """
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

    def _build_hf_dataset(self, samples: list[JDCSample]) -> Any:
        """Convert JDCSample list into a HuggingFace Dataset with a 'text' column.

        Each example is formatted as:
            <input_prompt>\\n<json-serialised model_output>

        The SFT training objective is to generate the JSON output given the prompt.

        Args:
            samples: JDCSample objects to convert.

        Returns:
            A HuggingFace datasets.Dataset with a single 'text' column.
        """
        from datasets import Dataset  # type: ignore[import]

        rows: list[dict[str, str]] = []
        truncated = 0
        model_cfg = self._config.model

        for sample in samples:
            # Re-serialise ParsedOutput to canonical JSON
            output_dict = sample.parsed_output.model_dump()
            output_json_str = json.dumps(output_dict)

            full_text = f"{sample.input_prompt}\n{output_json_str}"

            # Measure length and warn if truncation will occur
            token_ids = self._tokenizer(
                full_text,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]

            if len(token_ids) > int(model_cfg.max_seq_length):
                logger.debug(
                    f"Sample id='{sample.id}' exceeds max_seq_length "
                    f"({len(token_ids)} > {model_cfg.max_seq_length}). "
                    "It will be truncated during tokenization."
                )
                truncated += 1

            rows.append({"text": full_text})

        if truncated:
            logger.warning(
                f"{truncated}/{len(samples)} samples exceed max_seq_length "
                f"({model_cfg.max_seq_length}) and will be truncated."
            )

        return Dataset.from_list(rows)

    @staticmethod
    def _parse_generated_output(raw_text: str) -> ParsedOutput:
        """Extract and validate a ParsedOutput from raw generated text.

        Parsing strategy:
          1. Find the first '{' and the last '}' in the text.
          2. Extract the substring and parse with json.loads().
          3. Validate the parsed dict against ParsedOutput.
          4. Override is_ableist with derive_label(principle_id).

        Args:
            raw_text: The decoded model output string.

        Returns:
            A validated ParsedOutput instance.

        Raises:
            InferenceError: If no JSON object is found, parsing fails, or
                            validation fails.
        """
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")

        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            raise InferenceError(
                f"No JSON object found in generated output. "
                f"Raw text (first 300 chars): {raw_text[:300]!r}",
                raw_output=raw_text,
            )

        json_str = raw_text[start_idx : end_idx + 1]

        try:
            parsed_dict: dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise InferenceError(
                f"json.loads() failed on extracted substring: {exc}. "
                f"Substring (first 300 chars): {json_str[:300]!r}",
                raw_output=raw_text,
            ) from exc

        try:
            parsed_output = ParsedOutput.model_validate(parsed_dict)
        except ValidationError as exc:
            raise InferenceError(
                f"ParsedOutput validation failed: {exc}. Parsed dict: {parsed_dict}",
                raw_output=raw_text,
            ) from exc

        # Deterministically override is_ableist
        try:
            label = derive_label(parsed_output.principle_id)
        except ValueError as exc:
            raise InferenceError(
                f"derive_label failed for principle_id='{parsed_output.principle_id}': {exc}",
                raw_output=raw_text,
            ) from exc

        return ParsedOutput(
            justification_reasoning=parsed_output.justification_reasoning,
            evidence_quote=parsed_output.evidence_quote,
            principle_id=parsed_output.principle_id,
            principle_name=parsed_output.principle_name,
            is_ableist=label,
        )

    @staticmethod
    def _assert_cuda() -> None:
        """Assert CUDA is available or raise CUDANotAvailableError.

        Raises:
            CUDANotAvailableError: If torch.cuda.is_available() returns False.
        """
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
