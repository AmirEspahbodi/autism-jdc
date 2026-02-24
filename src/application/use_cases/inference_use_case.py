"""Single-sample inference use case."""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.application.ports.model_port import IModelService
from src.domain.entities import ParsedOutput
from src.domain.value_objects import derive_label


class InferenceUseCase:
    """Runs inference on a single input_prompt and returns a ParsedOutput.

    The use case loads the model from a checkpoint if not already loaded,
    runs generation, and guarantees that the returned is_ableist label is
    always derived deterministically from principle_id.

    Args:
        model_service: Concrete implementation of IModelService.
        checkpoint_path: Path to a saved LoRA adapter checkpoint.
    """

    def __init__(
        self,
        model_service: IModelService,
        checkpoint_path: Path,
    ) -> None:
        self._model_service = model_service
        self._checkpoint_path = checkpoint_path
        self._model_loaded: bool = False

    def execute(self, input_prompt: str) -> ParsedOutput:
        """Run inference on a single prompt.

        Args:
            input_prompt: The full input_prompt string (as stored in the dataset).

        Returns:
            ParsedOutput with all fields populated and is_ableist derived
            deterministically from principle_id.

        Raises:
            InferenceError: If generation or JSON parsing fails.
            ModelLoadError: If checkpoint cannot be loaded.
        """
        if not self._model_loaded:
            logger.info(
                f"InferenceUseCase: loading model from {self._checkpoint_path} …"
            )
            self._model_service.load_model()
            self._model_service.load_from_checkpoint(self._checkpoint_path)
            self._model_loaded = True

        logger.debug("InferenceUseCase: running prediction …")
        raw_pred: ParsedOutput = self._model_service.predict(input_prompt)

        # Always override is_ableist with the deterministic derivation
        final_pred = ParsedOutput(
            justification_reasoning=raw_pred.justification_reasoning,
            evidence_quote=raw_pred.evidence_quote,
            principle_id=raw_pred.principle_id,
            principle_name=raw_pred.principle_name,
            is_ableist=derive_label(raw_pred.principle_id),
        )

        logger.debug(
            f"InferenceUseCase: result principle_id={final_pred.principle_id}, "
            f"is_ableist={final_pred.is_ableist}"
        )
        return final_pred
