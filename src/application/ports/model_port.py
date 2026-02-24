"""Abstract model service port.

Defines the interface for all model adapters (loading, training,
inference, and persistence).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.entities import JDCSample, ParsedOutput


class IModelService(ABC):
    """Port (interface) for language model operations.

    Concrete implementations are responsible for all model-specific
    details (quantization, LoRA, tokenization, generation).
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load the base model and tokenizer from the configured source.

        Raises:
            ModelLoadError: If model or tokenizer initialization fails.
            CUDANotAvailableError: If no CUDA device is detected.
        """
        ...

    @abstractmethod
    def train(
        self,
        train_data: list[JDCSample],
        val_data: list[JDCSample],
    ) -> None:
        """Fine-tune the model on the provided training data.

        Args:
            train_data: Training samples as JDCSample objects.
            val_data: Validation samples for evaluation during training.

        Raises:
            ModelLoadError: If the model has not been loaded before training.
            RuntimeError: If the training loop encounters an unrecoverable error.
        """
        ...

    @abstractmethod
    def predict(self, prompt: str) -> ParsedOutput:
        """Run inference on a single input prompt.

        Args:
            prompt: The full input_prompt string to feed to the model.

        Returns:
            A ParsedOutput instance with the model's prediction.

        Raises:
            InferenceError: If generation fails or the output cannot be parsed.
            ModelLoadError: If the model has not been loaded.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fine-tuned model (LoRA adapter weights) to disk.

        Args:
            path: Directory path to save the adapter weights and tokenizer.

        Raises:
            OSError: If the target path cannot be written.
        """
        ...

    @abstractmethod
    def load_from_checkpoint(self, path: Path) -> None:
        """Load a previously saved LoRA adapter from a checkpoint directory.

        Args:
            path: Directory path containing saved adapter weights.

        Raises:
            ModelLoadError: If the checkpoint cannot be loaded.
        """
        ...
