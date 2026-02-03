"""
Domain interfaces module - Abstract ports for the JDC system.

This module defines the abstract interfaces (ports) that the infrastructure
layer must implement. Following Clean Architecture, these are pure abstractions
with no implementation details.
"""

from abc import ABC, abstractmethod
from typing import Any

from src.domain.types import (
    EvaluationMetrics,
    Justification,
    LabeledExample,
    PredictionResult,
)


class LLMTrainer(ABC):
    """Abstract interface for fine-tuning large language models.

    This port defines the contract for training adapters. Different
    implementations might use different frameworks (HuggingFace, JAX, etc.)
    but all must conform to this interface.
    """

    @abstractmethod
    def train(
        self,
        training_examples: list[LabeledExample],
        validation_examples: list[LabeledExample] | None = None,
    ) -> None:
        """Fine-tune the model on the provided examples.

        Args:
            training_examples: List of labeled examples for training.
            validation_examples: Optional list of examples for validation.

        Raises:
            TrainingError: If training fails for any reason.
        """
        pass

    @abstractmethod
    def save_model(self, output_path: str) -> None:
        """Save the fine-tuned model to disk.

        Args:
            output_path: Path where the model should be saved.

        Raises:
            IOError: If saving fails.
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load a previously fine-tuned model.

        Args:
            model_path: Path to the saved model.

        Raises:
            IOError: If loading fails.
        """
        pass


class InferenceEngine(ABC):
    """Abstract interface for generating text from a fine-tuned model.

    This port defines the contract for inference adapters. The implementation
    is responsible for managing model loading, generation, and resource cleanup.
    """

    @abstractmethod
    def generate_justification(
        self,
        sentence: str,
        context_before: str | None = None,
        context_after: str | None = None,
        knowledge_base_text: str | None = None,
    ) -> str:
        """Generate a justification for the given sentence.

        Args:
            sentence: The target sentence to analyze.
            context_before: Optional preceding context.
            context_after: Optional following context.
            knowledge_base_text: The symbolic knowledge base as text.

        Returns:
            Raw text output from the model (should be JSON).

        Raises:
            InferenceError: If generation fails.
        """
        pass

    @abstractmethod
    def batch_generate(
        self,
        examples: list[LabeledExample],
        knowledge_base_text: str,
    ) -> list[str]:
        """Generate justifications for a batch of examples.

        Args:
            examples: List of labeled examples.
            knowledge_base_text: The symbolic knowledge base as text.

        Returns:
            List of raw text outputs (one per example).

        Raises:
            InferenceError: If generation fails.
        """
        pass


class JustificationParser(ABC):
    """Abstract interface for parsing model outputs into structured justifications.

    This port defines the contract for parsing adapters. The implementation
    must handle malformed JSON, markdown formatting, and other edge cases.
    """

    @abstractmethod
    def parse(self, raw_output: str) -> Justification:
        """Parse raw model output into a structured Justification.

        Args:
            raw_output: Raw text output from the model.

        Returns:
            Parsed Justification object.

        Raises:
            ParsingError: If the output cannot be parsed.
        """
        pass

    @abstractmethod
    def validate_json_schema(self, data: dict[str, Any]) -> bool:
        """Validate that parsed JSON matches the expected schema.

        Args:
            data: Parsed JSON data.

        Returns:
            True if schema is valid, False otherwise.
        """
        pass


class MetricsRepository(ABC):
    """Abstract interface for computing and storing evaluation metrics.

    This port defines the contract for metrics adapters. The implementation
    is responsible for calculating F1, precision, recall, and other metrics.
    """

    @abstractmethod
    def compute_metrics(
        self,
        predictions: list[PredictionResult],
    ) -> EvaluationMetrics:
        """Compute evaluation metrics from prediction results.

        Args:
            predictions: List of prediction results.

        Returns:
            Computed evaluation metrics.
        """
        pass

    @abstractmethod
    def save_metrics(
        self,
        metrics: EvaluationMetrics,
        output_path: str,
    ) -> None:
        """Save metrics to disk (e.g., as JSON).

        Args:
            metrics: Metrics to save.
            output_path: Path where metrics should be saved.

        Raises:
            IOError: If saving fails.
        """
        pass


class DataLoader(ABC):
    """Abstract interface for loading training and test data.

    This port defines the contract for data loading adapters.
    """

    @abstractmethod
    def load_training_data(self) -> list[LabeledExample]:
        """Load training examples.

        Returns:
            List of labeled training examples.

        Raises:
            DataLoadError: If loading fails.
        """
        pass

    @abstractmethod
    def load_test_data(self) -> list[LabeledExample]:
        """Load test examples.

        Returns:
            List of labeled test examples.

        Raises:
            DataLoadError: If loading fails.
        """
        pass


class ReportGenerator(ABC):
    """Abstract interface for generating evaluation reports.

    This port defines the contract for report generation adapters.
    """

    @abstractmethod
    def generate_report(
        self,
        predictions: list[PredictionResult],
        metrics: EvaluationMetrics,
        output_path: str,
    ) -> None:
        """Generate a detailed evaluation report.

        Args:
            predictions: All prediction results.
            metrics: Computed metrics.
            output_path: Path where report should be saved.

        Raises:
            IOError: If report generation fails.
        """
        pass


# Custom exceptions for domain layer
class DomainException(Exception):
    """Base exception for domain-level errors."""

    pass


class TrainingError(DomainException):
    """Raised when training fails."""

    pass


class InferenceError(DomainException):
    """Raised when inference fails."""

    pass


class ParsingError(DomainException):
    """Raised when parsing fails."""

    pass


class DataLoadError(DomainException):
    """Raised when data loading fails."""

    pass
