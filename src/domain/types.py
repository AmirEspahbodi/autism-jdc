"""
Domain types module - Pure domain entities with no external dependencies.
Refactored to support Pre-formatted SFT datasets.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Principle:
    """Value object representing a knowledge base principle."""

    id: str
    name: str
    definition: str

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("Principle ID cannot be empty")
        if not self.name or not self.name.strip():
            raise ValueError("Principle name cannot be empty")
        if not self.definition or not self.definition.strip():
            raise ValueError("Principle definition cannot be empty")


@dataclass(frozen=True)
class Justification:
    """Value object representing a model-generated justification."""

    principle_id: str
    justification_text: str
    evidence_quote: str

    def __post_init__(self) -> None:
        if not self.principle_id or not self.principle_id.strip():
            raise ValueError("Principle ID cannot be empty")
        if not self.justification_text or not self.justification_text.strip():
            raise ValueError("Justification text cannot be empty")

    def is_ableist(self) -> bool:
        return self.principle_id in {"P1", "P2", "P3", "P4"}

    def to_label(self) -> int:
        return 1 if self.is_ableist() else 0


@dataclass
class LabeledExample:
    """Entity representing a training or test instance.

    Refactored to support polymorphic data:
    1. Legacy: Structured data (sentence, ground_truth_justification)
    2. SFT: Pre-formatted strings (input_prompt, model_output)
    """

    # SFT Pre-formatted Fields
    input_prompt: Optional[str] = None
    model_output: Optional[str] = None

    # Legacy Structured Fields (Made Optional)
    sentence: Optional[str] = None
    ground_truth_justification: Optional[Justification] = None
    ground_truth_label: Optional[int] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    metadata: Optional[dict[str, str]] = None

    def __post_init__(self) -> None:
        """Validate that either SFT fields OR Legacy fields are present."""

        # Mode 1: Pre-formatted SFT
        if self.input_prompt is not None and self.model_output is not None:
            if not self.input_prompt.strip():
                raise ValueError("input_prompt cannot be empty")
            return  # Valid SFT example

        # Mode 2: Legacy Structured Data
        if (
            self.sentence is not None
            and self.ground_truth_justification is not None
            and self.ground_truth_label is not None
        ):
            if not self.sentence.strip():
                raise ValueError("Sentence cannot be empty")
            if self.ground_truth_label not in {0, 1}:
                raise ValueError("Label must be 0 or 1")

            # Ensure label matches justification
            expected_label = self.ground_truth_justification.to_label()
            if self.ground_truth_label != expected_label:
                raise ValueError(
                    f"Label {self.ground_truth_label} does not match "
                    f"justification principle {self.ground_truth_justification.principle_id}"
                )
            return  # Valid Legacy example

        raise ValueError(
            "LabeledExample must contain either (input_prompt, model_output) "
            "OR (sentence, ground_truth_justification, ground_truth_label)."
        )

    def get_full_context(self) -> str:
        """Get the full text. Returns input_prompt if available."""
        if self.input_prompt:
            return self.input_prompt

        parts = []
        if self.context_before:
            parts.append(self.context_before)
        if self.sentence:
            parts.append(self.sentence)
        if self.context_after:
            parts.append(self.context_after)
        return " ".join(parts)


@dataclass
class PredictionResult:
    """Entity representing a single prediction result."""

    sentence: str
    ground_truth_justification: Justification
    ground_truth_label: int
    raw_output: str
    predicted_justification: Optional[Justification] = None
    predicted_label: Optional[int] = None
    parsing_error: Optional[str] = None

    @property
    def is_correct(self) -> Optional[bool]:
        if self.predicted_label is None:
            return None
        return self.predicted_label == self.ground_truth_label

    @property
    def has_parsing_error(self) -> bool:
        return self.parsing_error is not None


@dataclass
class EvaluationMetrics:
    """Entity representing evaluation metrics."""

    precision: float
    recall: float
    f1_score: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_examples: int
    parsing_failures: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "total_examples": self.total_examples,
            "parsing_failures": self.parsing_failures,
        }
