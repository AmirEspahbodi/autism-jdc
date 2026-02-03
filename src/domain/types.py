"""
Domain types module - Pure domain entities with no external dependencies.

This module defines the core business entities of the JDC system.
Following Clean Architecture, these types have NO dependencies on
external libraries (no torch, transformers, pandas, etc.).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Principle:
    """Value object representing a knowledge base principle.

    A principle is an immutable rule that defines what constitutes
    ableist or non-ableist language in the neurodiversity context.

    Attributes:
        id: Unique identifier (e.g., "P0", "P1", "P2").
        name: Human-readable name of the principle.
        definition: Detailed explanation of the principle.

    Example:
        >>> principle = Principle(
        ...     id="P1",
        ...     name="Pathologizing Language",
        ...     definition="Language that frames neurodivergent traits as deficits..."
        ... )
    """

    id: str
    name: str
    definition: str

    def __post_init__(self) -> None:
        """Validate principle attributes."""
        if not self.id or not self.id.strip():
            raise ValueError("Principle ID cannot be empty")
        if not self.name or not self.name.strip():
            raise ValueError("Principle name cannot be empty")
        if not self.definition or not self.definition.strip():
            raise ValueError("Principle definition cannot be empty")


@dataclass(frozen=True)
class Justification:
    """Value object representing a model-generated justification.

    This is the core output of the JDC system. The model generates
    a structured justification that explains why a piece of text
    does or does not violate a principle.

    Attributes:
        principle_id: ID of the principle this justification refers to (e.g., "P0").
        justification_text: The reasoning explaining the classification.
        evidence_quote: Direct quote from input that supports the reasoning.

    Example:
        >>> justification = Justification(
        ...     principle_id="P1",
        ...     justification_text="The phrase 'suffers from autism' pathologizes...",
        ...     evidence_quote="He suffers from autism"
        ... )
    """

    principle_id: str
    justification_text: str
    evidence_quote: str

    def __post_init__(self) -> None:
        """Validate justification attributes."""
        if not self.principle_id or not self.principle_id.strip():
            raise ValueError("Principle ID cannot be empty")
        if not self.justification_text or not self.justification_text.strip():
            raise ValueError("Justification text cannot be empty")

    def is_ableist(self) -> bool:
        """Determine if this justification indicates ableist language.

        Deterministic mapping logic:
        - P0 (Neutral Language) -> NOT ableist (Label 0)
        - P1, P2, P3, P4 (Violations) -> Ableist (Label 1)

        Returns:
            True if the justification indicates ableist language, False otherwise.
        """
        return self.principle_id in {"P1", "P2", "P3", "P4"}

    def to_label(self) -> int:
        """Convert justification to binary label.

        Returns:
            1 if ableist, 0 if not ableist.
        """
        return 1 if self.is_ableist() else 0


@dataclass
class LabeledExample:
    """Entity representing a training or test instance.

    This combines input text with the expected justification output.
    Used for both fine-tuning and evaluation.

    Attributes:
        sentence: The target sentence to classify.
        context_before: Optional preceding context (for contextual understanding).
        context_after: Optional following context (for contextual understanding).
        ground_truth_justification: The expected justification.
        ground_truth_label: The binary label (0=not ableist, 1=ableist).
        metadata: Optional dictionary for additional information.

    Example:
        >>> example = LabeledExample(
        ...     sentence="People with autism are broken and need fixing.",
        ...     ground_truth_justification=Justification(
        ...         principle_id="P1",
        ...         justification_text="This pathologizes autism...",
        ...         evidence_quote="broken and need fixing"
        ...     ),
        ...     ground_truth_label=1
        ... )
    """

    sentence: str
    ground_truth_justification: Justification
    ground_truth_label: int
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    metadata: Optional[dict[str, str]] = None

    def __post_init__(self) -> None:
        """Validate labeled example attributes."""
        if not self.sentence or not self.sentence.strip():
            raise ValueError("Sentence cannot be empty")
        if self.ground_truth_label not in {0, 1}:
            raise ValueError("Label must be 0 or 1")

        # Ensure label matches justification
        expected_label = self.ground_truth_justification.to_label()
        if self.ground_truth_label != expected_label:
            raise ValueError(
                f"Label {self.ground_truth_label} does not match "
                f"justification principle {self.ground_truth_justification.principle_id} "
                f"(expected {expected_label})"
            )

    def get_full_context(self) -> str:
        """Get the full text including context.

        Returns:
            Combined text with context before, target sentence, and context after.
        """
        parts = []
        if self.context_before:
            parts.append(self.context_before)
        parts.append(self.sentence)
        if self.context_after:
            parts.append(self.context_after)
        return " ".join(parts)


@dataclass
class PredictionResult:
    """Entity representing a single prediction result.

    This captures all information about a model's prediction,
    including whether parsing was successful.

    Attributes:
        sentence: The input sentence.
        predicted_justification: The model's generated justification (if parsed).
        predicted_label: The derived binary label (if justification was parsed).
        ground_truth_justification: The expected justification.
        ground_truth_label: The true binary label.
        raw_output: The raw text output from the model.
        parsing_error: Error message if JSON parsing failed.
        is_correct: Whether the prediction matches ground truth.
    """

    sentence: str
    ground_truth_justification: Justification
    ground_truth_label: int
    raw_output: str
    predicted_justification: Optional[Justification] = None
    predicted_label: Optional[int] = None
    parsing_error: Optional[str] = None

    @property
    def is_correct(self) -> Optional[bool]:
        """Check if prediction is correct.

        Returns:
            True if correct, False if incorrect, None if parsing failed.
        """
        if self.predicted_label is None:
            return None
        return self.predicted_label == self.ground_truth_label

    @property
    def has_parsing_error(self) -> bool:
        """Check if there was a parsing error."""
        return self.parsing_error is not None


@dataclass
class EvaluationMetrics:
    """Entity representing evaluation metrics.

    Attributes:
        precision: Precision score (TP / (TP + FP)).
        recall: Recall score (TP / (TP + FN)).
        f1_score: F1 score (harmonic mean of precision and recall).
        accuracy: Overall accuracy.
        true_positives: Count of true positives.
        false_positives: Count of false positives.
        true_negatives: Count of true negatives.
        false_negatives: Count of false negatives.
        total_examples: Total number of examples.
        parsing_failures: Number of examples where parsing failed.
    """

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
        """Convert metrics to dictionary."""
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
