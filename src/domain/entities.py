"""Domain entities: core Pydantic v2 data models for the JDC project.

These models represent the fundamental business objects.  They have
ZERO dependencies on infrastructure or application layers.
"""
from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator

from src.domain.value_objects import PrincipleID, derive_label


class KBPrinciple(BaseModel):
    """A single Knowledge Base principle describing a category of ableism.

    Args:
        principle_id: Unique identifier such as "P0" through "P4".
        principle_name: Short human-readable name.
        definition: Prose definition of the principle.
        examples: List of concrete example phrases.
    """

    principle_id: str
    principle_name: str
    definition: str
    examples: list[str]

    @field_validator("principle_id")
    @classmethod
    def validate_principle_id(cls, v: str) -> str:
        """Ensure the principle_id is a valid PrincipleID value.

        Args:
            v: Raw principle_id string.

        Returns:
            Validated principle_id string.

        Raises:
            ValueError: If v is not in {P0, P1, P2, P3, P4}.
        """
        try:
            PrincipleID(v)
        except ValueError as exc:
            raise ValueError(
                f"principle_id must be one of {[p.value for p in PrincipleID]}, got '{v}'"
            ) from exc
        return v


class RawSample(BaseModel):
    """A dataset record exactly as loaded from the JSON file.

    The model_output field is intentionally kept as a raw string; callers
    are responsible for parsing it with json.loads().

    Args:
        id: Unique sample identifier.
        input_prompt: Full prompt string fed to the model.
        model_output: JSON object encoded as a string.
    """

    id: str
    input_prompt: str
    model_output: str


class ParsedOutput(BaseModel):
    """The structured content of a model_output string after parsing.

    Args:
        justification_reasoning: Free-text reasoning chain.
        evidence_quote: Direct quote from the target sentence used as evidence.
        principle_id: Selected KB principle identifier.
        principle_name: Human-readable name of the selected principle.
        is_ableist: Boolean label (may come from the dataset or be derived).
    """

    justification_reasoning: str
    evidence_quote: str
    principle_id: str
    principle_name: str
    is_ableist: bool

    @field_validator("principle_id")
    @classmethod
    def validate_principle_id(cls, v: str) -> str:
        """Ensure principle_id is a valid PrincipleID.

        Args:
            v: Raw principle_id string.

        Returns:
            Validated principle_id string.

        Raises:
            ValueError: If v is not a valid PrincipleID.
        """
        try:
            PrincipleID(v)
        except ValueError as exc:
            raise ValueError(
                f"principle_id '{v}' is not valid. "
                f"Must be one of {[p.value for p in PrincipleID]}."
            ) from exc
        return v

    @model_validator(mode="after")
    def validate_is_ableist_consistency(self) -> "ParsedOutput":
        """Cross-field validator: warn if is_ableist is inconsistent with principle_id.

        Returns:
            Self, unmodified (consistency is enforced by derive_label at a
            higher layer; here we merely validate the parsed data contract).

        Note:
            We do NOT raise an error here because the dataset may contain
            inconsistencies.  Callers should always use derive_label() and
            not rely on the is_ableist field for final label computation.
        """
        expected = derive_label(self.principle_id)
        if self.is_ableist != expected:
            # We allow the inconsistency but callers will override via derive_label.
            pass
        return self


class JDCSample(BaseModel):
    """A fully-processed training/evaluation sample ready for use.

    Args:
        id: Unique sample identifier.
        input_prompt: The full prompt string.
        parsed_output: Structured output parsed from the dataset.
        derived_label: Authoritative label computed via derive_label(principle_id).
    """

    id: str
    input_prompt: str
    parsed_output: ParsedOutput
    derived_label: bool

    @model_validator(mode="after")
    def validate_derived_label(self) -> "JDCSample":
        """Ensure derived_label matches derive_label(parsed_output.principle_id).

        Returns:
            Self after validation.

        Raises:
            ValueError: If derived_label does not match the expected derivation.
        """
        expected = derive_label(self.parsed_output.principle_id)
        if self.derived_label != expected:
            raise ValueError(
                f"derived_label={self.derived_label} does not match "
                f"derive_label('{self.parsed_output.principle_id}')={expected}. "
                "Always use derive_label() to set this field."
            )
        return self


class EvaluationResult(BaseModel):
    """Quantitative evaluation results for a single dataset split.

    Args:
        split: Name of the evaluated split ("validation" or "test").
        f1: Macro-averaged F1 score for binary classification.
        precision: Macro-averaged precision for binary classification.
        recall: Macro-averaged recall for binary classification.
        accuracy: Binary classification accuracy.
        principle_accuracy: Exact-match accuracy over all 5 principle classes.
        confusion_matrix: 2×2 confusion matrix as nested list.
        classification_report: Full sklearn classification report string.
    """

    split: str
    f1: float
    precision: float
    recall: float
    accuracy: float
    principle_accuracy: float
    confusion_matrix: list[list[int]]
    classification_report: str
