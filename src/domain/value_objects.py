"""Domain value objects: PrincipleID enum and label derivation logic.

This module is the single source of truth for the mapping between
KB principle identifiers and binary ableism labels.  No other module
should implement this mapping.
"""

from __future__ import annotations

from enum import Enum


class PrincipleID(str, Enum):
    """Enumeration of all valid Knowledge Base principle identifiers.

    Each value corresponds to a principle in the symbolic KB that is
    embedded in every input_prompt.
    """

    P0 = "P0"  # Not Ableist
    P1 = "P1"  # Medical Model Framing
    P2 = "P2"  # Eugenicist Hierarchy
    P3 = "P3"  # Promotion of Harmful Tropes
    P4 = "P4"  # Centering Neurotypical Perspectives
    P5 = "P5"  # Not Ableist
    P6 = "P6"  # Medical Model Framing
    P7 = "P7"  # Eugenicist Hierarchy
    P8 = "P8"  # Promotion of Harmful Tropes
    P9 = "P9"  # Centering Neurotypical Perspectives


#: The set of principle IDs whose selection implies an ableist label.
ABLEIST_PRINCIPLES: frozenset[PrincipleID] = frozenset(
    {PrincipleID.P1, PrincipleID.P2, PrincipleID.P3, PrincipleID.P4}
)


def derive_label(principle_id: str) -> bool:
    """Deterministically derive the binary ableism label from a principle ID.

    The rule is hardcoded by design: the label is NOT a model prediction
    but a logical consequence of which KB principle was selected.

    Args:
        principle_id: A string in {"P0", "P1", "P2", "P3", "P4"}.

    Returns:
        True  if the principle_id belongs to ABLEIST_PRINCIPLES.
        False if the principle_id is P0 (Not Ableist).

    Raises:
        ValueError: If the principle_id is not a valid PrincipleID string.

    Example:
        >>> derive_label("P1")
        True
        >>> derive_label("P0")
        False
    """
    try:
        pid = PrincipleID(principle_id)
    except ValueError as exc:
        raise ValueError(
            f"Invalid principle_id '{principle_id}'. "
            f"Must be one of {[p.value for p in PrincipleID]}."
        ) from exc

    return pid in ABLEIST_PRINCIPLES
