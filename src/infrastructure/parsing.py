"""
Infrastructure parsing module - Robust JSON parsing from LLM outputs.

This module contains adapters for parsing model outputs that may contain
markdown formatting, preambles, or other artifacts.
"""

import json
import re
from typing import Any

from src.domain import Justification, JustificationParser, ParsingError


class RobustJSONParser(JustificationParser):
    """Robust parser for extracting Justification objects from LLM outputs.

    This parser handles common edge cases:
    - Markdown code blocks (```json ... ```)
    - Preamble text before the JSON
    - Trailing text after the JSON
    - Malformed JSON that can be partially recovered

    Attributes:
        strict: If True, raises on any parsing error. If False, attempts recovery.
    """

    def __init__(self, strict: bool = False) -> None:
        """Initialize the parser.

        Args:
            strict: Whether to use strict parsing (no error recovery).
        """
        self.strict = strict

    def parse(self, raw_output: str) -> Justification:
        """Parse raw model output into a Justification.

        Args:
            raw_output: Raw text output from the model.

        Returns:
            Parsed Justification object.

        Raises:
            ParsingError: If the output cannot be parsed.
        """
        try:
            # Step 1: Extract JSON from markdown or text
            json_text = self._extract_json(raw_output)

            # Step 2: Parse JSON
            data = json.loads(json_text)

            # Step 3: Validate schema
            if not self.validate_json_schema(data):
                raise ParsingError("JSON schema validation failed")

            # Step 4: Create Justification object
            return Justification(
                principle_id=data["principle_id"],
                justification_text=data["justification_text"],
                evidence_quote=data.get("evidence_quote", ""),
            )

        except json.JSONDecodeError as e:
            if self.strict:
                raise ParsingError(f"Invalid JSON: {str(e)}") from e

            # Attempt recovery
            try:
                return self._attempt_recovery(raw_output)
            except Exception as recovery_error:
                raise ParsingError(
                    f"JSON parsing failed and recovery unsuccessful: {str(recovery_error)}"
                ) from e

        except Exception as e:
            raise ParsingError(f"Parsing failed: {str(e)}") from e

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown or preamble.

        Args:
            text: Raw text potentially containing JSON.

        Returns:
            Extracted JSON string.

        Raises:
            ParsingError: If no JSON can be found.
        """
        # Remove common markdown code block markers
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = re.sub(r"^```\s*", "", text)

        # Try to find JSON object using braces
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            return match.group(0)

        # If no braces found, assume the entire text is JSON
        return text.strip()

    def _attempt_recovery(self, raw_output: str) -> Justification:
        """Attempt to recover a Justification from malformed output.

        This method uses heuristics to extract principle_id and justification
        even if the JSON is malformed.

        Args:
            raw_output: Raw output that failed JSON parsing.

        Returns:
            Recovered Justification.

        Raises:
            ParsingError: If recovery fails.
        """
        # Try to extract principle_id
        principle_match = re.search(
            r'"?principle_id"?\s*:\s*"?(P[0-4])"?', raw_output, re.IGNORECASE
        )
        if not principle_match:
            raise ParsingError("Could not extract principle_id")

        principle_id = principle_match.group(1)

        # Try to extract justification_text
        justification_match = re.search(
            r'"?justification_text"?\s*:\s*"([^"]+)"',
            raw_output,
            re.IGNORECASE | re.DOTALL,
        )
        if not justification_match:
            # Fall back to extracting any text after principle_id
            justification_text = "Recovered partial justification"
        else:
            justification_text = justification_match.group(1)

        # Try to extract evidence_quote
        evidence_match = re.search(
            r'"?evidence_quote"?\s*:\s*"([^"]+)"', raw_output, re.IGNORECASE | re.DOTALL
        )
        evidence_quote = evidence_match.group(1) if evidence_match else ""

        return Justification(
            principle_id=principle_id,
            justification_text=justification_text,
            evidence_quote=evidence_quote,
        )

    def validate_json_schema(self, data: dict[str, Any]) -> bool:
        """Validate that parsed JSON matches the expected schema.

        Args:
            data: Parsed JSON data.

        Returns:
            True if valid, False otherwise.
        """
        # Check required fields
        required_fields = ["principle_id", "justification_text"]
        for field in required_fields:
            if field not in data:
                return False
            if not isinstance(data[field], str):
                return False

        # Check principle_id is valid
        valid_principles = {"P0", "P1", "P2", "P3", "P4"}
        if data["principle_id"] not in valid_principles:
            return False

        # evidence_quote is optional but should be string if present
        if "evidence_quote" in data and not isinstance(data["evidence_quote"], str):
            return False

        return True


class LenientJSONParser(RobustJSONParser):
    """Lenient parser that always attempts recovery.

    This is a convenience subclass with strict=False by default.
    """

    def __init__(self) -> None:
        """Initialize lenient parser."""
        super().__init__(strict=False)


class StrictJSONParser(RobustJSONParser):
    """Strict parser that never attempts recovery.

    This is a convenience subclass with strict=True by default.
    """

    def __init__(self) -> None:
        """Initialize strict parser."""
        super().__init__(strict=True)
