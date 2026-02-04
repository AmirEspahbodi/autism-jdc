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
    - Braces inside strings (e.g., "text with {braces}")
    - Escaped quotes (\")
    - Nested structures
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
        if not raw_output or not raw_output.strip():
            raise ParsingError("Model returned empty output")

        try:
            # Step 1: Extract JSON from markdown or text
            json_text = self._extract_json(raw_output)

            if not json_text or not json_text.strip():
                raise ParsingError("No JSON content found in output")

            # Step 2: Parse JSON (with lenient fallback)
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                # Try json5 if available (more lenient - handles trailing commas, etc.)
                try:
                    import json5

                    data = json5.loads(json_text)
                except (ImportError, Exception):
                    # Fall back to standard error handling
                    raise

            # Step 3: Validate and normalize schema
            if not self.validate_json_schema(data):
                raise ParsingError("JSON schema validation failed")

            # Step 4: Create Justification object (data is already normalized by validate_json_schema)
            return Justification(
                principle_id=data["principle_id"],  # Already uppercase after validation
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
        """
        Extract JSON using balanced brace counting instead of regex.

        This handles:
        - Braces inside strings (e.g., "text with {braces}")
        - Escaped quotes (\")
        - Nested structures
        - Markdown code blocks

        Args:
            text: Raw text potentially containing JSON.

        Returns:
            Extracted JSON string.

        Raises:
            ParsingError: If no JSON can be found.
        """
        # Step 1: Remove markdown fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```", "", text)

        # Step 2: Find first opening brace
        start_idx = text.find("{")
        if start_idx == -1:
            # No JSON found, return as-is for error handling downstream
            return text.strip()

        # Step 3: Balanced brace counting with string tracking
        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            # Handle escape sequences
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            # Track string boundaries (only " toggles strings in JSON)
            if char == '"':
                in_string = not in_string
                continue

            # Only count braces outside of strings
            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1

                    # Found matching closing brace
                    if brace_count == 0:
                        return text[start_idx : i + 1]

        # Incomplete JSON - model may have hit max_new_tokens limit
        incomplete_json = text[start_idx:]

        return self._complete_incomplete_json(incomplete_json)

    def _complete_incomplete_json(self, partial: str) -> str:
        """
        Attempt to complete truncated JSON output.

        Handles cases where max_new_tokens cut off mid-generation.

        Args:
            partial: Incomplete JSON string.

        Returns:
            Completed JSON string (best effort).
        """
        # Count unmatched braces
        brace_balance = partial.count("{") - partial.count("}")

        # Check if we're in the middle of a string
        # Count quotes, accounting for escaped quotes
        quote_count = len(re.findall(r'(?<!\\)"', partial))
        in_string = (quote_count % 2) == 1

        # Complete the JSON
        completed = partial

        # Close string if needed
        if in_string:
            completed += '"'

        # Close braces
        completed += "}" * brace_balance

        return completed

    def _attempt_recovery(self, raw_output: str) -> Justification:
        """
        Attempt to recover a Justification from malformed output.

        Uses heuristics to extract fields even if JSON is invalid.

        Args:
            raw_output: Raw output that failed JSON parsing.

        Returns:
            Recovered Justification.

        Raises:
            ParsingError: If recovery fails.
        """
        # Extract principle_id (case-insensitive search)
        principle_match = re.search(
            r'"?principle_id"?\s*:\s*"?(P[0-4])"?', raw_output, re.IGNORECASE
        )

        if not principle_match:
            raise ParsingError("Could not extract principle_id from output")

        principle_id = principle_match.group(1).upper()

        # Extract justification_text
        justification_match = re.search(
            r'"?justification_text"?\s*:\s*"([^"]+)"',
            raw_output,
            re.IGNORECASE | re.DOTALL,
        )

        if not justification_match:
            # Fallback: use generic message
            justification_text = "Recovered partial justification from malformed output"
        else:
            justification_text = justification_match.group(1)

        # Extract evidence_quote (optional field)
        evidence_match = re.search(
            r'"?evidence_quote"?\s*:\s*"([^"]+)"', raw_output, re.IGNORECASE | re.DOTALL
        )
        evidence_quote = evidence_match.group(1) if evidence_match else ""

        return Justification(
            principle_id=principle_id,  # Already normalized to uppercase
            justification_text=justification_text,
            evidence_quote=evidence_quote,
        )

    def validate_json_schema(self, data: dict[str, Any]) -> bool:
        """
        Validate that parsed JSON matches the expected schema.

        Args:
            data: Parsed JSON data (modified in-place to normalize principle_id).

        Returns:
            True if valid, False otherwise.
        """
        # Check required fields exist and are strings
        required_fields = ["principle_id", "justification_text"]
        for field in required_fields:
            if field not in data:
                return False
            if not isinstance(data[field], str):
                return False

        principle_id = data["principle_id"].strip().upper()
        valid_principles = {"P0", "P1", "P2", "P3", "P4"}

        if principle_id not in valid_principles:
            return False

        data["principle_id"] = principle_id

        # Validate optional evidence_quote field
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
