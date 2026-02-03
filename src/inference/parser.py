"""
Resilient JSON parser for handling LLM-generated structured outputs.

LLMs may produce:
- Valid JSON
- JSON wrapped in markdown code blocks (```json ... ```)
- JSON with trailing commas or missing braces
- Plain text explanations before/after JSON
- Completely malformed outputs

This module provides robust parsing with fallback strategies.
"""

import json
import re
import logging
from typing import Dict, Optional, Any
from json_repair import repair_json

logger = logging.getLogger(__name__)


class ResilientJSONParser:
    """
    Multi-strategy JSON parser with graceful degradation.
    
    Parsing strategies (in order):
    1. Direct JSON parsing
    2. Extract JSON from markdown code blocks
    3. Regex extraction of JSON-like structures
    4. JSON repair library
    5. Fallback to error principle
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        fallback_principle: str = "P_ERR",
        expected_keys: Optional[list[str]] = None
    ):
        """
        Initialize parser.
        
        Args:
            max_retries: Number of parsing strategies to attempt
            fallback_principle: Principle ID to use when all parsing fails
            expected_keys: Keys expected in valid JSON (for validation)
        """
        self.max_retries = max_retries
        self.fallback_principle = fallback_principle
        self.expected_keys = expected_keys or ["principle_id", "justification"]
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM output with multiple fallback strategies.
        
        Args:
            text: Raw text output from LLM
            
        Returns:
            Parsed dictionary with at minimum {principle_id, justification}
            
        Note:
            If all parsing fails, returns fallback structure with P_ERR.
            This prevents evaluation crashes while logging failures.
        """
        if not text or not text.strip():
            logger.warning("Empty output received")
            return self._create_fallback("Empty output")
        
        # Strategy 1: Direct JSON parsing
        result = self._try_direct_parse(text)
        if result:
            return result
        
        # Strategy 2: Extract from markdown code blocks
        result = self._try_markdown_extraction(text)
        if result:
            return result
        
        # Strategy 3: Regex extraction
        result = self._try_regex_extraction(text)
        if result:
            return result
        
        # Strategy 4: JSON repair library
        result = self._try_repair(text)
        if result:
            return result
        
        # All strategies failed - use fallback
        logger.error(f"All parsing strategies failed for text: {text[:200]}...")
        return self._create_fallback(text)
    
    def _try_direct_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempt direct JSON parsing."""
        try:
            data = json.loads(text.strip())
            if self._validate_structure(data):
                logger.debug("Direct JSON parse successful")
                return data
        except json.JSONDecodeError:
            pass
        return None
    
    def _try_markdown_extraction(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from markdown code blocks.
        
        Handles formats like:
        ```json
        {"principle_id": "P1", "justification": "..."}
        ```
        
        or
        
        ```
        {"principle_id": "P1", "justification": "..."}
        ```
        """
        # Pattern for ```json or ``` code blocks
        patterns = [
            r'```json\s*(.*?)\s*```',  # ```json ... ```
            r'```\s*(.*?)\s*```',       # ``` ... ```
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                json_str = match.group(1).strip()
                try:
                    data = json.loads(json_str)
                    if self._validate_structure(data):
                        logger.debug("Markdown extraction successful")
                        return data
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _try_regex_extraction(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON-like structures using regex.
        
        Looks for anything resembling {"principle_id": ..., "justification": ...}
        even if embedded in surrounding text.
        """
        # Pattern to match object with our expected keys
        # More permissive - allows for various quote styles and whitespace
        pattern = r'\{[^{}]*"principle_id"\s*:\s*"([^"]+)"[^{}]*"justification"\s*:\s*"([^"]*(?:"[^"]*)*)"[^{}]*\}'
        
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            # Reconstruct JSON from captured groups
            try:
                # Extract the matched JSON-like string
                json_str = match.group(0)
                data = json.loads(json_str)
                if self._validate_structure(data):
                    logger.debug("Regex extraction successful")
                    return data
            except (json.JSONDecodeError, IndexError):
                pass
        
        # Try a more aggressive extraction
        # Find all text between first { and last }
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                if self._validate_structure(data):
                    logger.debug("Aggressive regex extraction successful")
                    return data
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _try_repair(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair malformed JSON.
        
        Uses json-repair library which can handle:
        - Missing closing braces
        - Trailing commas
        - Single quotes instead of double quotes
        - Comments in JSON
        """
        try:
            # First try to extract JSON-like substring
            brace_match = re.search(r'\{.*\}', text, re.DOTALL)
            text_to_repair = brace_match.group(0) if brace_match else text
            
            # Attempt repair
            repaired = repair_json(text_to_repair)
            data = json.loads(repaired)
            
            if self._validate_structure(data):
                logger.debug("JSON repair successful")
                return data
        except Exception as e:
            logger.debug(f"JSON repair failed: {e}")
        
        return None
    
    def _validate_structure(self, data: Any) -> bool:
        """
        Validate that parsed data has expected structure.
        
        Args:
            data: Parsed data
            
        Returns:
            True if data is dict with expected keys, False otherwise
        """
        if not isinstance(data, dict):
            return False
        
        # Check for expected keys
        for key in self.expected_keys:
            if key not in data:
                logger.warning(f"Missing expected key: {key}")
                return False
        
        # Validate principle_id format (should be like P0, P1, etc.)
        principle_id = data.get("principle_id", "")
        if not isinstance(principle_id, str):
            logger.warning(f"Invalid principle_id type: {type(principle_id)}")
            return False
        
        # Ensure justification is a string
        justification = data.get("justification", "")
        if not isinstance(justification, str):
            logger.warning(f"Invalid justification type: {type(justification)}")
            return False
        
        return True
    
    def _create_fallback(self, original_text: str) -> Dict[str, Any]:
        """
        Create fallback structure when all parsing fails.
        
        Args:
            original_text: Original unparseable text
            
        Returns:
            Dictionary with error principle and truncated original text
        """
        return {
            "principle_id": self.fallback_principle,
            "justification": f"PARSE_ERROR: {original_text[:200]}",
            "parse_failed": True
        }
    
    def batch_parse(self, texts: list[str]) -> list[Dict[str, Any]]:
        """
        Parse multiple outputs.
        
        Args:
            texts: List of raw text outputs
            
        Returns:
            List of parsed dictionaries
        """
        results = []
        failed_count = 0
        
        for i, text in enumerate(texts):
            result = self.parse(text)
            results.append(result)
            
            if result.get("parse_failed", False):
                failed_count += 1
        
        if failed_count > 0:
            logger.warning(
                f"Failed to parse {failed_count}/{len(texts)} outputs "
                f"({100*failed_count/len(texts):.1f}%)"
            )
        
        return results


class PrincipleMapper:
    """
    Maps principle IDs to binary ableism labels.
    
    Example mapping:
        P0 -> 0 (Not Ableist)
        P1, P2, P3, P4 -> 1 (Ableist)
        P_ERR -> -1 (Error)
    """
    
    def __init__(
        self,
        ableist_principles: list[str] = None,
        non_ableist_principles: list[str] = None,
        error_principle: str = "P_ERR"
    ):
        """
        Initialize mapper.
        
        Args:
            ableist_principles: List of principle IDs mapped to 1
            non_ableist_principles: List of principle IDs mapped to 0
            error_principle: Principle ID for parsing errors (mapped to -1)
        """
        self.ableist_principles = set(ableist_principles or ["P1", "P2", "P3", "P4"])
        self.non_ableist_principles = set(non_ableist_principles or ["P0"])
        self.error_principle = error_principle
    
    def map_to_label(self, principle_id: str) -> int:
        """
        Map principle ID to binary label.
        
        Args:
            principle_id: Principle identifier (e.g., "P1")
            
        Returns:
            1 (ableist), 0 (non-ableist), or -1 (error)
        """
        principle_id = principle_id.strip().upper()
        
        if principle_id in self.ableist_principles:
            return 1
        elif principle_id in self.non_ableist_principles:
            return 0
        elif principle_id == self.error_principle:
            return -1
        else:
            logger.warning(f"Unknown principle_id: {principle_id}, treating as error")
            return -1
    
    def batch_map(self, principle_ids: list[str]) -> list[int]:
        """Map multiple principle IDs to labels."""
        return [self.map_to_label(pid) for pid in principle_ids]
