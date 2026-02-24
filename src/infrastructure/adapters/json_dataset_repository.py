"""JSON-file-based implementation of IDatasetRepository.

Reads the three dataset JSON files from disk, validates each record, and
returns lists of fully-processed JDCSample objects.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig
from pydantic import ValidationError

from src.application.ports.repository_port import IDatasetRepository
from src.domain.entities import JDCSample, ParsedOutput, RawSample
from src.domain.exceptions import DatasetLoadError, MalformedOutputError
from src.domain.value_objects import derive_label


class JsonDatasetRepository(IDatasetRepository):
    """Loads JDC dataset splits from JSON files on disk.

    Args:
        config: OmegaConf DictConfig containing ``data`` section with
                ``dataset_dir``, ``train_file``, ``validation_file``,
                and ``test_file`` keys.
    """

    def __init__(self, config: DictConfig) -> None:
        self._dataset_dir = Path(str(config.data.dataset_dir))
        self._train_file = self._dataset_dir / str(config.data.train_file)
        self._validation_file = self._dataset_dir / str(config.data.validation_file)
        self._test_file = self._dataset_dir / str(config.data.test_file)

    def load_train(self) -> list[JDCSample]:
        """Load the training split.

        Returns:
            List of JDCSample objects parsed from ``train_dataset.json``.

        Raises:
            DatasetLoadError: If the file cannot be read.
        """
        return self._load_file(self._train_file, split="train")

    def load_validation(self) -> list[JDCSample]:
        """Load the validation split.

        Returns:
            List of JDCSample objects parsed from ``validation_dataset.json``.

        Raises:
            DatasetLoadError: If the file cannot be read.
        """
        return self._load_file(self._validation_file, split="validation")

    def load_test(self) -> list[JDCSample]:
        """Load the test split.

        Returns:
            List of JDCSample objects parsed from ``test_dataset.json``.

        Raises:
            DatasetLoadError: If the file cannot be read.
        """
        return self._load_file(self._test_file, split="test")

    def _load_file(self, path: Path, split: str) -> list[JDCSample]:
        """Read and parse a single JSON dataset file.

        Args:
            path: Absolute path to the JSON file.
            split: Split name (used for log messages only).

        Returns:
            List of validated JDCSample objects.

        Raises:
            DatasetLoadError: If the file does not exist or cannot be parsed.
        """
        if not path.exists():
            raise DatasetLoadError(
                f"Dataset file not found: {path}. "
                "Ensure the ./dataset/ directory is populated before running."
            )

        logger.info(f"Loading dataset split='{split}' from {path} …")

        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DatasetLoadError(
                f"Cannot read dataset file {path}: {exc}"
            ) from exc

        try:
            records: list[Any] = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise DatasetLoadError(
                f"Dataset file {path} is not valid JSON: {exc}"
            ) from exc

        if not isinstance(records, list):
            raise DatasetLoadError(
                f"Dataset file {path} must contain a JSON array at the top level, "
                f"got {type(records).__name__}."
            )

        samples: list[JDCSample] = []
        skipped = 0

        for idx, record in enumerate(records):
            sample_id = record.get("id", f"<unknown idx={idx}>") if isinstance(record, dict) else f"<idx={idx}>"
            result = self._parse_record(record, sample_id, split)
            if result is None:
                skipped += 1
            else:
                samples.append(result)

        logger.info(
            f"Dataset split='{split}': loaded={len(samples)}, skipped={skipped} "
            f"(total records={len(records)})."
        )
        return samples

    def _parse_record(
        self,
        record: Any,
        sample_id: str,
        split: str,
    ) -> JDCSample | None:
        """Parse and validate a single raw dict into a JDCSample.

        Args:
            record: The raw dict from the JSON array.
            sample_id: Identifier used in log messages.
            split: Split name used in log messages.

        Returns:
            A JDCSample on success, or None if the record should be skipped.
        """
        # --- Step 1: validate outer schema with RawSample ---
        try:
            raw_sample = RawSample.model_validate(record)
        except ValidationError as exc:
            logger.warning(
                f"[{split}] Skipping record id='{sample_id}': "
                f"RawSample validation failed: {exc}"
            )
            return None

        # --- Step 2: parse model_output string as JSON ---
        try:
            output_dict: Any = json.loads(raw_sample.model_output)
        except json.JSONDecodeError as exc:
            logger.warning(
                f"[{split}] Skipping sample id='{raw_sample.id}': "
                f"model_output is not valid JSON: {exc}. "
                f"Raw value (first 200 chars): {raw_sample.model_output[:200]!r}"
            )
            return None

        # --- Step 3: validate parsed dict with ParsedOutput ---
        try:
            parsed_output = ParsedOutput.model_validate(output_dict)
        except ValidationError as exc:
            logger.warning(
                f"[{split}] Skipping sample id='{raw_sample.id}': "
                f"ParsedOutput validation failed: {exc}"
            )
            return None

        # --- Step 4: compute derived_label deterministically ---
        try:
            derived = derive_label(parsed_output.principle_id)
        except ValueError as exc:
            logger.warning(
                f"[{split}] Skipping sample id='{raw_sample.id}': "
                f"derive_label failed: {exc}"
            )
            return None

        # --- Step 5: assemble JDCSample ---
        try:
            return JDCSample(
                id=raw_sample.id,
                input_prompt=raw_sample.input_prompt,
                parsed_output=parsed_output,
                derived_label=derived,
            )
        except ValidationError as exc:
            logger.warning(
                f"[{split}] Skipping sample id='{raw_sample.id}': "
                f"JDCSample construction failed: {exc}"
            )
            return None
