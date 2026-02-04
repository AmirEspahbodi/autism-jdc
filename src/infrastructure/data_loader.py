"""
Infrastructure data loader module - Data loading implementations.
Refactored to include PreformattedDataLoader.
"""

import json
from pathlib import Path
from typing import Optional

from src.domain import DataLoader, DataLoadError, Justification, LabeledExample


class PreformattedDataLoader(DataLoader):
    """Data loader for pre-formatted SFT datasets.

    Reads from JSON files where inputs are already fully formatted.
    Supports both training (dataset.json) and testing (test_dataset.json) splits.
    """

    def __init__(
        self,
        train_path: Path = Path("./dataset/dataset.json"),
        test_path: Path = Path("./dataset/test_dataset.json"),
    ) -> None:
        """
        Args:
            train_path: Path to the training dataset file.
            test_path: Path to the test dataset file.
        """
        self.train_path = train_path
        self.test_path = test_path

    def _load_data_from_json(self, path: Path) -> list[LabeledExample]:
        """Helper to load and validate SFT data from a JSON file."""
        if not path.exists():
            # Gracefully handle missing files by returning empty list
            # This allows the pipeline to continue or fall back if needed
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise DataLoadError(
                    f"Dataset at {path} must be a JSON list of objects."
                )

            examples = []
            for i, item in enumerate(data):
                if "input_prompt" not in item or "model_output" not in item:
                    raise DataLoadError(
                        f"Item {i} in {path} missing 'input_prompt' or 'model_output'"
                    )

                examples.append(
                    LabeledExample(
                        input_prompt=item["input_prompt"],
                        model_output=item["model_output"],
                    )
                )

            return examples

        except json.JSONDecodeError as e:
            raise DataLoadError(f"Invalid JSON in dataset file {path}: {e}")
        except Exception as e:
            raise DataLoadError(f"Failed to load dataset from {path}: {e}")

    def load_training_data(self) -> list[LabeledExample]:
        """Load pre-formatted training data."""
        if not self.train_path.exists():
            raise DataLoadError(
                f"Training dataset file not found at: {self.train_path}"
            )

        return self._load_data_from_json(self.train_path)

    def load_validation_data(self) -> list[LabeledExample]:
        """Load validation data."""
        # Future: could allow a separate val_path in __init__
        return []

    def load_test_data(self) -> list[LabeledExample]:
        """Load pre-formatted test data."""
        data = self._load_data_from_json(self.test_path)
        if not data:
            # Optional: Log warning here if strictly expected,
            # but returning empty list adheres to contract.
            pass
        return data


class FileBasedDataLoader(DataLoader):
    """Legacy Data loader that reads from structured JSON files (train.json)."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def load_training_data(self) -> list[LabeledExample]:
        return self._load_from_file(self.data_dir / "train.json")

    def load_validation_data(self) -> list[LabeledExample]:
        for filename in ["val.json", "validation.json", "dev.json"]:
            path = self.data_dir / filename
            if path.exists():
                return self._load_from_file(path)
        raise DataLoadError(f"No validation file found in {self.data_dir}")

    def load_test_data(self) -> list[LabeledExample]:
        return self._load_from_file(self.data_dir / "test.json")

    def _load_from_file(self, filepath: Path) -> list[LabeledExample]:
        try:
            import json

            with open(filepath, "r") as f:
                data = json.load(f)

            examples = []
            for item in data:
                justification = Justification(
                    principle_id=item["justification"]["principle_id"],
                    justification_text=item["justification"]["justification_text"],
                    evidence_quote=item["justification"]["evidence_quote"],
                )

                example = LabeledExample(
                    sentence=item["sentence"],
                    context_before=item.get("context_before"),
                    context_after=item.get("context_after"),
                    ground_truth_justification=justification,
                    ground_truth_label=item["label"],
                )
                examples.append(example)
            return examples

        except Exception as e:
            raise DataLoadError(f"Failed to load data from {filepath}: {str(e)}") from e
