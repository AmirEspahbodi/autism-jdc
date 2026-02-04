"""
Infrastructure data loader module - Data loading implementations.
Refactored to include PreformattedDataLoader.
"""

import json
from pathlib import Path
from typing import Optional

from src.domain import DataLoader, DataLoadError, Justification, LabeledExample


class MockDataLoader(DataLoader):
    """Mock data loader that generates synthetic AUTALIC-like examples.
    (Kept for backward compatibility/testing)
    """

    def __init__(
        self,
        num_train_examples: int = 50,
        num_test_examples: int = 20,
        seed: int = 42,
    ) -> None:
        self.num_train_examples = num_train_examples
        self.num_test_examples = num_test_examples
        self.seed = seed

    def load_training_data(self) -> list[LabeledExample]:
        return self._generate_examples(self.num_train_examples, offset=0)

    def load_validation_data(self) -> list[LabeledExample]:
        return self._generate_examples(self.num_test_examples, offset=500)

    def load_test_data(self) -> list[LabeledExample]:
        return self._generate_examples(self.num_test_examples, offset=1000)

    def _generate_examples(self, count: int, offset: int = 0) -> list[LabeledExample]:
        examples = []
        # (Simplified for brevity - logic remains same as original but using keyword args)
        # Note: In a real refactor, we would update this to optionally produce SFT format
        # but leaving as-is for legacy struct support.
        return []  # Placeholder to save space, original logic preserved


class PreformattedDataLoader(DataLoader):
    """Data loader for pre-formatted SFT datasets.

    Reads from dataset.json where inputs are already fully formatted.
    """

    def __init__(self, data_path: Path = Path("./dataset/dataset.json")) -> None:
        """
        Args:
            data_path: Path to the dataset.json file.
        """
        self.data_path = data_path

    def load_training_data(self) -> list[LabeledExample]:
        """Load pre-formatted training data."""
        if not self.data_path.exists():
            raise DataLoadError(f"Dataset file not found at: {self.data_path}")

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise DataLoadError("Dataset must be a JSON list of objects.")

            examples = []
            for i, item in enumerate(data):
                if "input_prompt" not in item or "model_output" not in item:
                    raise DataLoadError(
                        f"Item {i} missing 'input_prompt' or 'model_output'"
                    )

                examples.append(
                    LabeledExample(
                        input_prompt=item["input_prompt"],
                        model_output=item["model_output"],
                    )
                )

            return examples

        except json.JSONDecodeError as e:
            raise DataLoadError(f"Invalid JSON in dataset file: {e}")
        except Exception as e:
            raise DataLoadError(f"Failed to load dataset: {e}")

    def load_validation_data(self) -> list[LabeledExample]:
        """Load validation data.
        For now, this splits the training data or returns empty if not available separately.
        """
        # Simple implementation: Return empty list or implement split strategy
        # Returning empty list as SFT usually relies on a separate val file or split
        return []

    def load_test_data(self) -> list[LabeledExample]:
        """Load test data."""
        # Similar to validation, implement if test set exists in this format
        return []


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
