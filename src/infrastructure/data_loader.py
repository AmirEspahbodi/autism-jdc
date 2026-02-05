"""
Infrastructure data loader module - Data loading implementations.
Refactored to include PreformattedDataLoader with proper validation splitting.
"""

import json
import random
from pathlib import Path
from typing import Optional

from src.domain import DataLoader, DataLoadError, Justification, LabeledExample


class PreformattedDataLoader(DataLoader):
    """Data loader for pre-formatted SFT datasets.

    Reads from JSON files where inputs are already fully formatted.
    Supports both training (dataset.json) and testing (test_dataset.json) splits.
    Now implements deterministic validation splitting to ensure scientific integrity.
    """

    def __init__(
        self,
        train_path: Path = Path(
            "/content/drive/MyDrive/autism_jdc_dataset/_train_dataset.json"
        ),
        test_path: Path = Path(
            "/content/drive/MyDrive/autism_jdc_dataset/_test_dataset.json"
        ),
        val_path: Path = Path(
            "/content/drive/MyDrive/autism_jdc_dataset/_validation_dataset.json"
        ),
        val_split_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        Args:
            train_path: Path to the training dataset file.
            test_path: Path to the test dataset file.
            val_path: Path to an explicit validation dataset file.
            val_split_ratio: Ratio of training data to use for validation if val_path is missing.
            seed: Random seed for deterministic splitting.
        """
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.val_split_ratio = val_split_ratio
        self.seed = seed

        # Cache to store the split to ensure consistency between load_training and load_validation
        self._train_cache: Optional[list[LabeledExample]] = None
        self._val_cache: Optional[list[LabeledExample]] = None

    def _load_data_from_json(self, path: Path) -> list[LabeledExample]:
        """Helper to load and validate SFT data from a JSON file."""
        if not path.exists():
            # Gracefully handle missing files by returning empty list
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

    def _prepare_train_val_split(self) -> None:
        """
        Prepares training and validation data.
        If explicit validation file exists, uses it.
        Otherwise, performs a deterministic split of the training file.
        """
        if self._train_cache is not None:
            return  # Already loaded

        # Priority A: Check for explicit validation file
        if self.val_path.exists():
            print(f"✓ Loading explicit validation file: {self.val_path}")
            self._train_cache = self._load_data_from_json(self.train_path)
            self._val_cache = self._load_data_from_json(self.val_path)
            return

        # Priority B: Deterministic Split from Training Data
        print(f"ℹ No validation file found at {self.val_path}")
        print(
            f"  Performing deterministic split (Ratio: {self.val_split_ratio}, Seed: {self.seed})..."
        )

        full_data = self._load_data_from_json(self.train_path)
        if not full_data:
            self._train_cache = []
            self._val_cache = []
            return

        # Deterministic shuffle
        rng = random.Random(self.seed)
        rng.shuffle(full_data)

        # Calculate split index
        split_idx = int(len(full_data) * (1 - self.val_split_ratio))

        # Slice data
        self._train_cache = full_data[:split_idx]
        self._val_cache = full_data[split_idx:]

        print(
            f"✓ Split created: {len(self._train_cache)} Training, {len(self._val_cache)} Validation examples"
        )

    def load_training_data(self) -> list[LabeledExample]:
        """Load pre-formatted training data (minus validation split)."""
        self._prepare_train_val_split()
        return self._train_cache if self._train_cache else []

    def load_validation_data(self) -> list[LabeledExample]:
        """Load validation data."""
        self._prepare_train_val_split()
        return self._val_cache if self._val_cache else []

    def load_test_data(self) -> list[LabeledExample]:
        """Load pre-formatted test data."""
        data = self._load_data_from_json(self.test_path)
        if not data:
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
