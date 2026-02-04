"""
Infrastructure data loader module - Data loading implementations.

This module contains adapters for loading training and test data.
Includes a mock data loader for demonstration purposes.
"""

from pathlib import Path
from typing import Optional

from src.domain import DataLoader, DataLoadError, Justification, LabeledExample


class MockDataLoader(DataLoader):
    """Mock data loader that generates synthetic AUTALIC-like examples.

    This loader creates realistic training and test examples to demonstrate
    the JDC system's functionality when the actual dataset is not available.

    The examples cover all five principle categories (P0-P4) and include
    contextual information.

    Attributes:
        num_train_examples: Number of training examples to generate.
        num_test_examples: Number of test examples to generate.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_train_examples: int = 50,
        num_test_examples: int = 20,
        seed: int = 42,
    ) -> None:
        """Initialize the mock data loader.

        Args:
            num_train_examples: Number of training examples.
            num_test_examples: Number of test examples.
            seed: Random seed for reproducibility.
        """
        self.num_train_examples = num_train_examples
        self.num_test_examples = num_test_examples
        self.seed = seed

    def load_training_data(self) -> list[LabeledExample]:
        """Generate mock training examples.

        Returns:
            List of synthetic labeled training examples.
        """
        # Train uses offset 0
        return self._generate_examples(self.num_train_examples, offset=0)

    def load_validation_data(self) -> list[LabeledExample]:
        """Generate mock validation examples.

        # FIX: Data Leakage Prevention
        Uses a distinct offset (500) to ensure validation data is mathematically
        distinct from Training (0+) and Test (1000+) data.

        Returns:
            List of synthetic labeled validation examples.
        """
        # Validation uses offset 500 (distinct from train and test)
        # Use same size as test set for simplicity
        return self._generate_examples(self.num_test_examples, offset=500)

    def load_test_data(self) -> list[LabeledExample]:
        """Generate mock test examples.

        Returns:
            List of synthetic labeled test examples.
        """
        # Test uses offset 1000
        return self._generate_examples(self.num_test_examples, offset=1000)

    def _generate_examples(self, count: int, offset: int = 0) -> list[LabeledExample]:
        """Generate a list of labeled examples.

        Args:
            count: Number of examples to generate.
            offset: Offset for example indexing (to avoid duplicates).

        Returns:
            List of labeled examples.
        """
        examples = []

        # Template examples for each principle
        templates = self._get_templates()

        for i in range(count):
            # Cycle through principles
            principle_idx = (i + offset) % 5
            principle_id = f"P{principle_idx}"
            label = 0 if principle_id == "P0" else 1

            # Get template for this principle
            template = templates[principle_id][i % len(templates[principle_id])]

            example = LabeledExample(
                sentence=template["sentence"],
                context_before=template.get("context_before"),
                context_after=template.get("context_after"),
                ground_truth_justification=Justification(
                    principle_id=principle_id,
                    justification_text=template["justification"],
                    evidence_quote=template["evidence"],
                ),
                ground_truth_label=label,
            )

            examples.append(example)

        return examples

    def _get_templates(self) -> dict[str, list[dict]]:
        """Get template examples for each principle.

        Returns:
            Dictionary mapping principle IDs to lists of templates.
        """
        return {
            "P0": [
                {
                    "sentence": "The employee has ADHD and brings unique problem-solving approaches to the team.",
                    "context_before": "Our team is very diverse.",
                    "context_after": "We value different perspectives.",
                    "justification": "This sentence uses neutral, respectful language that recognizes ADHD as a neurological difference rather than a deficit. It highlights strengths and contributions.",
                    "evidence": "brings unique problem-solving approaches",
                },
                {
                    "sentence": "Autistic individuals often have exceptional attention to detail.",
                    "context_before": "Research shows that neurodiversity benefits workplaces.",
                    "context_after": None,
                    "justification": "This statement recognizes autism as a neurological variation and highlights positive traits without stereotyping or pathologizing.",
                    "evidence": "exceptional attention to detail",
                },
                {
                    "sentence": "People with dyslexia may process information differently and often excel in visual thinking.",
                    "context_before": None,
                    "context_after": "Many successful entrepreneurs are dyslexic.",
                    "justification": "The language acknowledges differences while emphasizing strengths, treating dyslexia as a variation rather than a disorder.",
                    "evidence": "process information differently and often excel",
                },
            ],
            "P1": [
                {
                    "sentence": "He suffers from autism and needs to be cured.",
                    "context_before": "The patient was diagnosed last year.",
                    "context_after": "His parents are seeking treatment options.",
                    "justification": "This sentence pathologizes autism by using 'suffers from' and frames it as a disease requiring a cure, rather than a neurological difference.",
                    "evidence": "suffers from autism and needs to be cured",
                },
                {
                    "sentence": "ADHD is a disorder that impairs normal brain function.",
                    "context_before": None,
                    "context_after": "Medication is often prescribed to correct this.",
                    "justification": "This language pathologizes ADHD by calling it a 'disorder' and framing neurodivergent brain function as 'impaired' or abnormal.",
                    "evidence": "impairs normal brain function",
                },
                {
                    "sentence": "She has a severe case of dyslexia that prevents her from functioning properly.",
                    "context_before": "The student struggles in class.",
                    "context_after": None,
                    "justification": "The phrase 'severe case' medicalizes dyslexia, and 'prevents functioning properly' frames it as a fundamental deficit.",
                    "evidence": "prevents her from functioning properly",
                },
            ],
            "P2": [
                {
                    "sentence": "People with autism are like robots lacking emotions.",
                    "context_before": "Some misconceptions about autism persist.",
                    "context_after": "This is completely false.",
                    "justification": "This uses a dehumanizing metaphor comparing autistic people to robots, denying their emotional capacity and humanity.",
                    "evidence": "like robots lacking emotions",
                },
                {
                    "sentence": "His ADHD brain is broken and needs fixing like a malfunctioning machine.",
                    "context_before": None,
                    "context_after": "He's trying various medications.",
                    "justification": "This compares a neurodivergent brain to 'broken' machinery, using a dehumanizing mechanical metaphor that denies agency.",
                    "evidence": "broken and needs fixing like a malfunctioning machine",
                },
                {
                    "sentence": "Dyslexic students are just wired wrong, like defective computers.",
                    "context_before": "The teacher was discussing learning differences.",
                    "context_after": None,
                    "justification": "This metaphor compares neurodivergent individuals to 'defective computers', dehumanizing them and framing their neurology as faulty.",
                    "evidence": "wired wrong, like defective computers",
                },
            ],
            "P3": [
                {
                    "sentence": "All autistic people are math geniuses who can't socialize.",
                    "context_before": "There are many stereotypes about autism.",
                    "context_after": "None",
                    "justification": "This makes overgeneralized assumptions about all autistic individuals, stereotyping both their abilities and social skills.",
                    "evidence": "All autistic people are math geniuses who can't socialize",
                },
                {
                    "sentence": "People with ADHD are always hyperactive and can never focus on anything.",
                    "context_before": None,
                    "context_after": "This is a common misconception.",
                    "justification": "This stereotypes all people with ADHD as having the same behaviors, ignoring individual variation and different ADHD presentations.",
                    "evidence": "always hyperactive and can never focus",
                },
                {
                    "sentence": "Every dyslexic person reads backwards and sees letters upside down.",
                    "context_before": "Many myths about dyslexia exist.",
                    "context_after": None,
                    "justification": "This overgeneralizes characteristics and perpetuates stereotypes about dyslexia that don't apply to all individuals.",
                    "evidence": "Every dyslexic person reads backwards",
                },
            ],
            "P4": [
                {
                    "sentence": "Autistic people shouldn't be allowed in regular classrooms because they disrupt learning.",
                    "context_before": "The school board discussed inclusion policies.",
                    "context_after": None,
                    "justification": "This explicitly excludes neurodivergent individuals from participation in mainstream education based on discriminatory assumptions.",
                    "evidence": "shouldn't be allowed in regular classrooms",
                },
                {
                    "sentence": "We can't hire someone with ADHD for this job; they're too unreliable.",
                    "context_before": "The hiring manager was reviewing applications.",
                    "context_after": None,
                    "justification": "This language excludes people with ADHD from employment opportunities based on stereotypical assumptions about reliability.",
                    "evidence": "can't hire someone with ADHD",
                },
                {
                    "sentence": "People with learning disabilities don't belong in higher education.",
                    "context_before": None,
                    "context_after": "They should pursue other paths.",
                    "justification": "This explicitly excludes neurodivergent individuals from educational opportunities, denying their right to access higher education.",
                    "evidence": "don't belong in higher education",
                },
            ],
        }


class FileBasedDataLoader(DataLoader):
    """Data loader that reads from JSON files.

    This can be used when actual AUTALIC dataset files are available.

    Attributes:
        data_dir: Directory containing the data files.
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize the file-based data loader.

        Args:
            data_dir: Directory containing train.json, val.json, and test.json.
        """
        self.data_dir = data_dir

    def load_training_data(self) -> list[LabeledExample]:
        """Load training data from file.

        Returns:
            List of labeled training examples.

        Raises:
            DataLoadError: If loading fails.
        """
        return self._load_from_file(self.data_dir / "train.json")

    def load_validation_data(self) -> list[LabeledExample]:
        """Load validation data from file.

        # FIX: Data Leakage Prevention
        Attempts to load 'val.json' or 'validation.json'.

        Returns:
            List of labeled validation examples.

        Raises:
            DataLoadError: If loading fails.
        """
        # Try different common names
        for filename in ["val.json", "validation.json", "dev.json"]:
            path = self.data_dir / filename
            if path.exists():
                return self._load_from_file(path)

        raise DataLoadError(
            f"No validation file found in {self.data_dir}. "
            "Expected val.json, validation.json, or dev.json. "
            "Validation data MUST be distinct from test data to prevent leakage."
        )

    def load_test_data(self) -> list[LabeledExample]:
        """Load test data from file.

        Returns:
            List of labeled test examples.

        Raises:
            DataLoadError: If loading fails.
        """
        return self._load_from_file(self.data_dir / "test.json")

    def _load_from_file(self, filepath: Path) -> list[LabeledExample]:
        """Load examples from a JSON file.

        Args:
            filepath: Path to the JSON file.

        Returns:
            List of labeled examples.

        Raises:
            DataLoadError: If loading or parsing fails.
        """
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
