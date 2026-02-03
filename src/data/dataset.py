import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from datasets import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class JDCDataset:
    """
    Dataset handler for Justification-Driven Classification.

    Expected data format:
    {
        "input_prompt": "Sentence: ...\nContext: ...\nKnowledge Base: ...",
        "target_output": '{"principle_id": "P1", "justification": "..."}',
        "label": 1  # Optional: ground truth for evaluation
    }
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 2048,
        use_chat_template: bool = True,
    ):
        """
        Initialize dataset handler.

        Args:
            tokenizer: Tokenizer for formatting
            max_seq_length: Maximum sequence length
            use_chat_template: Use model's chat template (recommended)
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_chat_template = use_chat_template

    def load_data(self, data_path: Union[str, Path]) -> List[Dict]:
        """
        Load data from JSON or JSONL file.

        Args:
            data_path: Path to data file

        Returns:
            List of data dictionaries
        """
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if data_path.suffix == ".jsonl":
            return self._load_jsonl(data_path)
        elif data_path.suffix == ".json":
            return self._load_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

    def _load_json(self, path: Path) -> List[Dict]:
        """Load from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        logger.info(f"Loaded {len(data)} examples from {path}")
        return data

    def _load_jsonl(self, path: Path) -> List[Dict]:
        """Load from JSONL file."""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        logger.info(f"Loaded {len(data)} examples from {path}")
        return data

    def format_instruction(
        self, input_prompt: str, target_output: Optional[str] = None
    ) -> str:
        """
        Format data into instruction-following format.

        Args:
            input_prompt: Input containing sentence, context, KB
            target_output: Target JSON output (for training)

        Returns:
            Formatted text for training/inference

        Note:
            For Llama 3, this uses the chat template format:
            <|begin_of_text|><|start_header_id|>user<|end_header_id|>
            {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {output}<|eot_id|>

            For Mistral, it uses:
            [INST] {input} [/INST] {output}
        """
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            # Use model's built-in chat template
            messages = [{"role": "user", "content": input_prompt}]

            if target_output:
                messages.append({"role": "assistant", "content": target_output})

            # Apply chat template
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=(target_output is None)
            )

            return formatted
        else:
            # Fallback to simple format
            if target_output:
                return f"[INST] {input_prompt} [/INST] {target_output}"
            else:
                return f"[INST] {input_prompt} [/INST]"

    def prepare_dataset(self, data: List[Dict], for_training: bool = True) -> Dataset:
        """
        Prepare Hugging Face Dataset for training/evaluation.

        Args:
            data: List of data dictionaries
            for_training: If True, include target outputs

        Returns:
            Hugging Face Dataset
        """
        formatted_data = []

        for example in data:
            input_prompt = example.get("input_prompt", "")

            if for_training:
                target_output = example.get("target_output", "")
                text = self.format_instruction(input_prompt, target_output)
            else:
                text = self.format_instruction(input_prompt, None)

            formatted_example = {
                "text": text,
                "input_prompt": input_prompt,
            }

            # Include ground truth label if available (for evaluation)
            if "label" in example:
                formatted_example["label"] = example["label"]

            # Include target output for evaluation comparison
            if "target_output" in example:
                formatted_example["target_output"] = example["target_output"]

            formatted_data.append(formatted_example)

        dataset = Dataset.from_list(formatted_data)
        logger.info(f"Created dataset with {len(dataset)} examples")

        return dataset

    def create_training_dataset(self, data_path: Union[str, Path]) -> Dataset:
        """
        Complete pipeline: Load and prepare training dataset.

        Args:
            data_path: Path to training data

        Returns:
            Training-ready Dataset
        """
        data = self.load_data(data_path)
        return self.prepare_dataset(data, for_training=True)

    def create_eval_dataset(self, data_path: Union[str, Path]) -> Dataset:
        """
        Complete pipeline: Load and prepare evaluation dataset.

        Args:
            data_path: Path to evaluation data

        Returns:
            Evaluation-ready Dataset
        """
        data = self.load_data(data_path)
        return self.prepare_dataset(data, for_training=False)


def formatting_func(example: Dict) -> str:
    """
    Formatting function for SFTTrainer.

    Args:
        example: Single example from dataset

    Returns:
        Formatted text string

    Note:
        This is used by SFTTrainer's formatting_func parameter.
        It expects the 'text' key to already contain the formatted
        instruction-response pair.
    """
    return example["text"]


def validate_data_format(data: List[Dict]) -> None:
    """
    Validate data format before training.

    Args:
        data: List of data dictionaries

    Raises:
        ValueError: If data format is invalid
    """
    required_keys = ["input_prompt", "target_output"]

    for i, example in enumerate(data[:5]):  # Check first 5 examples
        for key in required_keys:
            if key not in example:
                raise ValueError(
                    f"Example {i} missing required key: {key}. "
                    f"Found keys: {list(example.keys())}"
                )

        # Validate target_output is valid JSON
        try:
            target = json.loads(example["target_output"])
            if "principle_id" not in target or "justification" not in target:
                raise ValueError(
                    f"Example {i}: target_output must contain "
                    "'principle_id' and 'justification'"
                )
        except json.JSONDecodeError:
            raise ValueError(
                f"Example {i}: target_output is not valid JSON: "
                f"{example['target_output']}"
            )

    logger.info(f"Data format validated: {len(data)} examples")
