# Neuro-Symbolic Justification-Driven Classification (JDC) System

A production-grade machine learning system for neurodiversity-aware detection of ableist language using fine-tuned Large Language Models (LLMs) with symbolic knowledge grounding.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Format](#dataset-format)
- [Configuration](#configuration)
- [Usage](#usage)
- [Knowledge Base](#knowledge-base)
- [Metrics & Evaluation](#metrics--evaluation)
- [Advanced Topics](#advanced-topics)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

---

## 🎯 Overview

The JDC system combines **symbolic AI** (knowledge-based principles) with **neural approaches** (fine-tuned LLMs) to detect and classify ableist language patterns in text. It generates structured, interpretable justifications grounded in neurodiversity principles.

### Key Capabilities

- **Binary Classification**: Detects whether text contains ableist language (classes: 0=Safe, 1=Ableist)
- **Principle-Based Justification**: Maps detections to specific neurodiversity principles (P0-P4)
- **Interpretable Output**: Generates structured JSON with evidence quotes and reasoning
- **Production-Ready**: Built with Clean Architecture for maintainability and extensibility
- **Resource-Efficient**: Uses LoRA fine-tuning with 4-bit quantization for consumer GPUs

### Use Cases

- Content moderation for neurodiversity-inclusive platforms
- Assistive writing tools for inclusive language
- Research on bias detection in NLP systems
- Educational tools for neurodiversity awareness

---

## 🏗️ Architecture

The system follows **Clean Architecture** principles with strict layer separation:
```
┌─────────────────────────────────────────────────────┐
│                 Presentation Layer                   │
│              (CLI - src/main.py)                    │
└─────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│               Application Layer                      │
│         (Use Cases - src/application/)              │
│   • FineTuneModelUseCase                            │
│   • EvaluateModelUseCase                            │
└─────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│                 Domain Layer                         │
│          (Business Logic - src/domain/)             │
│   • Interfaces (Ports)                              │
│   • Types (Entities & Value Objects)                │
│   • Domain Exceptions                               │
└─────────────────────────────────────────────────────┘
                        ▲
┌─────────────────────────────────────────────────────┐
│              Infrastructure Layer                    │
│         (Adapters - src/infrastructure/)            │
│   • LoRAAdapter (Training)                          │
│   • HuggingFaceInferenceAdapter                     │
│   • PreformattedDataLoader                          │
│   • RobustJSONParser                                │
│   • StandardMetricsRepository                       │
└─────────────────────────────────────────────────────┘
```

### Design Patterns

- **Dependency Inversion**: Core logic depends on abstractions, not implementations
- **Adapter Pattern**: External frameworks wrapped in domain-specific adapters
- **Strategy Pattern**: Pluggable parsers (Lenient/Strict), data loaders
- **Repository Pattern**: Metrics storage and retrieval abstraction

---

## ✨ Features

### Training Pipeline

- ✅ **LoRA Fine-Tuning**: Parameter-efficient training using PEFT
- ✅ **Quantization Support**: 4-bit/8-bit quantization via bitsandbytes
- ✅ **Validation Splitting**: Deterministic train/val split with configurable ratio
- ✅ **Chat Template Support**: Native Llama 3 and Mistral prompt formatting
- ✅ **Gradient Checkpointing**: Memory-efficient training for large models
- ✅ **Mixed Precision**: FP16 training for faster convergence

### Inference & Evaluation

- ✅ **Batch Generation**: Efficient batch inference with dynamic padding
- ✅ **Robust JSON Parsing**: Handles markdown, malformed JSON, truncated outputs
- ✅ **Comprehensive Metrics**: Precision, Recall, F1, Accuracy, Confusion Matrix
- ✅ **Detailed Reporting**: JSON reports with per-example predictions and errors
- ✅ **Error Recovery**: Lenient parsing with heuristic field extraction

### Data Handling

- ✅ **SFT Dataset Format**: Pre-formatted instruction-response pairs
- ✅ **Validation Safety**: Separate validation split to prevent data leakage
- ✅ **Flexible Loading**: Support for explicit val files or automatic splitting
- ✅ **Polymorphic Examples**: Handles both SFT and legacy structured formats

---

## 📦 Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended: ≥16GB VRAM)
- 32GB+ RAM (for 8B parameter models)

### Setup
```bash
# Clone the repository
git clone https://github.com/your-org/jdc-system.git
cd jdc-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p dataset outputs cache
```

### GPU Verification
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 🚀 Quick Start

### 1. Prepare Your Dataset

Create `dataset/dataset.json` with pre-formatted SFT examples:
```json
[
  {
    "input_prompt": "Classify the following sentence:\n\"People with autism are just broken versions of normal people.\"",
    "model_output": "{\"principle_id\": \"P2\", \"justification_text\": \"The sentence uses dehumanizing language...\", \"evidence_quote\": \"broken versions of normal people\"}"
  }
]
```

See [Dataset Format](#dataset-format) for complete specification.

### 2. Run Full Pipeline
```bash
# Train and evaluate in one command
python src/main.py --mode full
```

### 3. Individual Stages
```bash
# Training only
python src/main.py --mode train --epochs 3 --batch-size 4

# Evaluation only (requires trained model)
python src/main.py --mode eval
```

### 4. Model Selection
```bash
# Use Mistral instead of Llama 3
python src/main.py --model mistral --mode full

# Custom output directory
python src/main.py --output-dir ./my_results
```

---

## 📊 Dataset Format

The system expects datasets in **SFT (Supervised Fine-Tuning)** format with instruction-response pairs.

### Training Dataset (`dataset/dataset.json`)
```json
[
  {
    "input_prompt": "Classify: \"Autistic people lack empathy.\"",
    "model_output": "{\"principle_id\": \"P3\", \"justification_text\": \"This is a harmful stereotype...\", \"evidence_quote\": \"lack empathy\"}"
  },
  {
    "input_prompt": "Classify: \"Neurodivergent individuals bring unique perspectives.\"",
    "model_output": "{\"principle_id\": \"P0\", \"justification_text\": \"This is respectful language...\", \"evidence_quote\": \"unique perspectives\"}"
  }
]
```

### Test Dataset (`dataset/test_dataset.json`)

Same format as training data. Used for final evaluation.

### Validation Dataset (Optional: `dataset/val_dataset.json`)

If provided, used directly. Otherwise, automatically created from training data with 10% split.

### Expected JSON Schema (model_output)
```json
{
  "principle_id": "P0|P1|P2|P3|P4",
  "justification_text": "Explanation of classification",
  "evidence_quote": "Quoted text supporting the decision"
}
```

**Label Mapping**:
- P0 → Safe (label: 0)
- P1, P2, P3, P4 → Ableist (label: 1)

---

## ⚙️ Configuration

### Command-Line Arguments
```bash
python src/main.py \
  --mode full|train|eval \
  --model llama3|mistral \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --output-dir ./outputs \
  --data-dir ./dataset
```

### Programmatic Configuration

Edit `src/config.py` for fine-grained control:
```python
from src.config import SystemConfig, ModelType, LoRAConfig

config = SystemConfig(
    model_type=ModelType.LLAMA3_8B,
    lora_config=LoRAConfig(
        r=16,                    # LoRA rank
        lora_alpha=32,           # LoRA alpha
        target_modules=[         # Modules to adapt
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    ),
    training_hyperparameters=TrainingHyperparameters(
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4,
        max_seq_length=2048
    )
)
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 3 | Training epochs |
| `batch_size` | 4 | Per-device batch size |
| `learning_rate` | 2e-4 | Initial learning rate |
| `lora_r` | 16 | LoRA rank (higher = more parameters) |
| `lora_alpha` | 32 | LoRA scaling factor |
| `max_seq_length` | 2048 | Maximum token length |
| `quantization_type` | 4bit | Model quantization (4bit/8bit/none) |

---

## 💻 Usage

### Training
```python
from pathlib import Path
from src.config import SystemConfig, ModelType
from src.application import FineTuneModelUseCase
from src.infrastructure import LoRAAdapter, PreformattedDataLoader

# Configure system
config = SystemConfig(
    model_type=ModelType.LLAMA3_8B,
    data_dir=Path("./dataset"),
    output_dir=Path("./outputs")
)

# Initialize components
trainer = LoRAAdapter(config)
data_loader = PreformattedDataLoader(
    train_path=config.data_dir / "dataset.json"
)

# Execute training
use_case = FineTuneModelUseCase(trainer, data_loader, config)
use_case.execute(use_validation=True)
```

### Inference
```python
from src.infrastructure import HuggingFaceInferenceAdapter

# Load fine-tuned model
model_path = "./outputs/fine_tuned_model"
engine = HuggingFaceInferenceAdapter(config, model_path)

# Generate prediction
sentence = "Autistic people are inspirations."
output = engine.generate_justification(sentence)
print(output)  # JSON string with principle_id, justification, evidence
```

### Parsing Outputs
```python
from src.infrastructure import LenientJSONParser, ParsingError

parser = LenientJSONParser()

try:
    justification = parser.parse(raw_model_output)
    print(f"Principle: {justification.principle_id}")
    print(f"Label: {justification.to_label()}")  # 0 or 1
    print(f"Reasoning: {justification.justification_text}")
except ParsingError as e:
    print(f"Parsing failed: {e}")
```

---

## 📚 Knowledge Base

The system uses a symbolic knowledge base defining neurodiversity principles:

### Principles

| ID | Name | Classification | Description |
|----|------|----------------|-------------|
| **P0** | Neutral Language | Safe | Language respecting neurological diversity |
| **P1** | Pathologizing Language | Ableist | Frames neurodivergence as deficits/diseases |
| **P2** | Dehumanizing Metaphors | Ableist | Compares people to objects/machinery |
| **P3** | Stereotyping | Ableist | Overgeneralized assumptions about abilities |
| **P4** | Exclusionary Language | Ableist | Excludes neurodivergent people from society |

### Example Classifications
```python
# P0 (Safe)
"Neurodivergent individuals contribute unique perspectives to teams."

# P1 (Ableist - Pathologizing)
"ADHD is a disorder that needs to be cured."

# P2 (Ableist - Dehumanizing)
"They're like robots - no emotions."

# P3 (Ableist - Stereotyping)
"All autistic people are math geniuses."

# P4 (Ableist - Exclusionary)
"People with disabilities can't handle real jobs."
```

---

## 📈 Metrics & Evaluation

### Computed Metrics

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: (TP + TN) / Total
- **Confusion Matrix**: TP, FP, TN, FN counts
- **Parsing Failure Rate**: % of unparseable outputs

### Parsing Failure Handling

**Critical**: Parsing failures are treated as **incorrect predictions** and penalize all metrics:

- Assigned label: `-1` (system error)
- Counted as False Negatives if ground truth = 1 (Ableist)
- Penalizes Accuracy regardless of ground truth

This ensures model reliability is measured, not just classification ability.

### Output Files
```
outputs/
├── fine_tuned_model/          # Saved LoRA adapter + tokenizer
├── checkpoints/               # Training checkpoints
├── evaluation_metrics.json    # Numerical metrics
└── evaluation_report.json     # Detailed per-example results
```

### Sample Metrics Output
```json
{
  "precision": 0.8750,
  "recall": 0.9333,
  "f1_score": 0.9032,
  "accuracy": 0.8800,
  "true_positives": 14,
  "false_positives": 2,
  "true_negatives": 8,
  "false_negatives": 1,
  "total_examples": 25,
  "parsing_failures": 0
}
```

---

## 🔬 Advanced Topics

### Custom Data Loaders

Implement the `DataLoader` interface for custom formats:
```python
from src.domain import DataLoader, LabeledExample

class CustomLoader(DataLoader):
    def load_training_data(self) -> list[LabeledExample]:
        # Your loading logic
        return examples
    
    def load_validation_data(self) -> list[LabeledExample]:
        # Validation split logic
        return val_examples
    
    def load_test_data(self) -> list[LabeledExample]:
        # Test data loading
        return test_examples
```

### Strict vs. Lenient Parsing
```python
from src.infrastructure import StrictJSONParser, LenientJSONParser

# Strict: Fails on any parsing error
strict_parser = StrictJSONParser()

# Lenient: Attempts heuristic recovery
lenient_parser = LenientJSONParser()
```

**Recommendation**: Use `LenientJSONParser` for evaluation to maximize successful parses.

### Multi-GPU Training
```python
# Automatically uses all available GPUs
trainer = LoRAAdapter(config)
# HuggingFace Transformers handles multi-GPU via device_map="auto"
```

### Memory Optimization

For OOM errors:
1. Reduce `batch_size` (e.g., 2 or 1)
2. Increase `gradient_accumulation_steps` (e.g., 8)
3. Reduce `max_seq_length` (e.g., 1024)
4. Use 4-bit quantization (already default)

---

## 📁 Project Structure
```
jdc-system/
├── src/
│   ├── domain/              # Domain layer (pure logic)
│   │   ├── __init__.py
│   │   ├── interfaces.py    # Abstract ports (LLMTrainer, InferenceEngine, etc.)
│   │   └── types.py         # Domain entities (Justification, LabeledExample, etc.)
│   ├── application/         # Application layer (use cases)
│   │   ├── __init__.py
│   │   └── services.py      # FineTuneModelUseCase, EvaluateModelUseCase
│   ├── infrastructure/      # Infrastructure layer (adapters)
│   │   ├── __init__.py
│   │   ├── llm.py          # LoRAAdapter, HuggingFaceInferenceAdapter
│   │   ├── data_loader.py  # PreformattedDataLoader, FileBasedDataLoader
│   │   ├── parsing.py      # RobustJSONParser, LenientJSONParser
│   │   └── metrics.py      # StandardMetricsRepository, ReportGenerator
│   ├── config.py           # Pydantic configuration models
│   └── main.py             # CLI entry point
├── dataset/                # Data directory
│   ├── dataset.json        # Training data (required)
│   ├── test_dataset.json   # Test data (required)
│   └── val_dataset.json    # Validation data (optional)
├── outputs/                # Generated outputs
├── cache/                  # Model cache
├── requirements.txt
└── README.md
```

---

## 🤝 Contributing

### Development Setup
```bash
# Install dev dependencies
pip install ruff mypy pytest

# Run type checking
mypy src/

# Format code
ruff check src/ --fix
```

### Code Style

- Follow Clean Architecture principles
- Use type hints for all functions
- Document with docstrings (Google style)
- Maintain <100 line functions
- Write unit tests for domain logic

### Pull Request Process

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Transformers** by Hugging Face for LLM infrastructure
- **PEFT** library for LoRA implementation
- **TRL** for SFTTrainer
- Neurodiversity advocacy community for principle definitions

---
