# Neuro-Symbolic Justification-Driven Classification (JDC) System

A production-grade Python implementation of a neuro-symbolic AI system that fine-tunes Large Language Models (LLMs) using LoRA to perform justification-driven classification of ableist language.

## 🎯 Core Concept

Unlike traditional classification systems that directly output labels, this system:

1. **Generates Justifications**: Fine-tunes an LLM to produce structured JSON reasoning based on a symbolic Knowledge Base
2. **Derives Labels**: Maps the generated justification to binary labels through deterministic rules
3. **Ensures Interpretability**: Every classification decision is backed by explicit reasoning

## 🏗️ Architecture

Built following **Clean Architecture** (Onion/Hexagonal) principles:

```
┌─────────────────────────────────────────┐
│         Domain Layer (Pure)                                 │
│  - Entities (Principle, Justification)                     │
│  - Interfaces (Ports)                                           │
│  - No external dependencies                            │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼─────────────────────────┐
│       Application Layer                                        │
│  - FineTuneModelUseCase                                │
│  - EvaluateModelUseCase                                 │
│  - Orchestration logic                                        │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      Infrastructure Layer                                       │
│  - LoRAAdapter (PEFT, transformers)                 │
│  - HuggingFaceInferenceAdapter                      │
│  - RobustJSONParser                                         │
│  - StandardMetricsRepository                           │
└─────────────────────────────────────────┘
```

### Layer Responsibilities

- **Domain**: Pure business logic, no framework dependencies
- **Application**: Use cases that orchestrate domain entities
- **Infrastructure**: Concrete implementations using external libraries

## 🚀 Features

- ✅ **LoRA Fine-Tuning**: Efficient parameter-efficient fine-tuning for consumer GPUs
- ✅ **4-bit/8-bit Quantization**: Train on RTX 4090 or similar
- ✅ **Multi-Model Support**: Llama 3 8B, Mistral 7B
- ✅ **Robust Parsing**: Handles malformed JSON from LLM outputs
- ✅ **Comprehensive Metrics**: F1, Precision, Recall, Accuracy
- ✅ **Detailed Reporting**: Qualitative evaluation with error analysis
- ✅ **Type-Safe**: Full mypy type hints
- ✅ **Configuration Management**: Pydantic-based configs

## 📋 Requirements

### Hardware
- **Recommended**: GPU with 16GB+ VRAM (RTX 4090, A100, etc.)
- **Minimum**: 8GB VRAM with 4-bit quantization
- **CPU**: Will work but very slow

### Software
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 20GB+ disk space (for model caching)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd jdc_system
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 📖 Usage

### Quick Start: Full Pipeline

Run both training and evaluation:

```bash
python main.py --mode full
```

### Training Only

Fine-tune the model on generated examples:

```bash
python main.py --mode train --epochs 5 --batch-size 4
```

### Evaluation Only

Evaluate an already fine-tuned model:

```bash
python main.py --mode eval
```

### Advanced Options

```bash
# Use Mistral instead of Llama 3
python main.py --model mistral

# Custom output directory
python main.py --output-dir ./my_results

# Adjust hyperparameters
python main.py --epochs 10 --batch-size 8 --learning-rate 1e-4
```

## 🎓 Knowledge Base

The system uses 5 principles for neurodiversity-aware classification:

| ID  | Principle | Label |
|-----|-----------|-------|
| P0  | Neutral Language | 0 (Not Ableist) |
| P1  | Pathologizing Language | 1 (Ableist) |
| P2  | Dehumanizing Metaphors | 1 (Ableist) |
| P3  | Stereotyping | 1 (Ableist) |
| P4  | Exclusionary Language | 1 (Ableist) |

### Mapping Logic

```python
# Deterministic mapping in domain/types.py
def is_ableist(self) -> bool:
    return self.principle_id in {"P1", "P2", "P3", "P4"}
```

## 📊 Output Structure

After running the pipeline, you'll find:

```
outputs/
├── checkpoints/           # Training checkpoints
├── fine_tuned_model/      # Final LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_config.json
├── evaluation_metrics.json # Quantitative metrics
└── evaluation_report.json  # Detailed qualitative report
```

### Metrics JSON Example

```json
{
  "precision": 0.8750,
  "recall": 0.9333,
  "f1_score": 0.9032,
  "accuracy": 0.9000,
  "true_positives": 14,
  "false_positives": 2,
  "true_negatives": 4,
  "false_negatives": 1,
  "total_examples": 20,
  "parsing_failures": 0
}
```

## 🧪 Testing with Real Data

To use your own AUTALIC dataset:

1. Prepare data in JSON format:

```json
[
  {
    "sentence": "Your target sentence here",
    "context_before": "Optional context",
    "context_after": "Optional context",
    "justification": {
      "principle_id": "P1",
      "justification_text": "Explanation...",
      "evidence_quote": "Relevant quote"
    },
    "label": 1
  }
]
```

2. Replace `MockDataLoader` with `FileBasedDataLoader` in `main.py`:

```python
from infrastructure import FileBasedDataLoader

data_loader = FileBasedDataLoader(data_dir=Path("./data"))
```

## 🔧 Customization

### Change Base Model

Edit `config.py`:

```python
from config import ModelType

config = SystemConfig(
    model_type=ModelType.MISTRAL_7B  # or ModelType.LLAMA3_8B
)
```

### Adjust LoRA Rank

Higher rank = more parameters = better fit but slower:

```python
config.lora_config.r = 32  # Default: 16
```

### Modify Quantization

```python
from config import QuantizationType

config.quantization_config.quantization_type = QuantizationType.BIT_8
```

## 🧰 Project Structure

```
jdc_system/
├── domain/                 # Pure domain logic
│   ├── types.py           # Entities (Principle, Justification, etc.)
│   └── interfaces.py      # Ports (LLMTrainer, InferenceEngine, etc.)
├── application/           # Use cases
│   └── services.py        # FineTuneModelUseCase, EvaluateModelUseCase
├── infrastructure/        # Adapters
│   ├── llm.py            # LoRAAdapter, HuggingFaceInferenceAdapter
│   ├── parsing.py        # RobustJSONParser
│   ├── metrics.py        # StandardMetricsRepository
│   └── data_loader.py    # MockDataLoader, FileBasedDataLoader
├── config.py             # Pydantic configuration models
├── main.py               # Entry point with dependency injection
└── requirements.txt      # Python dependencies
```

## 📝 Design Decisions

### Why Clean Architecture?

- **Testability**: Each layer can be tested independently
- **Flexibility**: Easy to swap implementations (e.g., switch from HuggingFace to JAX)
- **Maintainability**: Clear separation of concerns
- **Domain-Driven**: Business logic is protected from framework changes

### Why LoRA?

- **Efficiency**: Only trains ~0.1% of parameters
- **Memory**: Fits in consumer GPUs
- **Speed**: Faster training than full fine-tuning
- **Preservation**: Doesn't corrupt base model knowledge

### Why Deterministic Mapping?

- **Consistency**: Same justification always produces same label
- **Interpretability**: Clear rule-based logic
- **Debuggability**: Easy to trace classification decisions

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python main.py --batch-size 2

# Use 4-bit quantization (default)
# Or reduce LoRA rank in config.py
config.lora_config.r = 8
```

### Model Download Fails

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/disk
```

### Parsing Errors

The system uses `LenientJSONParser` by default which attempts recovery.
For strict validation:

```python
from infrastructure import StrictJSONParser

parser = StrictJSONParser()
```

## 📚 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Clean Architecture Book](https://www.amazon.com/Clean-Architecture-Craftsmans-Software-Structure/dp/0134494164)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Built for neurodiversity-aware NLP research
- Inspired by the AUTALIC dataset and neuro-symbolic AI principles
- Uses HuggingFace transformers, PEFT, and bitsandbytes

---

**Note**: This is a demonstration system using synthetic data. For production use with real AUTALIC data, replace `MockDataLoader` with actual dataset loading logic.
