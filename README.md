# Justification-Driven Classification (JDC)

A production-ready Python project for fine-tuning **Llama 3 8B** (via Unsloth + QLoRA) on the task of **ableism detection in autism/disability Reddit discourse**, using a symbolic Knowledge Base to generate structured JSON justifications.

---

## Architecture

The project follows **Onion / Hexagonal Architecture** with strict layer separation:

```
Domain (innermost) → Application → Infrastructure (outermost)
```

| Layer | Location | Responsibility |
|---|---|---|
| Domain | `src/domain/` | Entities, value objects, exceptions (zero external deps) |
| Application | `src/application/` | Use cases and abstract ports (interfaces) |
| Infrastructure | `src/infrastructure/` | Concrete adapters (Unsloth, sklearn, JSON files) |
| Entry Point | `src/main.py` | CLI wiring |

---

## Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with **CUDA 12.6+** (no CPU fallback for training)
- ~24 GB VRAM recommended (Llama 3 8B + QLoRA + bfloat16 activations)

---

## Installation

```bash
# 1. Clone / extract the project
cd jdc_project

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install Unsloth first (version-sensitive CUDA build)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Dataset

Place your three JSON dataset files in the `./dataset/` directory:

```
dataset/
├── train_dataset.json
├── validation_dataset.json
└── test_dataset.json
```

Each file must be a JSON array where each element has:
```json
{
  "id":           "unique-string-id",
  "input_prompt": "Full prompt including KB and context…",
  "model_output": "{\"justification_reasoning\": \"…\", \"evidence_quote\": \"…\", \"principle_id\": \"P1\", \"principle_name\": \"Medical Model Framing\", \"is_ableist\": true}"
}
```

---

## Configuration

Edit `config/config.yaml` to customise model, LoRA, training, and evaluation settings.

Key parameters:

| Key | Default | Description |
|---|---|---|
| `model.name` | `unsloth/Meta-Llama-3-8B-Instruct` | HuggingFace model ID |
| `model.max_seq_length` | `2048` | Max tokens per sample |
| `lora.r` | `16` | LoRA rank |
| `training.num_train_epochs` | `3` | Training epochs |
| `training.learning_rate` | `2e-4` | Learning rate |
| `evaluation.checkpoint_path` | `./outputs/checkpoints/best_model` | Checkpoint for eval/inference |

---

## Usage

All commands are run from the project root:

```bash
# Training
python -m src.main train

# Evaluation (both validation and test splits)
python -m src.main evaluate

# Evaluation (single split)
python -m src.main evaluate --splits validation

# Single-sample inference from a prompt file
python -m src.main infer --prompt-file my_prompt.txt

# Single-sample inference inline
python -m src.main infer --prompt "Your full prompt string here…"

# Custom config path
python -m src.main --config /path/to/my_config.yaml train
```

---

## Output Files

All outputs land in `./outputs/`:

```
outputs/
├── checkpoints/           ← Saved LoRA adapter weights (per epoch + best)
├── logs/
│   └── run.log            ← Full loguru log file
├── evaluation_validation_YYYYMMDD_HHMMSS.json
├── evaluation_test_YYYYMMDD_HHMMSS.json
├── disagreement_report_validation_YYYYMMDD_HHMMSS.json
└── disagreement_report_test_YYYYMMDD_HHMMSS.json
```

### Evaluation JSON Schema

```json
{
  "split": "validation",
  "f1": 0.9123,
  "precision": 0.9200,
  "recall": 0.9050,
  "accuracy": 0.9180,
  "principle_accuracy": 0.8750,
  "confusion_matrix": [[120, 8], [10, 112]],
  "classification_report": "…full sklearn report…"
}
```

### Disagreement Report Schema

```json
{
  "split": "validation",
  "total_samples": 250,
  "total_disagreements": 18,
  "false_positives": 8,
  "false_negatives": 10,
  "principle_pair_frequency": {"P1 -> P0": 5, "P0 -> P3": 3},
  "disagreements": [
    {
      "id": "sample-001",
      "input_prompt": "…first 500 chars…",
      "ground_truth_principle": "P1",
      "ground_truth_label": true,
      "predicted_principle": "P0",
      "predicted_label": false,
      "generated_justification": "…",
      "generated_evidence": "…",
      "error_type": "FALSE_NEGATIVE"
    }
  ]
}
```

---

## Knowledge Base Principles

| ID | Name | Description |
|---|---|---|
| P0 | Not Ableist | Neutral experience, community support, or reclaimed language |
| P1 | Medical Model Framing | Defines autism as disease, tragedy, or deficit |
| P2 | Eugenicist Hierarchy | Functioning labels or value based on societal utility |
| P3 | Promotion of Harmful Tropes | Debunked stereotypes like vaccine-autism links |
| P4 | Centering Neurotypical Perspectives | Frames through non-autistic inconvenience |

**Label derivation rule (hardcoded):**
- `principle_id ∈ {P1, P2, P3, P4}` → `is_ableist = True`
- `principle_id == P0` → `is_ableist = False`

The model **never directly predicts** `is_ableist` — it is always derived from `principle_id`.

---

## VRAM Budget (24 GB GPU)

| Component | Estimated VRAM |
|---|---|
| Llama 3 8B base weights (4-bit NF4) | ~4.5 GB |
| LoRA adapter weights (bf16, rank 16) | ~0.15 GB |
| 8-bit Adam optimizer states | ~1.0 GB |
| Activations (seq_len=2048, GC enabled) | ~4–6 GB |
| **Total (estimated)** | **~10–12 GB** |

Gradient checkpointing (Unsloth's own implementation) is enabled to trade compute for reduced activation memory.

---

## Type Checking

```bash
mypy src/ --strict
```

---

## Project Structure

```
jdc_project/
├── config/config.yaml
├── dataset/                   ← Place your JSON datasets here
├── outputs/
│   ├── checkpoints/
│   └── logs/
├── src/
│   ├── main.py                ← CLI entry point
│   ├── container.py           ← DI container
│   ├── domain/
│   │   ├── entities.py        ← Pydantic v2 data models
│   │   ├── value_objects.py   ← PrincipleID enum + derive_label()
│   │   └── exceptions.py      ← Custom exceptions
│   ├── application/
│   │   ├── ports/             ← Abstract interfaces
│   │   └── use_cases/         ← Train / Evaluate / Infer orchestration
│   └── infrastructure/
│       ├── adapters/          ← JSON repo, Unsloth service, sklearn evaluator
│       └── config/            ← OmegaConf loader
├── requirements.txt
├── pyproject.toml
└── README.md
```
