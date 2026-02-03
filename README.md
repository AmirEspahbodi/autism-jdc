# Justification-Driven Classification (JDC) Framework

A Neuro-Symbolic framework for detecting ableism in text using fine-tuned LLMs with structured reasoning.

## Overview

This project implements Steps 3 (Fine-Tuning) and 4 (Evaluation) of the JDC framework:
- **Step 3**: Fine-tune Llama 3 8B or Mistral 7B using QLoRA (4-bit quantization)
- **Step 4**: Evaluate the model with resilient JSON parsing and comprehensive metrics

## Project Structure

```
jdc_framework/
├── configs/
│   ├── train_config.yaml      # Training hyperparameters
│   └── eval_config.yaml       # Evaluation settings
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py         # Data loading and formatting
│   ├── modeling/
│   │   ├── __init__.py
│   │   └── model_loader.py    # Model/tokenizer setup with QLoRA
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py         # Custom training logic
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── generator.py       # Inference engine
│   │   └── parser.py          # Resilient JSON parser
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # Configuration schemas
│       └── metrics.py         # Evaluation metrics
├── scripts/
│   ├── train.py               # Training entry point
│   └── evaluate.py            # Evaluation entry point
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training (Step 3)

```bash
python scripts/train.py --config configs/train_config.yaml
```

### Evaluation (Step 4)

```bash
python scripts/evaluate.py --config configs/eval_config.yaml
```

## Key Features

- **QLoRA Fine-Tuning**: 4-bit quantization for efficient training on consumer GPUs
- **Resilient JSON Parsing**: Handles malformed model outputs gracefully
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Qualitative Analysis**: Disagreement logging for error analysis
- **Modular Architecture**: Clean separation of concerns for maintainability

## Configuration

Edit YAML files in `configs/` to adjust:
- Model selection (Llama 3 8B / Mistral 7B)
- LoRA hyperparameters (rank, alpha, dropout)
- Training settings (learning rate, batch size, epochs)
- Quantization settings (4-bit, 8-bit)

## Output

Training produces:
- `checkpoints/`: Model adapter weights
- `logs/`: Training logs

Evaluation produces:
- `results.json`: Aggregate metrics
- `disagreements.csv`: Cases where model disagrees with ground truth
- `predictions.json`: Full predictions with justifications
