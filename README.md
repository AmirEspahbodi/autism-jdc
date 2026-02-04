# Neuro-Symbolic Justification-Driven Classification (JDC) System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Architecture](https://img.shields.io/badge/Architecture-Clean%20Onion-purple)

A production-grade, neuro-symbolic AI system designed to detect ableist language. Unlike standard "black box" classifiers, the JDC system is **interpretable by design**: it fine-tunes Large Language Models (LLMs) to generate structured, JSON-based reasoning before assigning a classification label.

## 🧠 Core Concept

The system moves beyond simple binary classification by enforcing a **Justification-First** workflow:

1.  **Symbolic Grounding**: The model is grounded in a Knowledge Base of 5 specific neurodiversity principles (P0-P4).
2.  **Reasoning Generation**: The LLM generates a JSON object containing the violated principle, a textual justification, and the specific evidence quote.
3.  **Deterministic Labeling**: The final classification (Ableist/Not Ableist) is derived deterministically from the generated principle ID.

## 🏗️ System Architecture

The project is built on **Clean Architecture** (Onion Architecture) principles to ensure maintainability and testability.

```mermaid
graph TD
    subgraph Domain ["Domain Layer (Pure Python)"]
        Entities[Entities: Principle, Justification]
        Interfaces[Interfaces: LLMTrainer, DataLoader]
    end

    subgraph Application ["Application Layer"]
        Services[Use Cases: FineTuneModel, EvaluateModel]
    end

    subgraph Infrastructure ["Infrastructure Layer"]
        LoRA[LoRAAdapter (PEFT/Transformers)]
        Parser[RobustJSONParser]
        Metrics[StandardMetricsRepository]
        Loader[PreformattedDataLoader]
    end

    Application --> Domain
    Infrastructure --> Domain
    Infrastructure --> Application
    ## 🚀 Key Features
    
    * **QLoRA Fine-Tuning**: Efficient 4-bit/8-bit quantized training support for Llama 3 8B and Mistral 7B on consumer GPUs.
    * **Instruction Tuning**: Uses `trl`'s `SFTTrainer` with `DataCollatorForCompletionOnlyLM` to optimize only the model's reasoning generation.
    * **Robust JSON Parsing**: Specialized parsers capable of recovering valid JSON from malformed LLM outputs (markdown fences, trailing text).
    * **Strict Metrics**: Calculates F1, Precision, and Recall while rigorously penalizing parsing failures as incorrect predictions.
    * **Type-Safe Config**: Centralized configuration management using Pydantic.
```

## 🛠️ Installation

### Prerequisites
* Python 3.10+
* CUDA-enabled GPU (Recommended: 16GB+ VRAM for training)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/amirespahbodi/autism-jdc.git](https://github.com/amirespahbodi/autism-jdc.git)
    cd autism-jdc
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
## 📊 Data Preparation

The system supports a **Pre-formatted SFT** data format. This allows you to handle prompt templates offline for maximum control.

Create a file at `data/dataset.json` with the following structure:

```json
[
  {
    "input_prompt": "<|start_header_id|>user<|end_header_id|>\n\nAnalyze this sentence...\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "model_output": "{\n  \"principle_id\": \"P1\",\n  \"justification_text\": \"The text frames autism as a tragedy.\",\n  \"evidence_quote\": \"suffering from autism\"\n}"
  }
]
```
