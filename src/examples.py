"""
Example: Programmatic usage of the JDC system.

This script demonstrates how to use the JDC system programmatically
rather than through the CLI.
"""

from pathlib import Path

from src.application import EvaluateModelUseCase, FineTuneModelUseCase
from src.config import KnowledgeBaseConfig, SystemConfig
from src.domain import Justification, LabeledExample
from src.infrastructure import (
    DetailedReportGenerator,
    HuggingFaceInferenceAdapter,
    LenientJSONParser,
    LoRAAdapter,
    MockDataLoader,
    StandardMetricsRepository,
)


def example_create_custom_training_data():
    """Example: Create custom training examples."""

    # Create a custom labeled example
    example = LabeledExample(
        sentence="People with autism are broken and need to be fixed.",
        context_before="There are many misconceptions about neurodiversity.",
        context_after="This harmful view is being challenged by advocates.",
        ground_truth_justification=Justification(
            principle_id="P1",
            justification_text=(
                "This sentence pathologizes autism by using the word 'broken' "
                "and framing it as something requiring fixing, rather than "
                "recognizing it as a neurological difference."
            ),
            evidence_quote="broken and need to be fixed",
        ),
        ground_truth_label=1,  # Ableist
    )

    print("Custom Training Example:")
    print(f"  Sentence: {example.sentence}")
    print(f"  Label: {example.ground_truth_label}")
    print(f"  Principle: {example.ground_truth_justification.principle_id}")
    print(f"  Justification: {example.ground_truth_justification.justification_text}")

    return [example]


def example_configure_system():
    """Example: Custom system configuration."""

    from config import ModelType, QuantizationType

    # Create custom configuration
    config = SystemConfig(
        model_type=ModelType.LLAMA3_8B,
        output_dir=Path("./custom_outputs"),
        seed=12345,
    )

    # Customize LoRA settings
    config.lora_config.r = 32  # Increase rank for more capacity
    config.lora_config.lora_alpha = 64

    # Customize training
    config.training_hyperparameters.num_epochs = 5
    config.training_hyperparameters.batch_size = 2
    config.training_hyperparameters.learning_rate = 1e-4

    # Use 8-bit quantization instead of 4-bit
    config.quantization_config.quantization_type = QuantizationType.BIT_8

    print("Custom Configuration:")
    print(f"  Model: {config.model_type.value}")
    print(f"  LoRA Rank: {config.lora_config.r}")
    print(f"  Epochs: {config.training_hyperparameters.num_epochs}")
    print(f"  Quantization: {config.quantization_config.quantization_type.value}")

    return config


def example_run_inference_only():
    """Example: Run inference on a pre-trained model."""

    # Assume model is already trained
    config = SystemConfig()
    model_path = "./outputs/fine_tuned_model"

    if not Path(model_path).exists():
        print("⚠ Model not found. Please train first using main.py")
        return

    # Load inference engine
    inference_engine = HuggingFaceInferenceAdapter(config, model_path)

    # Prepare knowledge base
    kb_config = KnowledgeBaseConfig()
    kb_text = kb_config.get_all_principles_text()

    # Analyze a new sentence
    test_sentence = "Autistic people bring unique perspectives to problem-solving."

    print(f"\nAnalyzing: {test_sentence}")
    print("-" * 60)

    # Generate justification
    raw_output = inference_engine.generate_justification(
        sentence=test_sentence,
        knowledge_base_text=kb_text,
    )

    print(f"Raw output:\n{raw_output}")

    # Parse justification
    parser = LenientJSONParser()
    try:
        justification = parser.parse(raw_output)
        label = justification.to_label()

        print(f"\nParsed Justification:")
        print(f"  Principle: {justification.principle_id}")
        print(f"  Label: {label} ({'Ableist' if label == 1 else 'Not Ableist'})")
        print(f"  Reasoning: {justification.justification_text}")
        print(f"  Evidence: {justification.evidence_quote}")

    except Exception as e:
        print(f"Parsing failed: {e}")


def example_custom_use_case():
    """Example: Create a custom use case for batch prediction."""

    config = SystemConfig()
    model_path = "./outputs/fine_tuned_model"

    if not Path(model_path).exists():
        print("⚠ Model not found. Please train first using main.py")
        return

    # Set up components
    inference_engine = HuggingFaceInferenceAdapter(config, model_path)
    parser = LenientJSONParser()
    kb_config = KnowledgeBaseConfig()
    kb_text = kb_config.get_all_principles_text()

    # Test sentences
    test_sentences = [
        "People with ADHD bring creative energy to teams.",
        "Autistic people are like robots without emotions.",
        "Dyslexic students read everything backwards.",
    ]

    print("\nBatch Prediction Example:")
    print("=" * 60)

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. Analyzing: {sentence}")

        try:
            # Generate and parse
            raw_output = inference_engine.generate_justification(
                sentence=sentence,
                knowledge_base_text=kb_text,
            )
            justification = parser.parse(raw_output)
            label = justification.to_label()

            # Display result
            print(f"   Result: {label} ({justification.principle_id})")
            print(f"   Reasoning: {justification.justification_text[:100]}...")

        except Exception as e:
            print(f"   Error: {e}")


def main():
    """Run all examples."""

    print("\n" + "=" * 70)
    print("JDC System - Programmatic Usage Examples")
    print("=" * 70)

    print("\n[Example 1] Creating Custom Training Data")
    print("-" * 70)
    example_create_custom_training_data()

    print("\n[Example 2] Custom Configuration")
    print("-" * 70)
    example_configure_system()

    print("\n[Example 3] Inference Only")
    print("-" * 70)
    example_run_inference_only()

    print("\n[Example 4] Custom Batch Prediction")
    print("-" * 70)
    example_custom_use_case()

    print("\n" + "=" * 70)
    print("Examples Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
