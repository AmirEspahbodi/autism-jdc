"""
Main entry point for the JDC system.

This module demonstrates Clean Architecture principles through dependency
injection, wiring together domain entities, application use cases, and
infrastructure adapters.
"""

import argparse
import sys
from pathlib import Path

import torch

from src.application import EvaluateModelUseCase, FineTuneModelUseCase
from src.config import KnowledgeBaseConfig, SystemConfig
from src.infrastructure import (
    ConsoleReportGenerator,
    DetailedReportGenerator,
    HuggingFaceInferenceAdapter,
    LenientJSONParser,
    LoRAAdapter,
    MockDataLoader,
    StandardMetricsRepository,
)


def print_banner() -> None:
    """Print the system banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   Neuro-Symbolic Justification-Driven Classification (JDC) System   ║
║                                                                      ║
║   A Clean Architecture Implementation for Fine-Tuning LLMs          ║
║   with LoRA on Neurodiversity-Aware Language Classification         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_gpu() -> None:
    """Check GPU availability and print info."""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(
            f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("⚠ Warning: No GPU detected. Training will be slow.")
        print("  Consider using Google Colab or a GPU-enabled environment.")


def run_fine_tuning(config: SystemConfig) -> None:
    """Execute the fine-tuning pipeline.

    Args:
        config: System configuration.
    """
    print("\n" + "=" * 70)
    print("INITIALIZING FINE-TUNING PIPELINE")
    print("=" * 70)

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Dependency Injection: Wire up the adapters
    print("\n[Dependency Injection] Wiring adapters...")

    # Infrastructure Layer: Concrete implementations
    trainer = LoRAAdapter(config)
    data_loader = MockDataLoader(
        num_train_examples=50,  # Small for demonstration
        num_test_examples=20,
        seed=config.seed,
    )

    print("✓ Adapters initialized")

    # Application Layer: Use case
    fine_tune_use_case = FineTuneModelUseCase(
        trainer=trainer,
        data_loader=data_loader,
        config=config,
    )

    # Execute
    try:
        fine_tune_use_case.execute(use_validation=True)
        print("\n✓ Fine-tuning completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Fine-tuning failed: {str(e)}")
        return False


def run_evaluation(config: SystemConfig) -> None:
    """Execute the evaluation pipeline.

    Args:
        config: System configuration.
    """
    print("\n" + "=" * 70)
    print("INITIALIZING EVALUATION PIPELINE")
    print("=" * 70)

    # Check if model exists
    model_path = str(config.output_dir / "fine_tuned_model")
    if not Path(model_path).exists():
        print(f"\n✗ Error: Fine-tuned model not found at {model_path}")
        print("  Please run fine-tuning first: python main.py --mode train")
        return False

    # Dependency Injection: Wire up the adapters
    print("\n[Dependency Injection] Wiring adapters...")

    # Infrastructure Layer: Concrete implementations
    inference_engine = HuggingFaceInferenceAdapter(config, model_path)
    parser = LenientJSONParser()  # Use lenient parser for robustness
    metrics_repo = StandardMetricsRepository()
    report_generator = DetailedReportGenerator()
    console_reporter = ConsoleReportGenerator()
    data_loader = MockDataLoader(
        num_test_examples=20,
        seed=config.seed,
    )
    kb_config = KnowledgeBaseConfig()

    print("✓ Adapters initialized")

    # Application Layer: Use case
    evaluate_use_case = EvaluateModelUseCase(
        inference_engine=inference_engine,
        parser=parser,
        metrics_repo=metrics_repo,
        report_generator=report_generator,
        data_loader=data_loader,
        kb_config=kb_config,
    )

    # Execute
    try:
        output_base = str(config.output_dir / "evaluation")
        metrics = evaluate_use_case.execute(output_base)

        # Also print to console
        predictions_for_console = []  # Would need to extract from use case
        console_reporter.generate_report([], metrics)

        print("\n✓ Evaluation completed successfully!")
        print(f"\n📊 Results saved to:")
        print(f"   - Metrics: {output_base}_metrics.json")
        print(f"   - Report:  {output_base}_report.json")
        return True

    except Exception as e:
        print(f"\n✗ Evaluation failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def run_full_pipeline(config: SystemConfig) -> None:
    """Run both fine-tuning and evaluation.

    Args:
        config: System configuration.
    """
    print_banner()
    check_gpu()

    # Stage 3: Fine-tuning
    success = run_fine_tuning(config)
    if not success:
        print("\n✗ Pipeline aborted due to fine-tuning failure.")
        return

    # Stage 4: Evaluation
    success = run_evaluation(config)
    if not success:
        print("\n✗ Pipeline aborted due to evaluation failure.")
        return

    print("\n" + "=" * 70)
    print("✓ FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Neuro-Symbolic Justification-Driven Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (training + evaluation)
  python main.py --mode full

  # Only fine-tuning
  python main.py --mode train

  # Only evaluation (requires existing model)
  python main.py --mode eval

  # Use Mistral instead of Llama
  python main.py --model mistral

  # Custom output directory
  python main.py --output-dir ./my_results
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "train", "eval"],
        default="full",
        help="Pipeline mode: 'full' (train+eval), 'train' (only training), 'eval' (only evaluation)",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["llama3", "mistral"],
        default="llama3",
        help="Base model to use (default: llama3)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for models and results (default: ./outputs)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )

    args = parser.parse_args()

    # Create configuration
    from config import ModelType

    config = SystemConfig(
        model_type=ModelType.LLAMA3_8B
        if args.model == "llama3"
        else ModelType.MISTRAL_7B,
        output_dir=Path(args.output_dir),
    )

    # Override hyperparameters from CLI
    config.training_hyperparameters.num_epochs = args.epochs
    config.training_hyperparameters.batch_size = args.batch_size
    config.training_hyperparameters.learning_rate = args.learning_rate

    # Execute based on mode
    if args.mode == "full":
        run_full_pipeline(config)
    elif args.mode == "train":
        print_banner()
        check_gpu()
        run_fine_tuning(config)
    elif args.mode == "eval":
        print_banner()
        check_gpu()
        run_evaluation(config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
